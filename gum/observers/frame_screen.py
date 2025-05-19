# frame_screen.py
from __future__ import annotations
###############################################################################
# Imports                                                                     #
###############################################################################

from dotenv import load_dotenv
load_dotenv()

# ─ Standard library ──────────────────────────────────────────────────────────
import asyncio
import json
import logging
import os
import shlex
import subprocess
import tempfile
import time
from datetime import timedelta
from importlib.resources import files as get_package_file
from typing import Any, Dict, Optional

# ─ Third-party ───────────────────────────────────────────────────────────────
import mss
from PIL import Image
from pynput import mouse, keyboard
from google import genai
from google.genai import types as gtypes

# ─ Local (adjust the import paths to match your project layout) ──────────────
from .observer import Observer           # your existing base-class
from .window_geometry import is_app_visible
from ..schemas import Update             # whatever you already use

###############################################################################
# Helper functions                                                            #
###############################################################################

def _sec_to_mmss(sec: float) -> str:
    mins, secs = divmod(int(sec), 60)
    return f"{mins:02d}:{secs:02d}"


def _build_video_ffmpeg(jpg_paths: list[str], out_path: str,
                        fps: int = 1, crf: int = 28, preset: str = "veryfast") -> None:
    """
    JPEG → MP4 with ffmpeg concat (libx264, yuv420p) @ *fps*.
    """
    txtfile = out_path + ".list"
    with open(txtfile, "w") as fh:
        for p in jpg_paths:
            fh.write(f"file '{p}'\n")

    cmd = (
        f"ffmpeg -y -r {fps} -f concat -safe 0 -i {txtfile} "
        f"-c:v libx264 -pix_fmt yuv420p -crf {crf} -preset {preset} {out_path}"
    )
    subprocess.run(shlex.split(cmd), check=True)
    os.remove(txtfile)

###############################################################################
# Realtime screen & keystroke observer                                        #
###############################################################################

class FrameScreen(Observer):
    """Capture screen frames + low-latency input-events, periodically emit
    a 1 fps compressed MP4 and ask Gemini for dense captions.
    """

    # ────────────── runtime knobs
    _CAPTURE_FPS: int = 10              # refresh live frame buffers
    _MOVE_IDLE_SEC: float = 0.35        # idle gap ending a "move" gesture
    _SCROLL_IDLE_SEC: float = 0.35      # idle gap ending a "scroll" gesture
    _FRAME_IDLE_SEC: float = 1.00       # idle gap before we snapshot screen

    _MON_START: int = 1                 # first real display in mss.monitors
    _VIDEO_INTERVAL_SEC: int = 30       # flush when buffer ≥ this
    _VIDEO_FPS: int = 1                 # constructed video fps
    _VIDEO_CRF: int = 28                # ffmpeg compression (lower → bigger)
    _VIDEO_PRESET: str = "veryfast"     # ffmpeg speed / compression trade-off

    # ───────────────────────── construction
    def __init__(
        self,
        screenshots_dir: str = "~/.cache/gum/screenshots",
        keystrokes_path: str = "~/.cache/gum/keystrokes.log",  # JSONL
        skip_when_visible: Optional[str | list[str]] = None,
        debug: bool = False,
    ) -> None:

        # screenshot output directory
        self.screens_dir = os.path.abspath(os.path.expanduser(screenshots_dir))
        os.makedirs(self.screens_dir, exist_ok=True)

        # low-latency input-event log (JSONL)
        self.keystrokes_path = os.path.abspath(os.path.expanduser(keystrokes_path))
        os.makedirs(os.path.dirname(self.keystrokes_path), exist_ok=True)
        self._keys_fh = open(self.keystrokes_path, "a", buffering=1)

        # pause capture if any window in *guard* set is visible
        self._guard = {skip_when_visible} if isinstance(skip_when_visible, str) else set(skip_when_visible or [])
        self.debug = debug

        # per-monitor live frame buffers
        self._frames: Dict[int, Any] = {}
        self._frame_lock = asyncio.Lock()

        # mouse-press bookkeeping (down→up duration)
        self._press_state: Optional[dict] = None
        self._button_pressed: Optional[str] = None

        # debounced gesture state/handles
        self._move_state: Optional[dict] = None
        self._move_handle: Optional[asyncio.TimerHandle] = None
        self._scroll_state: Optional[dict] = None
        self._scroll_handle: Optional[asyncio.TimerHandle] = None

        # debounced screenshot handle
        self._shot_handle: Optional[asyncio.TimerHandle] = None

        # dense caption base-prompt
        self._dense_caption_prompt = self._load_prompt("dense_caption.txt")

        # video-flush buffer
        self._video_buf: list[tuple[float, str]] = []      # [(abs_ts, jpg_path)…]
        self._last_video_start: float | None = None

        super().__init__()

    # ───────────────────────── prompt loader
    @staticmethod
    def _load_prompt(fname: str) -> str:
        return get_package_file("gum.prompts.screen").joinpath(fname).read_text()

    # ───────────────────────── misc helpers
    def _skip(self) -> bool:
        """True if we should pause because a guarded window is visible."""
        return is_app_visible(self._guard) if self._guard else False

    def _reset_timer(self, attr: str, delay: float, coro):
        """Cancel any existing timer stored at *attr* and start a new one."""
        loop = asyncio.get_running_loop()
        if (h := getattr(self, attr)):
            h.cancel()
        setattr(self, attr, loop.call_later(delay, lambda: asyncio.create_task(coro())))

    # ───────────────────────── screenshot + buffering
    async def _save_frame(self, frame, tag: str) -> str:
        ts = time.time()
        path = os.path.join(self.screens_dir, f"{ts:.5f}_{tag}.jpg")
        await asyncio.to_thread(
            Image.frombytes("RGB", (frame.width, frame.height), frame.rgb).save,
            path, "JPEG", quality=90,
        )

        # ─── push to video buffer ───
        self._video_buf.append((ts, path))
        if self._last_video_start is None:
            self._last_video_start = ts

        if ts - self._last_video_start >= self._VIDEO_INTERVAL_SEC:
            buf_copy = self._video_buf[:]
            self._video_buf.clear()
            self._last_video_start = None
            asyncio.create_task(self._finalize_video(buf_copy))

        return path

    # ───────────────────────── key/gesture logging
    async def _log_event(self, etype: str, payload: dict) -> None:
        event = {"ts": time.time(), "type": etype, **payload}
        line = json.dumps(event, separators=(",", ":")) + "\n"
        await asyncio.to_thread(self._keys_fh.write, line)
        await asyncio.to_thread(self._keys_fh.flush)
        await self.update_queue.put(Update(content=line.rstrip(), content_type="input_text"))

    # ───────────────────────── video → gemini worker
    async def _finalize_video(self, frames: list[tuple[float, str]]) -> None:
        """Build MP4, craft prompt, ask Gemini, emit JSON."""
        if not frames:
            return
        t0, t1 = frames[0][0], frames[-1][0]

        # 1. Build compressed MP4 ------------------------------------------------
        tmp_mp4 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        await asyncio.to_thread(
            _build_video_ffmpeg,
            [p for _, p in frames],
            tmp_mp4,
            self._VIDEO_FPS,
            self._VIDEO_CRF,
            self._VIDEO_PRESET,
        )

        # 2. Collect keystrokes within window -----------------------------------
        keys_between: list[str] = []
        with open(self.keystrokes_path) as fh:
            for line in fh:
                ev = json.loads(line)
                if t0 <= ev["ts"] <= t1:
                    rel = _sec_to_mmss(ev["ts"] - t0)
                    keys_between.append(f"{rel} {ev['type']} {ev.get('key','')}")

        # 3. Craft dense-caption prompt -----------------------------------------
        prompt_txt = self._dense_caption_prompt.replace(
            "{keystrokes}", "\n".join(keys_between)
        )

        # 4. Ask Gemini ----------------------------------------------------------
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        vid_file = client.files.upload(file=tmp_mp4)

        cfg = gtypes.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties={
                    "timestamp": gtypes.Schema(type=gtypes.Type.STRING),
                    "caption":   gtypes.Schema(type=gtypes.Type.STRING),
                },
            ),
        )

        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[vid_file, prompt_txt],
            config=cfg,
        )

        # 5. Emit caption JSON downstream ---------------------------------------
        await self.update_queue.put(Update(content=resp.text, content_type="application/json"))

        # Cleanup
        os.remove(tmp_mp4)

    # ────────────────────────── main async worker (unchanged except for calls)
    async def _worker(self) -> None:
        log = logging.getLogger("Screen")
        if self.debug:
            logging.basicConfig(level=logging.INFO,
                                format="%(asctime)s [Screen] %(message)s",
                                datefmt="%H:%M:%S")
        else:
            log.addHandler(logging.NullHandler())
            log.propagate = False

        loop = asyncio.get_running_loop()

        with mss.mss() as sct:
            mons = sct.monitors[self._MON_START:]

            # ─── OS-thread → asyncio trampolines ───
            mouse_listener = mouse.Listener(
                on_click=lambda x, y, btn, prs: asyncio.run_coroutine_threadsafe(
                    mouse_click(x, y, btn, prs), loop),
                on_move=lambda x, y: asyncio.run_coroutine_threadsafe(
                    mouse_move(x, y), loop),
                on_scroll=lambda x, y, dx, dy: asyncio.run_coroutine_threadsafe(
                    mouse_scroll(x, y, dx, dy), loop),
            )
            mouse_listener.start()

            keyboard_listener = keyboard.Listener(
                on_press=lambda k: asyncio.run_coroutine_threadsafe(key_event(k, True), loop),
                on_release=lambda k: asyncio.run_coroutine_threadsafe(key_event(k, False), loop),
            )
            keyboard_listener.start()

            # ─── flush helpers for debounced gestures ───
            async def flush_move():
                if self._move_state and not self._skip():
                    ms = self._move_state
                    await self._log_event(
                        "mouse_move",
                        {
                            "sx": ms["sx"], "sy": ms["sy"],
                            "ex": ms["ex"], "ey": ms["ey"],
                            "mon": ms["mon"],
                            "dt": ms["end"] - ms["start"],
                            **({"button": ms["button"]} if ms["button"] else {}),
                        },
                    )
                    self._schedule_screenshot(ms["mon"], sct, mons)
                self._move_state = None

            async def flush_scroll():
                if self._scroll_state and not self._skip():
                    ss = self._scroll_state
                    await self._log_event(
                        "mouse_scroll",
                        {
                            "sx": ss["sx"], "sy": ss["sy"],
                            "ex": ss["ex"], "ey": ss["ey"],
                            "dx": ss["dx"], "dy": ss["dy"],
                            "mon": ss["mon"],
                            "dt": ss["end"] - ss["start"],
                        },
                    )
                    self._schedule_screenshot(ss["mon"], sct, mons)
                self._scroll_state = None

            # ─── mouse event coroutines ───
            async def mouse_move(x: float, y: float):
                idx = self._mon_for(x, y, mons)
                if self._skip() or idx is None:
                    return

                now = time.time()
                if self._move_state is None:
                    self._move_state = {
                        "sx": x, "sy": y, "ex": x, "ey": y,
                        "start": now, "end": now,
                        "mon": idx,
                        "button": self._button_pressed,
                    }
                else:
                    self._move_state.update({"ex": x, "ey": y, "end": now})

                self._reset_timer("_move_handle", self._MOVE_IDLE_SEC, flush_move)

            async def mouse_scroll(x: float, y: float, dx: int, dy: int):
                idx = self._mon_for(x, y, mons)
                if self._skip() or idx is None:
                    return

                now = time.time()
                if self._scroll_state is None:
                    self._scroll_state = {
                        "sx": x, "sy": y, "ex": x, "ey": y,
                        "dx": dx, "dy": dy,
                        "start": now, "end": now,
                        "mon": idx,
                    }
                else:
                    self._scroll_state.update({
                        "ex": x, "ey": y,
                        "dx": self._scroll_state["dx"] + dx,
                        "dy": self._scroll_state["dy"] + dy,
                        "end": now,
                    })

                self._reset_timer("_scroll_handle", self._SCROLL_IDLE_SEC, flush_scroll)

            async def mouse_click(x: float, y: float, btn: mouse.Button, pressed: bool):
                button_name = (
                    "left" if btn == mouse.Button.left
                    else "right" if btn == mouse.Button.right
                    else "middle"
                )
                idx = self._mon_for(x, y, mons)
                if self._skip() or idx is None:
                    return

                if pressed:
                    # mouse DOWN — duration bookkeeping only
                    self._press_state = {
                        "button": button_name, "x": x, "y": y,
                        "mon": idx, "ts": time.time(),
                    }
                    self._button_pressed = button_name
                else:
                    # mouse UP — log + schedule screenshot after debounce
                    self._button_pressed = None
                    dt = 0.0
                    if self._press_state and self._press_state["button"] == button_name:
                        dt = time.time() - self._press_state["ts"]
                    await self._log_event(
                        "mouse_click",
                        {"button": button_name, "x": x, "y": y, "mon": idx, "dt": dt},
                    )
                    self._schedule_screenshot(idx, sct, mons)
                    self._press_state = None

            # ─── key event coroutine ───
            async def key_event(k, is_down: bool):
                if self._skip():
                    return
                try:
                    k_str = k.char if hasattr(k, "char") else str(k)
                except Exception:
                    k_str = str(k)
                await self._log_event("key_down" if is_down else "key_up", {"key": k_str})

            # ─── helper for delayed screenshots ───
            def _schedule_screenshot(idx: int, sct_: mss.mss, mons_: list[dict]):
                async def take_shot():
                    if self._skip():
                        return
                    frame = await asyncio.to_thread(sct_.grab, mons_[idx - 1])
                    path = await self._save_frame(frame, "after")
                    await self._log_event("frame", {"mon": idx, "path": path})
                self._reset_timer("_shot_handle", self._FRAME_IDLE_SEC, take_shot)

            # ─── monitor-lookup utility ───
            def _mon_for_local(x: float, y: float):
                return self._mon_for(x, y, mons)

            self._mon_for = _mon_for_local     # bind for inner coroutines

            # ─── main capture loop ───
            log.info("Screen observer started — guarding %s", self._guard or "∅")
            while self._running:
                t0 = time.time()
                for idx, m in enumerate(mons, 1):
                    frame = await asyncio.to_thread(sct.grab, m)
                    async with self._frame_lock:
                        self._frames[idx] = frame
                await asyncio.sleep(max(0, (1 / self._CAPTURE_FPS) - (time.time() - t0)))

            # ─── shutdown ───
            mouse_listener.stop()
            keyboard_listener.stop()
            for h in (self._move_handle, self._scroll_handle, self._shot_handle):
                if h:
                    h.cancel()
            self._keys_fh.close()
