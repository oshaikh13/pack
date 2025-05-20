from __future__ import annotations
###############################################################################
# Imports                                                                     #
###############################################################################

# — Standard library —
import asyncio
import json
import logging
import os
import time
from collections import deque
from importlib.resources import files as get_package_file
from typing import Any, Dict, Optional

# — Third‑party —
import mss
from PIL import Image
from pynput import mouse, keyboard
from moviepy import ImageSequenceClip
from google import genai
from google.genai import types as genai_types

# — Local —
from .observer import Observer
from ..schemas import Update

from .window_geometry import is_app_visible as _is_app_visible

###############################################################################
# Screen observer powered by Gemini                                           #
###############################################################################

class VideoScreen(Observer):
    """
    Capture before/after screenshots around user interactions, bundle them
    into short silent MP4 clips, and send the clip to Gemini for captioning.
    Additionally, log *all* raw mouse and keyboard events to a newline‑
    delimited JSON (JSONL) file for downstream analysis.
    Heavy CPU / blocking I/O work runs in background threads via
    `asyncio.to_thread`.
    """

    # ----------------------------- tuning knobs -----------------------------
    _CAPTURE_FPS: int = 10              # live "before-frame" refresh rate
    _DEBOUNCE_SEC: int = 1              # wait this long after an event
    _MON_START: int = 1                 # first real display in mss' list

    _SCREENSHOTS_PER_VIDEO: int = 30    # stitch when this many frames queued
    _SECONDS_PER_SCREENSHOT: int = 1    # still frame duration inside clip

    # ----------------------------- construction -----------------------------
    def __init__(
        self,
        screenshots_dir: str = "~/.cache/gum/screenshots",
        events_file: str = "~/.cache/gum/keystrokes.jsonl",
        skip_when_visible: str | list[str] | None = None,
        transcription_prompt: str | None = None,
        history_k: int = 10,
        debug: bool = False,
    ) -> None:

        # output dir --------------------------------------------------------
        self.screens_dir = os.path.abspath(os.path.expanduser(screenshots_dir))
        os.makedirs(self.screens_dir, exist_ok=True)

        # event log ---------------------------------------------------------
        self.events_path = os.path.abspath(os.path.expanduser(events_file))
        os.makedirs(os.path.dirname(self.events_path), exist_ok=True)
        # lazy open — create empty file if missing so tail -f works straightaway
        open(self.events_path, "a", encoding="utf‑8").close()

        # guard list --------------------------------------------------------
        self._guard = {skip_when_visible} if isinstance(skip_when_visible, str) else set(skip_when_visible or [])
        self.transcription_prompt = transcription_prompt or self._load_prompt("dense_caption.txt")
        self.debug = debug

        # shared state ------------------------------------------------------
        self._frames: Dict[int, Any] = {}
        self._frame_lock = asyncio.Lock()

        self._queued_paths: list[str] = []
        self._history: deque[str] = deque(maxlen=max(0, history_k))

        self._pending_event: Optional[dict] = None
        self._debounce_handle: Optional[asyncio.TimerHandle] = None

        # init base class ---------------------------------------------------
        super().__init__()

    # ----------------------------- static helpers --------------------------
    @staticmethod
    def _mon_for(x: float, y: float, mons: list[dict]) -> Optional[int]:
        for idx, m in enumerate(mons, 1):
            if m["left"] <= x < m["left"] + m["width"] and m["top"] <= y < m["top"] + m["height"]:
                return idx
        return None

    # ----------------------------- event logger ---------------------------
    async def _log_event(self, payload: dict) -> None:
        await asyncio.to_thread(
            lambda p: open(self.events_path, "a", encoding="utf‑8").write(json.dumps(p, ensure_ascii=False) + "\n"),
            payload,
        )

    # ----------------------------- I/O helpers -----------------------------
    async def _save_frame(self, frame, tag: str) -> str:
        ts   = f"{time.time():.5f}"
        path = os.path.join(self.screens_dir, f"{ts}_{tag}.jpg")
        await asyncio.to_thread(
            Image.frombytes("RGB", (frame.width, frame.height), frame.rgb).save,
            path,
            "JPEG",
            quality=90,
        )
        return path

    @staticmethod
    def _load_prompt(fname: str) -> str:
        return get_package_file("gum.prompts.screen").joinpath(fname).read_text()

    async def _create_video_from_frames(self, paths: list[str]) -> str:
        """Build a silent MP4 using moviepy.  Runs in a worker thread."""

        def _build() -> str:
            clip = ImageSequenceClip(paths, fps=1 / self._SECONDS_PER_SCREENSHOT)
            out = os.path.join(self.screens_dir, f"{time.time():.5f}.mp4")
            clip.write_videofile(out, codec="libx264", audio=False, logger=None)
            clip.close()
            return out
        
        # Offload the blocking MoviePy call into a real thread
        return await asyncio.to_thread(_build)

    def _call_gemini(self, prompt: str, video_path: str) -> str:
        generate_content_config = genai.types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=genai.types.Schema(
                type = genai.types.Type.OBJECT,
                required = ["transcriptions"],
                properties = {
                    "transcriptions": genai.types.Schema(
                        type = genai.types.Type.ARRAY,
                        items = genai.types.Schema(
                            type = genai.types.Type.OBJECT,
                            required = ["caption", "timestamp"],
                            properties = {
                                "caption": genai.types.Schema(
                                    type = genai.types.Type.STRING,
                                ),
                                "timestamp": genai.types.Schema(
                                    type = genai.types.Type.STRING,
                                ),
                            },
                        ),
                    ),
                },
            ),
        )

        client   = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        with open(video_path, "rb") as f:
            video_bytes = f.read()

        total_seconds = self._SCREENSHOTS_PER_VIDEO * self._SECONDS_PER_SCREENSHOT
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        curr_prompt = prompt.replace("{max_time}", f"{minutes:02d}:{seconds:02d}")

        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=genai_types.Content(
                parts=[
                    genai_types.Part(
                        inline_data=genai_types.Blob(
                            data=video_bytes,
                            mime_type="video/mp4"
                        )
                    ),
                    genai_types.Part(
                        text=curr_prompt
                    ),
                ]
            ),
            config=generate_content_config,
        )

        print("CAPTIONING")
        print(resp.text)

        return resp.text

    # --------------------------- processing logic --------------------------
    async def _maybe_flush_video(self) -> None:
        """If enough screenshots are queued, build video → Gemini → emit Update."""

        print(len(self._queued_paths) , self._SCREENSHOTS_PER_VIDEO)

        if len(self._queued_paths) < self._SCREENSHOTS_PER_VIDEO:
            return

        paths, self._queued_paths = (
            self._queued_paths[: self._SCREENSHOTS_PER_VIDEO],
            self._queued_paths[self._SCREENSHOTS_PER_VIDEO :],
        )

        video_path = await self._create_video_from_frames(paths)

        try:
            transcription = self._call_gemini(self.transcription_prompt, video_path)
        except Exception as exc:
            transcription = f"[Gemini call failed: {exc}]"

        await self.update_queue.put(Update(content=transcription, content_type="input_text"))

    async def _process_event(self, before_path: str, after_path: str | None) -> None:
        """Store screenshot paths and maybe trigger video flush."""
        self._history.append(before_path)
        self._queued_paths.append(before_path)
        if after_path:
            self._queued_paths.append(after_path)

        await self._maybe_flush_video()

    # ----------------------------- skip guard -----------------------------
    def _skip(self) -> bool:
        return _is_app_visible(self._guard) if self._guard else False

    # --------------------------- main async worker ------------------------
    async def _worker(self) -> None:  # overrides base class
        log = logging.getLogger("Screen")
        if self.debug:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s [Screen] %(message)s", datefmt="%H:%M:%S")
        else:
            log.addHandler(logging.NullHandler())
            log.propagate = False

        CAP_FPS  = self._CAPTURE_FPS
        DEBOUNCE = self._DEBOUNCE_SEC

        loop = asyncio.get_running_loop()

        # ------------------------------------------------------------------
        # All calls to mss are wrapped in `to_thread`
        # ------------------------------------------------------------------
        with mss.mss() as sct:
            mons = sct.monitors[self._MON_START:]

            # ---- mouse callbacks (pynput is sync → schedule into loop) ----
            def schedule_mouse_event(x: float, y: float, typ: str, **extra):
                asyncio.run_coroutine_threadsafe(mouse_event(x, y, typ, **extra), loop)

            mouse_listener = mouse.Listener(
                on_move=lambda x, y: schedule_mouse_event(x, y, "move"),
                on_click=lambda x, y, btn, prs: schedule_mouse_event(x, y, "click", button=str(btn), pressed=prs),
                on_scroll=lambda x, y, dx, dy: schedule_mouse_event(x, y, "scroll", dx=dx, dy=dy),
            )
            mouse_listener.start()

            # ---- keyboard callbacks (pynput) ----
            def schedule_key_event(k):
                asyncio.run_coroutine_threadsafe(key_event(k), loop)

            keyboard_listener = keyboard.Listener(on_press=schedule_key_event)
            keyboard_listener.start()

            # ---- nested helper inside the async context ----
            async def flush():
                if self._pending_event is None:
                    return
                if self._skip():
                    self._pending_event = None
                    return

                ev  = self._pending_event
                aft = await asyncio.to_thread(sct.grab, mons[ev["mon"] - 1])

                bef_path = await self._save_frame(ev["before"], "before")
                aft_path = await self._save_frame(aft,          "after")

                await self._process_event(bef_path, aft_path)

                log.info(f"{ev['type']} captured on monitor {ev['mon']}\n")
                self._pending_event = None

            def debounce_flush():
                asyncio.create_task(flush())

            # ---- mouse event reception ----
            async def mouse_event(x: float, y: float, typ: str, **kw):
                idx = self._mon_for(x, y, mons)
                if self._skip() or idx is None:
                    return

                # immediately log raw event -------------------------------
                await self._log_event({
                    "ts": time.time(),
                    "device": "mouse",
                    "type": typ,
                    "x": x,
                    "y": y,
                    **kw,
                })

                # lazily grab before-frame -------------------------------
                if self._pending_event is None:
                    async with self._frame_lock:
                        bf = self._frames.get(idx)
                    if bf is None:
                        return
                    self._pending_event = {"type": typ, "mon": idx, "before": bf}

                # reset debounce timer -----------------------------------
                if self._debounce_handle:
                    self._debounce_handle.cancel()
                self._debounce_handle = loop.call_later(DEBOUNCE, debounce_flush)

            # ---- keyboard event reception ----
            async def key_event(k):
                # translate key into printable / name
                try:
                    key_repr = k.char if hasattr(k, "char") and k.char else str(k)
                except AttributeError:
                    key_repr = str(k)

                await self._log_event({
                    "ts": time.time(),
                    "device": "keyboard",
                    "type": "press",
                    "key": key_repr,
                })

            # ---- main capture loop ----
            log.info(f"Screen observer started — guarding {self._guard or '∅'}")

            while self._running:  # flag inherited from Observer
                t0 = time.time()

                # refresh 'before' buffers
                for idx, m in enumerate(mons, 1):
                    frame = await asyncio.to_thread(sct.grab, m)
                    async with self._frame_lock:
                        self._frames[idx] = frame

                # fps throttle
                dt = time.time() - t0
                await asyncio.sleep(max(0, (1 / CAP_FPS) - dt))

            # shutdown ----------------------------------------------------
            mouse_listener.stop()
            keyboard_listener.stop()
            if self._debounce_handle:
                self._debounce_handle.cancel()
