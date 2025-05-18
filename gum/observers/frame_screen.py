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
from typing import Any, Dict, Optional

# — Third‑party —
import mss
from PIL import Image
from pynput import mouse, keyboard

# — Local —
from .observer import Observer
from .window_geometry import is_app_visible
from ..schemas import Update

###############################################################################
# Realtime screen & keystroke observer                                        #
###############################################################################

class FrameScreen(Observer):
    """Capture screen frames and low‑latency input events, writing **JSONL** lines.

    Screenshots are captured **only** after an input gesture (click, scroll or
    debounced mouse‑move), with their own *longer* debounce so that a burst of
    related activity maps to **one** frame.
    """

    # ────────────── tunables
    _CAPTURE_FPS: int = 10            # refresh live frame buffers
    _MOVE_IDLE_SEC: float = 0.35      # idle gap ending a "move" gesture
    _SCROLL_IDLE_SEC: float = 0.35    # idle gap ending a "scroll" gesture
    _FRAME_IDLE_SEC: float = 1.00     # idle gap before we snapshot screen
    _MON_START: int = 1               # first real display in mss.monitors

    # ───────────────────────── construction
    def __init__(
        self,
        screenshots_dir: str = "~/.cache/gum/screenshots",
        keystrokes_path: str = "~/.cache/gum/keystrokes.log",  # now *JSONL*
        skip_when_visible: Optional[str | list[str]] = None,
        debug: bool = False,
    ) -> None:

        # screenshot output directory
        self.screens_dir = os.path.abspath(os.path.expanduser(screenshots_dir))
        os.makedirs(self.screens_dir, exist_ok=True)

        # low‑latency input‑event log (JSONL)
        self.keystrokes_path = os.path.abspath(os.path.expanduser(keystrokes_path))
        os.makedirs(os.path.dirname(self.keystrokes_path), exist_ok=True)
        self._keys_fh = open(self.keystrokes_path, "a", buffering=1)

        # visibility guard (pause logging if guarded window is on screen)
        self._guard = {skip_when_visible} if isinstance(skip_when_visible, str) else set(skip_when_visible or [])
        self.debug = debug

        # per‑monitor live frame buffers
        self._frames: Dict[int, Any] = {}
        self._frame_lock = asyncio.Lock()

        # mouse‑press bookkeeping (down..up duration)
        self._press_state: Optional[dict] = None
        self._button_pressed: Optional[str] = None

        # debounced gesture state/handles
        self._move_state: Optional[dict] = None
        self._move_handle: Optional[asyncio.TimerHandle] = None
        self._scroll_state: Optional[dict] = None
        self._scroll_handle: Optional[asyncio.TimerHandle] = None

        # debounced screenshot handle
        self._shot_handle: Optional[asyncio.TimerHandle] = None

        super().__init__()

    # ───────────────────────── helpers
    @staticmethod
    def _mon_for(x: float, y: float, mons: list[dict]) -> Optional[int]:
        """Return (1‑based) monitor index containing the point, or *None*."""
        for idx, m in enumerate(mons, 1):
            if m["left"] <= x < m["left"] + m["width"] and m["top"] <= y < m["top"] + m["height"]:
                return idx
        return None

    async def _save_frame(self, frame, tag: str) -> str:
        ts = f"{time.time():.5f}"
        path = os.path.join(self.screens_dir, f"{ts}_{tag}.jpg")
        await asyncio.to_thread(
            Image.frombytes("RGB", (frame.width, frame.height), frame.rgb).save,
            path,
            "JPEG",
            quality=90,
        )
        return path

    async def _log_event(self, etype: str, payload: dict) -> None:
        """Append a single **JSONL** event and propagate downstream."""
        event = {"ts": time.time(), "type": etype, **payload}
        line = json.dumps(event, separators=(",", ":")) + "\n"
        await asyncio.to_thread(self._keys_fh.write, line)
        await asyncio.to_thread(self._keys_fh.flush)
        await self.update_queue.put(Update(content=line.rstrip(), content_type="input_text"))

    # ───────────────────────── debounce helpers
    def _skip(self) -> bool:
        """Whether we should pause because a guarded window is visible."""
        return is_app_visible(self._guard) if self._guard else False

    def _reset_timer(self, attr: str, delay: float, coro):
        """Cancel any existing timer stored at *attr* and start a new one."""
        loop = asyncio.get_running_loop()
        if (h := getattr(self, attr)):
            h.cancel()
        setattr(self, attr, loop.call_later(delay, lambda: asyncio.create_task(coro())))

    def _schedule_screenshot(self, idx: int, sct: mss.mss, mons: list[dict]):
        """Queue a screenshot *after* a longer idle gap than gesture debounce."""
        async def take_shot():
            if self._skip():
                return
            frame = await asyncio.to_thread(sct.grab, mons[idx - 1])
            path = await self._save_frame(frame, "after")
            await self._log_event("frame", {"mon": idx, "path": path})
        self._reset_timer("_shot_handle", self._FRAME_IDLE_SEC, take_shot)

    # ───────────────────────── main async worker
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

            # ─── OS‑thread → asyncio trampolines ───
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
