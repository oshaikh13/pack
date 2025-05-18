from __future__ import annotations
###############################################################################
# Imports                                                                     #
###############################################################################

# — Standard library —
import asyncio
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

    _CAPTURE_FPS: int = 10        # screen‑grab rate while idle
    _DEBOUNCE_SEC: int = 2        # wait after last event before "after" shot
    _MON_START: int = 1           # first real display in mss

    # ─────────────────────────────── construction
    def __init__(
        self,
        screenshots_dir: str = "~/.cache/gum/screenshots",
        keystrokes_path: str = "~/.cache/gum/keystrokes.log",
        skip_when_visible: Optional[str | list[str]] = None,
        debug: bool = False,
    ) -> None:

        # screenshot output
        self.screens_dir = os.path.abspath(os.path.expanduser(screenshots_dir))
        os.makedirs(self.screens_dir, exist_ok=True)

        # keystroke log output (line‑buffered for low latency)
        self.keystrokes_path = os.path.abspath(os.path.expanduser(keystrokes_path))
        os.makedirs(os.path.dirname(self.keystrokes_path), exist_ok=True)
        self._keys_fh = open(self.keystrokes_path, "a", buffering=1)

        # app‑visibility guard (e.g. do nothing if ChatGPT window is open)
        self._guard = {skip_when_visible} if isinstance(skip_when_visible, str) else set(skip_when_visible or [])
        self.debug = debug

        # state shared with worker
        self._frames: Dict[int, Any] = {}
        self._frame_lock = asyncio.Lock()

        self._pending_event: Optional[dict] = None
        self._debounce_handle: Optional[asyncio.TimerHandle] = None

        super().__init__()

    # ─────────────────────────────── tiny sync helpers
    @staticmethod
    def _mon_for(x: float, y: float, mons: list[dict]) -> Optional[int]:
        for idx, m in enumerate(mons, 1):
            if m["left"] <= x < m["left"] + m["width"] and m["top"] <= y < m["top"] + m["height"]:
                return idx
        return None

    # ─────────────────────────────── I/O helpers
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

    async def _log_keystroke(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        """Write a single key press to the keystroke log (tab‑separated
        ``<epoch>\t<key>``) and forward it downstream via :pyattr:`update_queue`.
        """
        try:
            k = key.char if hasattr(key, "char") else str(key)
        except Exception:
            k = str(key)
        ts = f"{time.time():.5f}"
        line = f"{ts}\t{k}\n"
        await asyncio.to_thread(self._keys_fh.write, line)
        await asyncio.to_thread(self._keys_fh.flush)
        # propagate to observers (optional)
        await self.update_queue.put(Update(content=line.strip(), content_type="input_text"))

    # ─────────────────────────────── skip guard
    def _skip(self) -> bool:
        return is_app_visible(self._guard) if self._guard else False

    # ─────────────────────────────── main async worker
    async def _worker(self) -> None:          # overrides base class
        log = logging.getLogger("Screen")
        if self.debug:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s [Screen] %(message)s", datefmt="%H:%M:%S")
        else:
            log.addHandler(logging.NullHandler())
            log.propagate = False

        CAP_FPS = self._CAPTURE_FPS
        loop = asyncio.get_running_loop()

        # ------------------------------------------------------------------
        # All calls to mss / Quartz are wrapped in `to_thread`
        # ------------------------------------------------------------------
        with mss.mss() as sct:
            mons = sct.monitors[self._MON_START:]

            # ---- mouse callbacks ----
            def schedule_mouse_event(x: float, y: float, typ: str):
                asyncio.run_coroutine_threadsafe(mouse_event(x, y, typ), loop)

            mouse_listener = mouse.Listener(
                on_move=lambda x, y: schedule_mouse_event(x, y, "move"),
                on_click=lambda x, y, btn, prs: schedule_mouse_event(x, y, "click") if prs else None,
                on_scroll=lambda x, y, dx, dy: schedule_mouse_event(x, y, "scroll"),
            )
            mouse_listener.start()

            # ---- keyboard callbacks ----
            def schedule_key_event(k):
                asyncio.run_coroutine_threadsafe(key_event(k), loop)

            keyboard_listener = keyboard.Listener(on_press=schedule_key_event)
            keyboard_listener.start()

            # ---- nested helpers ----
            async def flush():
                if self._pending_event is None:
                    return
                if self._skip():
                    self._pending_event = None
                    return

                ev = self._pending_event
                aft = await asyncio.to_thread(sct.grab, mons[ev["mon"] - 1])

                bef_path = await self._save_frame(ev["before"], "before")
                aft_path = await self._save_frame(aft, "after")
                log.info(f"Saved frames: {bef_path}, {aft_path}")

                self._pending_event = None

            def debounce_flush():
                # callback from loop.call_later → must create task
                asyncio.create_task(flush())

            # ── event handlers ──
            async def mouse_event(x: float, y: float, typ: str):
                idx = self._mon_for(x, y, mons)
                log.info(
                    f"{typ:<6} @({x:7.1f},{y:7.1f}) → mon={idx}   {'(guarded)' if self._skip() else ''}"
                )
                if self._skip() or idx is None:
                    return

                # lazily grab before‑frame
                if self._pending_event is None:
                    async with self._frame_lock:
                        bf = self._frames.get(idx)
                    if bf is None:
                        return
                    self._pending_event = {"type": typ, "mon": idx, "before": bf}

                # reset debounce timer
                if self._debounce_handle:
                    self._debounce_handle.cancel()
                self._debounce_handle = loop.call_later(self._DEBOUNCE_SEC, debounce_flush)

            async def key_event(k):
                if self._skip():
                    return
                await self._log_keystroke(k)

            # ---- main capture loop ----
            log.info(f"Screen observer started — guarding {self._guard or '∅'}")

            while self._running:                         # flag from base class
                t0 = time.time()

                # refresh 'before' buffers
                for idx, m in enumerate(mons, 1):
                    frame = await asyncio.to_thread(sct.grab, m)
                    async with self._frame_lock:
                        self._frames[idx] = frame

                # fps throttle
                dt = time.time() - t0
                await asyncio.sleep(max(0, (1 / CAP_FPS) - dt))

            # ---- shutdown ----
            mouse_listener.stop()
            keyboard_listener.stop()
            if self._debounce_handle:
                self._debounce_handle.cancel()
            self._keys_fh.close()
