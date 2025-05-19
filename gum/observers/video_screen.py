from __future__ import annotations
###############################################################################
# Imports                                                                     #
###############################################################################

# — Standard library —
import asyncio
import logging
import os
import time
from collections import deque
from importlib.resources import files as get_package_file
from typing import Any, Dict, Optional

# — Third‑party —
import mss
from PIL import Image
from pynput import mouse
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
    Heavy CPU / blocking I/O work runs in background threads via
    `asyncio.to_thread`.
    """

    # ----------------------------- tuning knobs -----------------------------
    _CAPTURE_FPS: int = 10              # live “before‑frame” refresh rate
    _DEBOUNCE_SEC: int = 1              # wait this long after an event
    _MON_START: int = 1                 # first real display in mss’ list

    _SCREENSHOTS_PER_VIDEO: int = 30    # stitch when this many frames queued
    _SECONDS_PER_SCREENSHOT: int = 1    # still frame duration inside clip

    # ----------------------------- construction -----------------------------
    def __init__(
        self,
        screenshots_dir: str = "~/.cache/gum/screenshots",
        skip_when_visible: Optional[str | list[str]] = None,
        transcription_prompt: str = "Describe what is happening on‑screen.",
        history_k: int = 10,
        debug: bool = False,
    ) -> None:

        # output dir --------------------------------------------------------
        self.screens_dir = os.path.abspath(os.path.expanduser(screenshots_dir))
        os.makedirs(self.screens_dir, exist_ok=True)

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

    async def _create_video_from_frames(self, paths: list[str]) -> str:
        """Build a silent MP4 using moviepy.  Runs in a worker thread."""
        async def _build() -> str:
            clip = ImageSequenceClip(paths, fps=1 / self._SECONDS_PER_SCREENSHOT)
            out  = os.path.join(self.screens_dir, f"{time.time():.5f}.mp4")
            clip.write_videofile(out, codec="libx264", audio=False, logger=None)
            clip.close()
            return out

        return await asyncio.to_thread(_build)

    async def _call_gemini(self, prompt: str, video_path: str) -> str:
        """Synchronous Google GenAI call wrapped in `to_thread`."""
        def _sync_call() -> str:
            generate_content_config = genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        "timestamp": genai_types.Schema(type=genai_types.Type.STRING),
                        "caption":   genai_types.Schema(type=genai_types.Type.STRING),
                    },
                ),
            )
            client   = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
            uploaded = client.files.upload(file=video_path)

            resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[uploaded, prompt],
                config=generate_content_config,
            )
            return resp.text

        return await asyncio.to_thread(_sync_call)

    # --------------------------- processing logic --------------------------
    async def _maybe_flush_video(self) -> None:
        """If enough screenshots are queued, build video → Gemini → emit Update."""
        if len(self._queued_paths) < self._SCREENSHOTS_PER_VIDEO:
            return

        paths, self._queued_paths = (
            self._queued_paths[: self._SCREENSHOTS_PER_VIDEO],
            self._queued_paths[self._SCREENSHOTS_PER_VIDEO :],
        )

        video_path = await self._create_video_from_frames(paths)
        try:
            transcription = await self._call_gemini(self.transcription_prompt, video_path)
            print("GEMINI CALLED")
            print(transcription)
        except Exception as exc:  # pragma: no cover
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
            def schedule_event(x: float, y: float, typ: str):
                asyncio.run_coroutine_threadsafe(mouse_event(x, y, typ), loop)

            listener = mouse.Listener(
                on_move=lambda x, y: schedule_event(x, y, "move"),
                on_click=lambda x, y, btn, prs: schedule_event(x, y, "click") if prs else None,
                on_scroll=lambda x, y, dx, dy: schedule_event(x, y, "scroll"),
            )
            listener.start()

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

                log.info(f"{ev['type']} captured on monitor {ev['mon']}\\n")
                self._pending_event = None

            def debounce_flush():
                asyncio.create_task(flush())

            # ---- mouse event reception ----
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
                self._debounce_handle = loop.call_later(DEBOUNCE, debounce_flush)

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

            # shutdown
            listener.stop()
            if self._debounce_handle:
                self._debounce_handle.cancel()
