from __future__ import annotations
###############################################################################
# Imports                                                                     #
###############################################################################

# — Standard library —
import base64
import logging
import os
import time
from collections import deque
from importlib.resources import files as get_package_file
from typing import Any, Dict, Iterable, List, Optional

import asyncio
from functools import partial

# — Third-party —
import mss
import Quartz
from PIL import Image
from pynput import mouse           # still synchronous
from shapely.geometry import box
from shapely.ops import unary_union

# — Local —
from .observer import Observer
from ..schemas import Update

# — OpenAI async client —
from openai import AsyncOpenAI
client = AsyncOpenAI()

###############################################################################
# Window‑geometry helpers                                                     #
###############################################################################


def _get_global_bounds() -> tuple[float, float, float, float]:
    """Return a bounding box enclosing **all** physical displays.

    Returns
    -------
    (min_x, min_y, max_x, max_y) tuple in Quartz global coordinates.
    """
    err, ids, cnt = Quartz.CGGetActiveDisplayList(16, None, None)
    if err != Quartz.kCGErrorSuccess:  # pragma: no cover (defensive)
        raise OSError(f"CGGetActiveDisplayList failed: {err}")

    min_x = min_y = float("inf")
    max_x = max_y = -float("inf")
    for did in ids[:cnt]:
        r = Quartz.CGDisplayBounds(did)
        x0, y0 = r.origin.x, r.origin.y
        x1, y1 = x0 + r.size.width, y0 + r.size.height
        min_x, min_y = min(min_x, x0), min(min_y, y0)
        max_x, max_y = max(max_x, x1), max(max_y, y1)
    return min_x, min_y, max_x, max_y


def _get_visible_windows() -> List[tuple[dict, float]]:
    """List *onscreen* windows with their visible‑area ratio.

    Each tuple is ``(window_info_dict, visible_ratio)`` where *visible_ratio*
    is in ``[0.0, 1.0]``.  Internal system windows (Dock, WindowServer, …) are
    ignored.
    """
    _, _, _, gmax_y = _get_global_bounds()

    opts = (
        Quartz.kCGWindowListOptionOnScreenOnly
        | Quartz.kCGWindowListOptionIncludingWindow
    )
    wins = Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID)

    occupied = None  # running union of opaque regions above the current window
    result: list[tuple[dict, float]] = []

    for info in wins:
        owner = info.get("kCGWindowOwnerName", "")
        if owner in ("Dock", "WindowServer", "Window Server"):
            continue

        bounds = info.get("kCGWindowBounds", {})
        x, y, w, h = (
            bounds.get("X", 0),
            bounds.get("Y", 0),
            bounds.get("Width", 0),
            bounds.get("Height", 0),
        )
        if w <= 0 or h <= 0:
            continue  # hidden or minimised

        inv_y = gmax_y - y - h  # Quartz→Shapely Y‑flip
        poly = box(x, inv_y, x + w, inv_y + h)
        if poly.is_empty:
            continue

        visible = poly if occupied is None else poly.difference(occupied)
        if not visible.is_empty:
            ratio = visible.area / poly.area
            result.append((info, ratio))
            occupied = poly if occupied is None else unary_union([occupied, poly])

    return result


def _is_app_visible(names: Iterable[str]) -> bool:
    """Return *True* if **any** window from *names* is at least partially visible."""
    targets = set(names)
    return any(
        info.get("kCGWindowOwnerName", "") in targets and ratio > 0
        for info, ratio in _get_visible_windows()
    )

###############################################################################
# Screen observer                                                             #
###############################################################################

class Screen(Observer):
    """
    Capture before/after screenshots around user interactions.
    Blocking work (Quartz, mss, Pillow, OpenAI Vision) is executed in
    background threads via `asyncio.to_thread`.
    """

    _CAPTURE_FPS: int = 10
    _PERIODIC_SEC: int = 30
    _DEBOUNCE_SEC: int = 3
    _MON_START: int = 1     # first real display in mss

    # ─────────────────────────────── construction
    def __init__(
        self,
        screenshots_dir: str = "~/.cache/gum/screenshots",
        skip_when_visible: Optional[str | list[str]] = None,
        transcription_prompt: Optional[str] = None,
        summary_prompt: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        history_k: int = 10,
        debug: bool = False,
    ) -> None:

        self.screens_dir = os.path.abspath(os.path.expanduser(screenshots_dir))
        os.makedirs(self.screens_dir, exist_ok=True)

        self._guard = {skip_when_visible} if isinstance(skip_when_visible, str) else set(skip_when_visible or [])

        self.transcription_prompt = transcription_prompt or self._load_prompt("transcribe.txt")
        self.summary_prompt       = summary_prompt       or self._load_prompt("summarize.txt")
        self.model_name = model_name

        self.debug = debug

        # state shared with worker
        self._frames: Dict[int, Any] = {}
        self._frame_lock = asyncio.Lock()

        self._history: deque[str] = deque(maxlen=max(0, history_k))
        self._pending_event: Optional[dict] = None
        self._debounce_handle: Optional[asyncio.TimerHandle] = None

        # call parent
        super().__init__()

    # ─────────────────────────────── tiny sync helpers
    @staticmethod
    def _load_prompt(fname: str) -> str:
        return get_package_file("gum.prompts.screen").joinpath(fname).read_text()

    @staticmethod
    def _mon_for(x: float, y: float, mons: list[dict]) -> Optional[int]:
        for idx, m in enumerate(mons, 1):
            if m["left"] <= x < m["left"] + m["width"] and m["top"] <= y < m["top"] + m["height"]:
                return idx
        return None

    @staticmethod
    def _encode_image(img_path: str) -> str:
        with open(img_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode()

    # ─────────────────────────────── OpenAI Vision (async)
    async def _call_gpt_vision(self, prompt: str, img_paths: list[str]) -> str:
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
            for encoded in (await asyncio.gather(
                *[asyncio.to_thread(self._encode_image, p) for p in img_paths]
            ))
        ]
        content.append({"type": "text", "text": prompt})

        rsp = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "text"},
        )
        return rsp.choices[0].message.content

    # ─────────────────────────────── I/O helpers
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

    async def _process_and_emit(self, before_path: str, after_path: str | None) -> None:
        # chronology: append 'before' first (history order == real order)
        self._history.append(before_path)
        prev_paths = list(self._history)

        # async OpenAI calls
        try:
            transcription = await self._call_gpt_vision(self.transcription_prompt, [before_path])
        except Exception as exc:                                        # pragma: no cover
            transcription = f"[transcription failed: {exc}]"

        summary = ""
        if after_path:
            prev_paths.append(after_path)
            try:
                summary = await self._call_gpt_vision(self.summary_prompt, prev_paths)
            except Exception as exc:                                    # pragma: no cover
                summary = f"[summary failed: {exc}]"

        txt = (transcription + summary).strip()
        await self.update_queue.put(Update(content=txt, content_type="input_text"))

    # ─────────────────────────────── skip guard
    def _skip(self) -> bool:
        return _is_app_visible(self._guard) if self._guard else False

    # ─────────────────────────────── main async worker
    async def _worker(self) -> None:          # overrides base class
        log = logging.getLogger("Screen")
        if self.debug:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s [Screen] %(message)s", datefmt="%H:%M:%S")
        else:
            log.addHandler(logging.NullHandler())
            log.propagate = False

        CAP_FPS  = self._CAPTURE_FPS
        PERIOD   = self._PERIODIC_SEC
        DEBOUNCE = self._DEBOUNCE_SEC

        loop = asyncio.get_running_loop()

        # ------------------------------------------------------------------
        # All calls to mss / Quartz are wrapped in `to_thread`
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

                ev = self._pending_event
                aft = await asyncio.to_thread(sct.grab, mons[ev["mon"] - 1])

                bef_path = await self._save_frame(ev["before"], "before")
                aft_path = await self._save_frame(aft, "after")
                await self._process_and_emit(bef_path, aft_path)

                log.info(f"{ev['type']} captured on monitor {ev['mon']}")
                self._pending_event = None

            def debounce_flush():
                # callback from loop.call_later → must create task
                asyncio.create_task(flush())

            # ---- mouse event reception ----
            async def mouse_event(x: float, y: float, typ: str):
                idx = self._mon_for(x, y, mons)
                log.info(
                    f"{typ:<6} @({x:7.1f},{y:7.1f}) → mon={idx}   {'(guarded)' if self._skip() else ''}"
                )
                if self._skip() or idx is None:
                    return

                # lazily grab before-frame
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
            last_periodic = time.time()

            while self._running:                         # flag from base class
                t0 = time.time()

                # refresh 'before' buffers
                for idx, m in enumerate(mons, 1):
                    frame = await asyncio.to_thread(sct.grab, m)
                    async with self._frame_lock:
                        self._frames[idx] = frame

                # periodic snapshots
                if not self._skip() and t0 - last_periodic >= PERIOD:
                    last_periodic = t0
                    for idx, m in enumerate(mons, 1):
                        frame = await asyncio.to_thread(sct.grab, m)
                        path  = await self._save_frame(frame, f"periodic_m{idx}")
                        # only transcribe single periodic frame (no summary)
                        try:
                            txt = await self._call_gpt_vision(self.transcription_prompt, [path])
                        except Exception as exc:      # pragma: no cover
                            txt = f"[periodic transcription failed: {exc}]"
                        await self.update_queue.put(Update(content=txt, content_type="input_text"))

                # fps throttle
                dt = time.time() - t0
                await asyncio.sleep(max(0, (1 / CAP_FPS) - dt))

            # shutdown
            listener.stop()
            if self._debounce_handle:
                self._debounce_handle.cancel()
