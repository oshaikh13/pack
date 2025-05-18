#!/usr/bin/env python3
import asyncio
from gum.observers.frame_screen import FrameScreen

async def main() -> None:
    screen = FrameScreen(
        screenshots_dir="~/.cache/gum/screens",
        keystrokes_path="~/.cache/gum/keys.log",
        debug=True                     # log to console
    )

    try:
        while True:
            await asyncio.sleep(3600)  # or asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        # 3️⃣  graceful shutdown
        await screen.stop()
        print("RealtimeScreen stopped — bye!")

if __name__ == "__main__":
    #
    # `asyncio.run()` creates the loop, runs `main()`,
    # and closes the loop when `main()` finishes.
    #
    asyncio.run(main())
