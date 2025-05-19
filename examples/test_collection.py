#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()
import asyncio
from gum.observers.video_screen import VideoScreen

async def main() -> None:
    screen = VideoScreen(
        # screenshots_dir="~/.cache/gum/screens",
        # keystrokes_path="~/.cache/gum/keys.log",
        debug=False                     # log to console
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
