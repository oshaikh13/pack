from dotenv import load_dotenv
load_dotenv()

import asyncio
from gum import gum
from gum.observers import Screen

async def main():
    async with gum("Omar Shaikh", Screen()):
        await asyncio.Future() # run forever (Ctrl-C to stop)

if __name__ == "__main__":
    asyncio.run(main())

