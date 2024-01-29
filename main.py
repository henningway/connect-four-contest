import asyncio

from fars_bot import FarsBot
from game import Simulation, MonkeyPlayer, Color

if __name__ == "__main__":
    asyncio.run(Simulation.single(MonkeyPlayer(Color.RED), FarsBot(Color.YELLOW)))
