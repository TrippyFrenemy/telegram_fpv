import asyncio, random
from src.config import settings


class RateLimiter:
    def __init__(self, rps: int, jitter_ms: int):
        self.delay = 1.0 / max(1, rps)
        self.jitter_ms = jitter_ms
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            await asyncio.sleep(self.delay + random.uniform(0, self.jitter_ms/1000))
