import asyncio
import time

class TokenBucketLimiter:
    def __init__(self, rate: float, capacity: float):
        """
        Token Bucket algorithm for rate limiting.
        :param rate: Number of tokens to add per second.
        :param capacity: Maximum number of tokens in the bucket.
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self, amount: float = 1.0):
        """
        Acquire a token from the bucket. Blocks if not enough tokens are available.
        """
        while True:
            async with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                
                # Refill tokens
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_refill = now

                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                
                # Calculate time to wait for enough tokens
                wait_time = (amount - self.tokens) / self.rate
            
            # Sleep outside the lock to allow other coroutines to run
            await asyncio.sleep(wait_time)
