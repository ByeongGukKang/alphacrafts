import asyncio
import datetime
import time
from types import coroutine

__all__ = [
    'MACRO_AS_ASYNC',
    'MACRO_ASYNC_SLEEP',
    'MACRO_ASYNC_YIELD',
    'MACRO_TIME_NS',
    'MACRO_DATETIME_NOW',
]

### Macros
def _MACRO_AS_ASYNC(func):
    async def _coro(*args, **kwargs):
        return func(*args, **kwargs)
    return _coro
MACRO_AS_ASYNC = _MACRO_AS_ASYNC

MACRO_ASYNC_SLEEP = asyncio.sleep
MACRO_ASYNC_YIELD = coroutine(lambda: (yield))
MACRO_TIME_NS = time.time_ns
MACRO_DATETIME_NOW = datetime.datetime.now
