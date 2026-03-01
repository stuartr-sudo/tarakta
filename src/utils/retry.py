from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any, Callable

from src.utils.logging import get_logger

logger = get_logger(__name__)


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 5.0,
    multiplier: float = 3.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """Exponential backoff retry decorator for async functions."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        delay = base_delay * (multiplier ** (attempt - 1))
                        logger.warning(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            delay=delay,
                            error=str(e),
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "retry_exhausted",
                            function=func.__name__,
                            attempts=max_attempts,
                            error=str(e),
                        )
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator
