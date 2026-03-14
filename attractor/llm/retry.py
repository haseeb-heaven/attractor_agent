"""Retry logic with exponential backoff for the Unified LLM Client."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from attractor.llm.errors import ProviderError, SDKError

T = TypeVar("T")


@dataclass
class BackoffConfig:
    """Configuration for retry backoff delays."""
    initial_delay_ms: int = 200
    backoff_factor: float = 2.0
    max_delay_ms: int = 60_000
    jitter: bool = True

    def delay_for_attempt(self, attempt: int) -> float:
        """Calculate delay in seconds for the given attempt (1-indexed)."""
        delay_ms = self.initial_delay_ms * (self.backoff_factor ** (attempt - 1))
        delay_ms = min(delay_ms, self.max_delay_ms)
        if self.jitter:
            delay_ms *= random.uniform(0.5, 1.5)
        return delay_ms / 1000.0


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3
    backoff: BackoffConfig | None = None

    def __post_init__(self) -> None:
        if self.backoff is None:
            self.backoff = BackoffConfig()


# Preset retry configurations
RETRY_NONE = RetryConfig(max_attempts=1)
RETRY_STANDARD = RetryConfig(
    max_attempts=5,
    backoff=BackoffConfig(initial_delay_ms=200, backoff_factor=2.0),
)
RETRY_AGGRESSIVE = RetryConfig(
    max_attempts=5,
    backoff=BackoffConfig(initial_delay_ms=500, backoff_factor=2.0),
)
RETRY_LINEAR = RetryConfig(
    max_attempts=3,
    backoff=BackoffConfig(initial_delay_ms=500, backoff_factor=1.0),
)
RETRY_PATIENT = RetryConfig(
    max_attempts=3,
    backoff=BackoffConfig(initial_delay_ms=2000, backoff_factor=3.0),
)


def is_retryable(error: Exception) -> bool:
    """Determine if an error is retryable."""
    if isinstance(error, ProviderError):
        return error.retryable
    # Network and timeout errors are retryable
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True
    return False


def retry_sync(
    fn: Callable[..., T],
    config: RetryConfig | None = None,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute a function with synchronous retry logic."""
    if config is None:
        config = RETRY_STANDARD
    assert config.backoff is not None

    last_error: Exception | None = None
    for attempt in range(1, config.max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            if not is_retryable(e) or attempt >= config.max_attempts:
                raise
            delay = config.backoff.delay_for_attempt(attempt)
            # Use retry_after from provider if available
            if isinstance(e, ProviderError) and e.retry_after is not None:
                delay = max(delay, e.retry_after)
            time.sleep(delay)

    assert last_error is not None
    raise last_error


async def retry_async(
    fn: Callable[..., Any],
    config: RetryConfig | None = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute an async function with retry logic."""
    if config is None:
        config = RETRY_STANDARD
    assert config.backoff is not None

    last_error: Exception | None = None
    for attempt in range(1, config.max_attempts + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            if not is_retryable(e) or attempt >= config.max_attempts:
                raise
            delay = config.backoff.delay_for_attempt(attempt)
            if isinstance(e, ProviderError) and e.retry_after is not None:
                delay = max(delay, e.retry_after)
            await asyncio.sleep(delay)

    assert last_error is not None
    raise last_error
