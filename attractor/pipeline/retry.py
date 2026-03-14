"""Pipeline retry policies with backoff configurations."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass


@dataclass
class BackoffConfig:
    """Configuration for retry delay calculation."""
    initial_delay_ms: int = 500
    backoff_factor: float = 2.0
    max_delay_ms: int = 60_000
    jitter: bool = True

    def delay_for_attempt(self, attempt: int) -> float:
        """Calculate delay in seconds for a given retry attempt (1-indexed)."""
        delay_ms = self.initial_delay_ms * (self.backoff_factor ** (attempt - 1))
        delay_ms = min(delay_ms, self.max_delay_ms)
        if self.jitter:
            delay_ms *= random.uniform(0.5, 1.5)
        return delay_ms / 1000.0


@dataclass
class RetryPolicy:
    """Controls how a stage retries on failure."""
    max_retries: int = 0
    backoff: BackoffConfig | None = None
    name: str = ""

    def should_retry(self, attempt: int) -> bool:
        """Whether to retry given the current attempt count."""
        return attempt < self.max_retries

    def get_delay(self, attempt: int) -> float:
        """Get delay in seconds before the next retry."""
        if self.backoff is None:
            return 0.0
        return self.backoff.delay_for_attempt(attempt)

    def wait(self, attempt: int) -> None:
        """Sleep for the appropriate delay before retrying."""
        delay = self.get_delay(attempt)
        if delay > 0:
            time.sleep(delay)


# Preset retry policies (spec §3.14)
RETRY_NONE = RetryPolicy(max_retries=0, name="none")
RETRY_STANDARD = RetryPolicy(
    max_retries=3,
    backoff=BackoffConfig(initial_delay_ms=500, backoff_factor=2.0),
    name="standard",
)
RETRY_AGGRESSIVE = RetryPolicy(
    max_retries=5,
    backoff=BackoffConfig(initial_delay_ms=500, backoff_factor=2.0),
    name="aggressive",
)
RETRY_LINEAR = RetryPolicy(
    max_retries=3,
    backoff=BackoffConfig(initial_delay_ms=500, backoff_factor=1.0, jitter=False),
    name="linear",
)
RETRY_PATIENT = RetryPolicy(
    max_retries=3,
    backoff=BackoffConfig(initial_delay_ms=2000, backoff_factor=3.0),
    name="patient",
)

# Named policy lookup
PRESET_POLICIES: dict[str, RetryPolicy] = {
    "none": RETRY_NONE,
    "standard": RETRY_STANDARD,
    "aggressive": RETRY_AGGRESSIVE,
    "linear": RETRY_LINEAR,
    "patient": RETRY_PATIENT,
}


def get_policy(name: str) -> RetryPolicy:
    """Get a preset retry policy by name, defaulting to 'standard'."""
    return PRESET_POLICIES.get(name, RETRY_STANDARD)
