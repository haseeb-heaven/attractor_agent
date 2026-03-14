"""Error hierarchy for the Unified LLM Client.

Every error inherits from SDKError. ProviderError subtypes map
to specific HTTP status codes and retryability classifications.
"""

from __future__ import annotations

from typing import Any


class SDKError(Exception):
    """Base error for all library errors."""

    def __init__(self, message: str, *, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause


# ---------------------------------------------------------------------------
# Provider errors
# ---------------------------------------------------------------------------

class ProviderError(SDKError):
    """Error returned by an LLM provider."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status_code: int | None = None,
        error_code: str | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
        raw: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message, cause=cause)
        self.provider = provider
        self.status_code = status_code
        self.error_code = error_code
        self.retryable = retryable
        self.retry_after = retry_after
        self.raw = raw


class AuthenticationError(ProviderError):
    """401: invalid API key or expired token."""

    def __init__(self, message: str = "Authentication failed", **kwargs: Any):
        super().__init__(message, retryable=False, **kwargs)


class AccessDeniedError(ProviderError):
    """403: insufficient permissions."""

    def __init__(self, message: str = "Access denied", **kwargs: Any):
        super().__init__(message, retryable=False, **kwargs)


class NotFoundError(ProviderError):
    """404: model or endpoint not found."""

    def __init__(self, message: str = "Not found", **kwargs: Any):
        super().__init__(message, retryable=False, **kwargs)


class InvalidRequestError(ProviderError):
    """400/422: malformed request or invalid parameters."""

    def __init__(self, message: str = "Invalid request", **kwargs: Any):
        super().__init__(message, retryable=False, **kwargs)


class RateLimitError(ProviderError):
    """429: rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", **kwargs: Any):
        kwargs.setdefault("retryable", True)
        super().__init__(message, **kwargs)


class ServerError(ProviderError):
    """500-599: provider internal error."""

    def __init__(self, message: str = "Server error", **kwargs: Any):
        kwargs.setdefault("retryable", True)
        super().__init__(message, **kwargs)


class ContentFilterError(ProviderError):
    """Response blocked by safety filter."""

    def __init__(self, message: str = "Content filtered", **kwargs: Any):
        super().__init__(message, retryable=False, **kwargs)


class ContextLengthError(ProviderError):
    """Input + output exceeds context window."""

    def __init__(self, message: str = "Context length exceeded", **kwargs: Any):
        super().__init__(message, retryable=False, **kwargs)


class QuotaExceededError(ProviderError):
    """Billing/usage quota exhausted."""

    def __init__(self, message: str = "Quota exceeded", **kwargs: Any):
        super().__init__(message, retryable=False, **kwargs)


# ---------------------------------------------------------------------------
# Non-provider errors
# ---------------------------------------------------------------------------

class RequestTimeoutError(SDKError):
    """Request or stream timed out."""
    pass


class AbortError(SDKError):
    """Request cancelled via abort signal."""
    pass


class NetworkError(SDKError):
    """Network-level failure."""
    pass


class StreamError(SDKError):
    """Error during stream consumption."""
    pass


class InvalidToolCallError(SDKError):
    """Tool call arguments failed validation."""
    pass


class NoObjectGeneratedError(SDKError):
    """Structured output parsing/validation failed."""
    pass


class ConfigurationError(SDKError):
    """SDK misconfiguration (missing provider, etc.)."""
    pass


# ---------------------------------------------------------------------------
# HTTP status code mapping
# ---------------------------------------------------------------------------

_STATUS_TO_ERROR: dict[int, type[ProviderError]] = {
    400: InvalidRequestError,
    401: AuthenticationError,
    403: AccessDeniedError,
    404: NotFoundError,
    413: ContextLengthError,
    422: InvalidRequestError,
    429: RateLimitError,
}


def error_from_status(
    status_code: int,
    message: str,
    *,
    provider: str = "",
    raw: dict[str, Any] | None = None,
    retry_after: float | None = None,
) -> ProviderError:
    """Create the appropriate ProviderError subclass from an HTTP status code."""
    if status_code in _STATUS_TO_ERROR:
        cls = _STATUS_TO_ERROR[status_code]
        return cls(
            message, provider=provider, status_code=status_code,
            raw=raw, retry_after=retry_after,
        )
    if 500 <= status_code < 600:
        return ServerError(
            message, provider=provider, status_code=status_code,
            raw=raw, retryable=True, retry_after=retry_after,
        )
    # Unknown status — default to retryable (conservative)
    return ProviderError(
        message, provider=provider, status_code=status_code,
        raw=raw, retryable=True, retry_after=retry_after,
    )
