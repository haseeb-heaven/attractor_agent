"""Loop detection — identifies repeating patterns in tool call history."""

from __future__ import annotations

from collections import Counter


def detect_loop(
    tool_calls: list[str],
    window: int = 10,
    threshold: int = 3,
) -> bool:
    """Detect if the agent is in a loop by examining recent tool call patterns.

    Checks if the same tool call name (or sequence) has been repeated
    `threshold` or more times in the last `window` calls.

    Args:
        tool_calls: List of recent tool call names/signatures.
        window: Number of recent calls to examine.
        threshold: Minimum repeat count to trigger detection.

    Returns:
        True if a loop is detected.
    """
    if len(tool_calls) < threshold:
        return False

    recent = tool_calls[-window:] if len(tool_calls) > window else tool_calls

    # Check for single-call repetition
    counts = Counter(recent)
    for name, count in counts.items():
        if count >= threshold:
            return True

    # Check for repeating pair patterns
    if len(recent) >= threshold * 2:
        pairs = [f"{recent[i]}->{recent[i+1]}" for i in range(len(recent) - 1)]
        pair_counts = Counter(pairs)
        for pair, count in pair_counts.items():
            if count >= threshold:
                return True

    return False
