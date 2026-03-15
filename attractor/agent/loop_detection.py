"""Loop detection — identifies repeating patterns in tool call history."""

from __future__ import annotations



def detect_loop(
    tool_call_signatures: list[str],
    window_size: int = 10,
) -> bool:
    """Detect if the agent is in a loop (Section 2.10).

    Checks for repeating patterns of length 1, 2, or 3 in the recent history.
    """
    recent_calls = tool_call_signatures[-window_size:]
    if len(recent_calls) < window_size:
        return False

    # Check for repeating patterns of length 1, 2, or 3
    for pattern_len in [1, 2, 3]:
        if window_size % pattern_len != 0:
            continue
        
        pattern = recent_calls[:pattern_len]
        all_match = True
        for i in range(pattern_len, window_size, pattern_len):
            if recent_calls[i : i + pattern_len] != pattern:
                all_match = False
                break
        
        if all_match:
            return True

    return False
