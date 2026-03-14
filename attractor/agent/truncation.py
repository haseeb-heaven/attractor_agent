"""Tool output truncation — head/tail split, character-first then line-based."""

from __future__ import annotations

from attractor.agent.config import SessionConfig


def truncate_output(output: str, max_chars: int, mode: str = "head_tail") -> str:
    """Character-based truncation using head/tail split.

    Always runs first as the primary safeguard (handles pathological
    cases like a 2-line CSV where each line is 10MB).
    """
    if len(output) <= max_chars:
        return output

    head_size = max_chars // 2
    tail_size = max_chars - head_size
    omitted = len(output) - head_size - tail_size

    return (
        output[:head_size]
        + f"\n\n[WARNING: Tool output was truncated. {omitted} characters removed "
        f"from the middle of the output.]\n\n"
        + output[-tail_size:]
    )


def truncate_lines(output: str, max_lines: int) -> str:
    """Line-based truncation using head/tail split.

    Secondary pass that runs after character truncation for readability.
    """
    lines = output.split("\n")
    if len(lines) <= max_lines:
        return output

    head_count = max_lines // 2
    tail_count = max_lines - head_count
    omitted = len(lines) - head_count - tail_count

    return (
        "\n".join(lines[:head_count])
        + f"\n[... {omitted} lines omitted ...]\n"
        + "\n".join(lines[-tail_count:])
    )


def truncate_tool_output(
    output: str,
    tool_name: str,
    config: SessionConfig | None = None,
) -> str:
    """Full truncation pipeline for tool output.

    Step 1: Character-based truncation (always runs first)
    Step 2: Line-based truncation (for readability)
    """
    if config is None:
        config = SessionConfig()

    # Default character limits
    default_char_limits = {
        "read_file": 50_000,
        "shell": 30_000,
        "grep": 20_000,
        "glob": 10_000,
        "edit_file": 10_000,
        "write_file": 10_000,
    }

    # Default line limits
    default_line_limits: dict[str, int | None] = {
        "shell": 256,
        "grep": 200,
        "glob": 500,
        "read_file": None,
        "edit_file": None,
    }

    # Step 1: Character-based truncation
    max_chars = config.tool_output_limits.get(
        tool_name, default_char_limits.get(tool_name, 30_000)
    )
    result = truncate_output(output, max_chars)

    # Step 2: Line-based truncation
    line_limit = config.tool_line_limits.get(
        tool_name, default_line_limits.get(tool_name)
    )
    if line_limit is not None:
        result = truncate_lines(result, line_limit)

    return result
