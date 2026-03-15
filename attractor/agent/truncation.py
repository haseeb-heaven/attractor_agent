"""Tool output truncation — head/tail split, character-first then line-based."""

from __future__ import annotations

from attractor.agent.config import SessionConfig


def truncate_output(output: str, max_chars: int, mode: str = "head_tail") -> str:
    """Character-based truncation (Section 5.1)."""
    if len(output) <= max_chars:
        return output

    removed = len(output) - max_chars

    if mode == "head_tail":
        half = max_chars // 2
        return (
            output[:half]
            + f"\n\n[WARNING: Tool output was truncated. {removed} characters were removed from the middle. "
            "The full output is available in the event stream. "
            "If you need to see specific parts, re-run the tool with more targeted parameters.]\n\n"
            + output[-half:]
        )

    # Default to tail truncation
    return (
        f"[WARNING: Tool output was truncated. First {removed} characters were removed. "
        "The full output is available in the event stream.]\n\n"
        + output[-max_chars:]
    )


def truncate_lines(output: str, max_lines: int) -> str:
    """Line-based truncation (Section 5.3)."""
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
    config: SessionConfig,
) -> str:
    """Full truncation pipeline (Section 5.3)."""
    # Section 5.2 defaults
    default_char_limits = {
        "read_file": 50_000,
        "shell": 30_000,
        "grep": 20_000,
        "glob": 20_000,
        "edit_file": 10_000,
        "apply_patch": 10_000,
        "write_file": 1_000,
        "spawn_agent": 20_000,
    }
    
    default_modes = {
        "read_file": "head_tail",
        "shell": "head_tail",
        "grep": "tail",
        "glob": "tail",
        "edit_file": "tail",
        "apply_patch": "tail",
        "write_file": "tail",
        "spawn_agent": "head_tail",
    }

    default_line_limits = {
        "shell": 256,
        "grep": 200,
        "glob": 500,
    }

    # Step 1: Character-based truncation (always runs first)
    max_chars = config.tool_output_limits.get(
        tool_name, default_char_limits.get(tool_name, 30_000)
    )
    mode = default_modes.get(tool_name, "tail")
    result = truncate_output(output, max_chars, mode)

    # Step 2: Line-based truncation (secondary)
    max_lines = config.tool_line_limits.get(
        tool_name, default_line_limits.get(tool_name)
    )
    if max_lines is not None:
        result = truncate_lines(result, max_lines)

    return result
