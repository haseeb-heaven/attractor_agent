"""Session configuration for the Coding Agent Loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionConfig:
    """Configuration for an agent session."""

    # Turn limits
    max_turns: int = 0  # 0 = unlimited
    max_tool_rounds_per_input: int = 0  # 0 = unlimited

    # Timeouts
    default_command_timeout_ms: int = 10_000
    max_command_timeout_ms: int = 600_000

    # Tool output limits (characters)
    tool_output_limits: dict[str, int] = field(default_factory=lambda: {
        "read_file": 50_000,
        "shell": 30_000,
        "grep": 20_000,
        "glob": 10_000,
        "edit_file": 10_000,
        "write_file": 10_000,
    })

    # Tool line limits (applied after character truncation)
    tool_line_limits: dict[str, int | None] = field(default_factory=lambda: {
        "shell": 256,
        "grep": 200,
        "glob": 500,
        "read_file": None,
        "edit_file": None,
    })

    # Context window
    context_window_size: int = 128_000  # tokens
    context_warning_threshold: float = 0.8

    # Loop detection
    loop_detection_window: int = 10
    loop_detection_threshold: int = 3

    # Subagents
    max_subagent_depth: int = 1

    # Env var filtering patterns
    sensitive_var_patterns: list[str] = field(default_factory=lambda: [
        "*_API_KEY", "*_SECRET", "*_TOKEN", "*_PASSWORD",
    ])

    # Reasoning effort
    reasoning_effort: str | None = None  # "low", "medium", "high"

    # Additional
    working_directory: str = ""
    model: str = ""
    provider: str = ""
