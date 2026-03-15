from dataclasses import dataclass, field


@dataclass
class SessionConfig:
    """Configuration for an agent session (Section 2.2)."""

    # Turn limits
    max_turns: int = 0  # 0 = unlimited
    max_tool_rounds_per_input: int = 0  # 0 = unlimited, per user input

    # Timeouts
    default_command_timeout_ms: int = 10_000   # 10 seconds
    max_command_timeout_ms: int = 600_000      # 10 minutes

    # Reasoning effort
    reasoning_effort: str | None = None  # "low", "medium", "high", or None

    # Tool output limits (characters)
    tool_output_limits: dict[str, int] = field(default_factory=lambda: {
        "read_file": 50_000,
        "shell": 30_000,
        "grep": 20_000,
        "glob": 20_000,
        "edit_file": 10_000,
        "apply_patch": 10_000,
        "write_file": 1_000,
        "spawn_agent": 20_000,
    })

    # Tool line limits (applied after character truncation)
    tool_line_limits: dict[str, int | None] = field(default_factory=lambda: {
        "shell": 256,
        "grep": 200,
        "glob": 500,
        "read_file": None,
        "edit_file": None,
        "write_file": None,
        "apply_patch": None,
    })

    # Loop detection
    enable_loop_detection: bool = True
    loop_detection_window: int = 10      # consecutive identical calls before warning

    # Subagents
    max_subagent_depth: int = 1       # max nesting level for subagents

    # Env context threshold (Section 5.5)
    context_warning_threshold: float = 0.8

    # Internal / Optional
    working_directory: str = ""
    model: str = ""
    provider: str = ""
