"""Core types and enums for the coding agent loop."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from attractor.llm.types import ToolCall, ToolResult, Usage, ToolDefinition


class SessionState(enum.Enum):
    """Lifecycle states for an agent session."""
    IDLE = "idle"
    PROCESSING = "processing"
    AWAITING_INPUT = "awaiting_input"
    CLOSED = "closed"


@dataclass
class UserTurn:
    """Input from the user."""
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AssistantTurn:
    """Output from the model."""
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str | None = None
    usage: Usage | None = None
    response_id: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolResultsTurn:
    """Results from executing one or more tool calls."""
    results: list[ToolResult]
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemTurn:
    """System message (instructions or context)."""
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class SteeringTurn:
    """User-injected steering message during a tool round."""
    content: str
    timestamp: float = field(default_factory=time.time)


# Union type for history entries
Turn = UserTurn | AssistantTurn | ToolResultsTurn | SystemTurn | SteeringTurn


class EventKind(enum.Enum):
    """Event types for the coding agent loop (Section 2.9)."""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    USER_INPUT = "user_input"
    ASSISTANT_TEXT_START = "assistant_text_start"
    ASSISTANT_TEXT_DELTA = "assistant_text_delta"
    ASSISTANT_TEXT_END = "assistant_text_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_OUTPUT_DELTA = "tool_call_output_delta"
    TOOL_CALL_END = "tool_call_end"
    STEERING_INJECTED = "steering_injected"
    TURN_LIMIT = "turn_limit"
    LOOP_DETECTION = "loop_detection"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class SessionEvent:
    """A typed event emitted during agent session execution."""
    kind: EventKind
    session_id: str
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecResult:
    """Result of a command execution."""
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_ms: int


@dataclass
class DirEntry:
    """FileSystem directory entry."""
    name: str
    is_dir: bool
    size: int | None = None


@runtime_checkable
class ExecutionEnvironment(Protocol):
    """Interface for tool execution environments (Section 4.1)."""

    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str: ...
    def read_file_raw(self, path: str) -> str: ...
    def write_file(self, path: str, content: str) -> None: ...
    def file_exists(self, path: str) -> bool: ...
    def list_directory(self, path: str, depth: int = 1) -> list[DirEntry]: ...

    def exec_command(
        self,
        command: str,
        timeout_ms: int,
        working_dir: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> ExecResult: ...

    def grep(self, pattern: str, path: str, case_insensitive: bool = False, max_results: int = 100) -> str: ...
    def glob(self, pattern: str, path: str) -> list[str]: ...

    def initialize(self) -> None: ...
    def cleanup(self) -> None: ...

    def working_directory(self) -> str: ...
    def platform(self) -> str: ...
    def os_version(self) -> str: ...


@runtime_checkable
class ProviderProfile(Protocol):
    """Interface for provider-aligned toolsets and prompts (Section 3.2)."""
    id: str
    model: str

    def build_system_prompt(self, environment: ExecutionEnvironment, project_docs: str = "") -> str: ...
    def tools(self) -> list[ToolDefinition]: ...
    def provider_options(self) -> dict[str, Any] | None: ...

    @property
    def supports_reasoning(self) -> bool: ...

    @property
    def supports_streaming(self) -> bool: ...

    @property
    def supports_parallel_tool_calls(self) -> bool: ...

    @property
    def context_window_size(self) -> int: ...
