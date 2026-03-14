"""Agent event system."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class EventKind(enum.Enum):
    """Event types for the coding agent loop."""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    STEERING = "steering"
    WARNING = "warning"
    ERROR = "error"
    CONTEXT_USAGE = "context_usage"
    TURN_LIMIT = "turn_limit"
    LOOP_DETECTED = "loop_detected"
    SUBAGENT_SPAWNED = "subagent_spawned"
    SUBAGENT_COMPLETED = "subagent_completed"


@dataclass
class SessionEvent:
    """A typed event emitted during agent session execution."""
    kind: EventKind = EventKind.SESSION_START
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)
    message: str = ""


SessionEventListener = Callable[[SessionEvent], None]


class AgentEventEmitter:
    """Event emitter for agent session events."""

    def __init__(self) -> None:
        self._listeners: list[SessionEventListener] = []

    def on(self, listener: SessionEventListener) -> None:
        self._listeners.append(listener)

    def emit(self, event: SessionEvent) -> None:
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass

    def emit_simple(self, kind: EventKind, *, message: str = "", **data: Any) -> None:
        self.emit(SessionEvent(kind=kind, message=message, data=data))
