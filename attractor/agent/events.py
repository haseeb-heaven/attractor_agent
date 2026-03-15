from __future__ import annotations

from typing import Any, Callable


from attractor.agent.types import EventKind, SessionEvent

SessionEventListener = Callable[[SessionEvent], None]


class AgentEventEmitter:
    """Event emitter for agent session events (Section 4.3)."""

    def __init__(self) -> None:
        self._listeners: list[SessionEventListener] = []

    def on(self, listener: SessionEventListener) -> None:
        """Register a listener for events."""
        self._listeners.append(listener)

    def emit(self, event: SessionEvent) -> None:
        """Deliver an event to all listeners."""
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                # Listeners should handle their own errors
                pass

    def emit_simple(self, kind: EventKind, session_id: str, **data: Any) -> None:
        """Convenience method to emit an event."""
        self.emit(SessionEvent(kind=kind, session_id=session_id, data=data))
