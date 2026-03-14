"""Pipeline event system — typed events for UI, logging, and metrics."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class PipelineEventKind(enum.Enum):
    """All event types emitted during pipeline execution."""
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_FAILED = "pipeline_failed"
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    STAGE_RETRYING = "stage_retrying"
    PARALLEL_STARTED = "parallel_started"
    PARALLEL_BRANCH_STARTED = "parallel_branch_started"
    PARALLEL_BRANCH_COMPLETED = "parallel_branch_completed"
    PARALLEL_COMPLETED = "parallel_completed"
    INTERVIEW_STARTED = "interview_started"
    INTERVIEW_COMPLETED = "interview_completed"
    INTERVIEW_TIMEOUT = "interview_timeout"
    CHECKPOINT_SAVED = "checkpoint_saved"
    EDGE_SELECTED = "edge_selected"
    CONTEXT_UPDATED = "context_updated"
    LOG = "log"


@dataclass
class PipelineEvent:
    """A typed pipeline execution event."""
    kind: PipelineEventKind = PipelineEventKind.LOG
    timestamp: float = field(default_factory=time.time)
    node_id: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    message: str = ""


# Event listener type
EventListener = Callable[[PipelineEvent], None]


class EventEmitter:
    """Simple event emitter for pipeline execution events."""

    def __init__(self) -> None:
        self._listeners: list[EventListener] = []
        self._history: list[PipelineEvent] = []

    def on(self, listener: EventListener) -> None:
        """Register an event listener."""
        self._listeners.append(listener)

    def emit(self, event: PipelineEvent) -> None:
        """Emit an event to all listeners."""
        self._history.append(event)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass  # Listeners must not crash the engine

    def emit_simple(
        self,
        kind: PipelineEventKind,
        *,
        node_id: str = "",
        message: str = "",
        **data: Any,
    ) -> None:
        """Convenience method to emit events."""
        self.emit(PipelineEvent(
            kind=kind, node_id=node_id, message=message, data=data,
        ))

    @property
    def history(self) -> list[PipelineEvent]:
        return list(self._history)
