"""Database storage abstractions for Attractor Pipeline runs."""

import abc
from datetime import datetime
from typing import Any


class StorageBackend(abc.ABC):
    """Abstract base class for persistent storage of pipeline runs."""

    @abc.abstractmethod
    def save_run(self, run_id: str, goal: str, config: dict[str, Any]) -> None:
        """Create a new run record."""
        pass

    @abc.abstractmethod
    def update_run(self, run_id: str, status: str, final_node: str, result: dict[str, Any] | None = None) -> None:
        """Update the status and result of a run."""
        pass

    @abc.abstractmethod
    def save_event(self, run_id: str, event_kind: str, node_id: str, payload: dict[str, Any]) -> None:
        """Store a pipeline event (log, stage start/end, etc)."""
        pass

    @abc.abstractmethod
    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Retrieve full details of a run."""
        pass

    @abc.abstractmethod
    def list_runs(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """List recent runs."""
        pass
