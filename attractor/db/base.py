"""Database storage abstractions for Attractor Pipeline runs."""

import abc
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
    def save_question(self, run_id: str, question_id: str, node_id: str, text: str, options: list[dict[str, Any]]) -> None:
        """Store a human-in-the-loop question."""
        pass

    @abc.abstractmethod
    def get_questions(self, run_id: str) -> list[dict[str, Any]]:
        """Retrieve unanswered questions for a run."""
        pass

    @abc.abstractmethod
    def answer_question(self, run_id: str, question_id: str, answer: dict[str, Any]) -> None:
        """Submit an answer to a question."""
        pass
