"""Context, Outcome, and Checkpoint — state management for pipeline runs."""

from __future__ import annotations

import enum
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class StageStatus(enum.Enum):
    """Outcome status of a node handler execution."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    RETRY = "retry"
    FAIL = "fail"
    SKIPPED = "skipped"


@dataclass
class Outcome:
    """Result of executing a node handler."""
    status: StageStatus = StageStatus.SUCCESS
    preferred_label: str = ""
    suggested_next_ids: list[str] = field(default_factory=list)
    context_updates: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    failure_reason: str = ""


class Context:
    """Thread-safe key-value store shared across all pipeline stages."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._logs: list[str] = []

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._values[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._values.get(key, default)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._values.keys())

    def items(self) -> list[tuple[str, Any]]:
        with self._lock:
            return list(self._values.items())

    def get_string(self, key: str, default: str = "") -> str:
        value = self.get(key)
        if value is None:
            return default
        return str(value)

    def append_log(self, entry: str) -> None:
        with self._lock:
            self._logs.append(entry)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._values)

    def clone(self) -> Context:
        with self._lock:
            ctx = Context()
            ctx._values = dict(self._values)
            ctx._logs = list(self._logs)
            return ctx

    def apply_updates(self, updates: dict[str, Any]) -> None:
        with self._lock:
            self._values.update(updates)

    @property
    def logs(self) -> list[str]:
        with self._lock:
            return list(self._logs)


@dataclass
class Checkpoint:
    """Serializable snapshot of execution state for crash recovery."""
    timestamp: float = 0.0
    current_node: str = ""
    completed_nodes: list[str] = field(default_factory=list)
    node_retries: dict[str, int] = field(default_factory=dict)
    context_values: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        """Serialize to JSON and write to filesystem."""
        data = {
            "timestamp": self.timestamp or time.time(),
            "current_node": self.current_node,
            "completed_nodes": self.completed_nodes,
            "node_retries": self.node_retries,
            "context": self.context_values,
            "logs": self.logs,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Checkpoint:
        """Deserialize from JSON file."""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            timestamp=data.get("timestamp", 0),
            current_node=data.get("current_node", ""),
            completed_nodes=data.get("completed_nodes", []),
            node_retries=data.get("node_retries", {}),
            context_values=data.get("context", {}),
            logs=data.get("logs", []),
        )

    @classmethod
    def from_context(
        cls,
        context: Context,
        current_node: str,
        completed_nodes: list[str],
        node_retries: dict[str, int] | None = None,
    ) -> Checkpoint:
        """Create a checkpoint from current execution state."""
        return cls(
            timestamp=time.time(),
            current_node=current_node,
            completed_nodes=list(completed_nodes),
            node_retries=dict(node_retries or {}),
            context_values=context.snapshot(),
            logs=context.logs,
        )
