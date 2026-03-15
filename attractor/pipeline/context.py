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



class ArtifactStore:
    """Named artifact store with file-backing for objects > 100KB (spec §5.5)."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._manifest: dict[str, dict[str, Any]] = {}

    def save(self, name: str, data: Any, metadata: dict[str, Any] | None = None) -> str:
        """Save data as an artifact, offloading to file if > 100KB."""
        serialized = json.dumps(data, default=str)
        size = len(serialized)
        
        artifact_info = {
            "name": name,
            "size": size,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }

        if size > 100 * 1024:
            file_path = self.base_dir / f"{name}_{int(time.time())}.json"
            file_path.write_text(serialized, encoding="utf-8")
            artifact_info["path"] = str(file_path)
            artifact_info["location"] = "file"
        else:
            artifact_info["data"] = data
            artifact_info["location"] = "memory"

        self._manifest[name] = artifact_info
        return name

    def load(self, name: str) -> Any:
        """Load artifact data by name."""
        info = self._manifest.get(name)
        if not info:
            return None
        
        if info["location"] == "file":
            return json.loads(Path(info["path"]).read_text(encoding="utf-8"))
        return info["data"]

    def list_artifacts(self) -> list[str]:
        return list(self._manifest.keys())


class Context:
    """Thread-safe key-value store shared across all pipeline stages."""

    def __init__(self, artifact_dir: str | Path | None = None) -> None:
        self._values: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._logs: list[str] = []
        self.artifacts = ArtifactStore(artifact_dir or Path(".artifacts"))

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            # If value is large, automatically store in ArtifactStore
            if isinstance(value, (str, bytes, list, dict)) and len(str(value)) > 50 * 1024:
                 self.artifacts.save(key, value)
            self._values[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            val = self._values.get(key, default)
            # If it's a large value, it might be in ArtifactStore (transparent load is tricky here without markers)
            return val

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
