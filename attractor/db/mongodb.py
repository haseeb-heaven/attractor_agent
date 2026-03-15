"""MongoDB implementation of StorageBackend."""

from datetime import datetime
from typing import Any

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from attractor.db.base import StorageBackend


class MongoBackend(StorageBackend):
    """Stores pipeline runs in a MongoDB database."""

    def __init__(self, connection_string: str = "mongodb://localhost:27017/", db_name: str = "attractor"):
        self.client: MongoClient[Any] = MongoClient(connection_string, serverSelectionTimeoutMS=2000)
        
        # Test connection gracefully
        try:
            self.client.admin.command('ping')
        except ConnectionFailure as e:
            # We don't crash here so the engine can fail gracefully or fallback if needed
            print(f"Warning: Could not connect to MongoDB at {connection_string}")
            
        self.db = self.client[db_name]
        self.runs = self.db.runs
        self.events = self.db.events
        
        # Simple indexes
        self.runs.create_index("run_id", unique=True)
        self.runs.create_index("created_at")
        self.events.create_index("run_id")
        self.events.create_index("timestamp")

    def save_run(self, run_id: str, goal: str, config: dict[str, Any]) -> None:
        now = datetime.now()
        self.runs.insert_one({
            "run_id": run_id,
            "goal": goal,
            "config": config,
            "status": "RUNNING",
            "final_node": None,
            "result": None,
            "created_at": now,
            "updated_at": now
        })

    def update_run(self, run_id: str, status: str, final_node: str, result: dict[str, Any] | None = None) -> None:
        now = datetime.now()
        self.runs.update_one(
            {"run_id": run_id},
            {"$set": {
                "status": status,
                "final_node": final_node,
                "result": result,
                "updated_at": now
            }}
        )

    def save_event(self, run_id: str, event_kind: str, node_id: str, payload: dict[str, Any]) -> None:
        now = datetime.now()
        self.events.insert_one({
            "run_id": run_id,
            "event_kind": event_kind,
            "node_id": node_id,
            "payload": payload,
            "timestamp": now
        })

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run = self.runs.find_one({"run_id": run_id}, {"_id": 0})
        if not run:
            return None
            
        events_cursor = self.events.find({"run_id": run_id}, {"_id": 0}).sort("timestamp", 1)
        run["events"] = list(events_cursor)
        
        return run

    def list_runs(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        cursor = self.runs.find({}, {"_id": 0}).sort("created_at", -1).skip(offset).limit(limit)
        return list(cursor)
