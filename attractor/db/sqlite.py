"""SQLite implementation of StorageBackend."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from attractor.db.base import StorageBackend


class SQLiteBackend(StorageBackend):
    """Stores pipeline runs in a local SQLite database file."""

    def __init__(self, db_path: str = "attractor_runs.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    goal TEXT,
                    config TEXT,
                    status TEXT,
                    final_node TEXT,
                    result TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    event_kind TEXT,
                    node_id TEXT,
                    payload TEXT,
                    timestamp TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
            ''')

    def save_run(self, run_id: str, goal: str, config: dict[str, Any]) -> None:
        now = datetime.now()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO runs (run_id, goal, config, status, created_at, updated_at)
                VALUES (?, ?, ?, 'RUNNING', ?, ?)
            ''', (run_id, goal, json.dumps(config), now, now))

    def update_run(self, run_id: str, status: str, final_node: str, result: dict[str, Any] | None = None) -> None:
        now = datetime.now()
        res_str = json.dumps(result) if result else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE runs
                SET status = ?, final_node = ?, result = ?, updated_at = ?
                WHERE run_id = ?
            ''', (status, final_node, res_str, now, run_id))

    def save_event(self, run_id: str, event_kind: str, node_id: str, payload: dict[str, Any]) -> None:
        now = datetime.now()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO events (run_id, event_kind, node_id, payload, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (run_id, event_kind, node_id, json.dumps(payload), now))

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
            if not row:
                return None
            
            run = dict(row)
            run['config'] = json.loads(run['config']) if run['config'] else {}
            run['result'] = json.loads(run['result']) if run['result'] else {}
            
            # Fetch events
            events = conn.execute('SELECT * FROM events WHERE run_id = ? ORDER BY timestamp ASC', (run_id,)).fetchall()
            run['events'] = []
            for e in events:
                event = dict(e)
                event['payload'] = json.loads(event['payload']) if event['payload'] else {}
                run['events'].append(event)
                
            return run

    def list_runs(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                'SELECT * FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?',
                (limit, offset)
            ).fetchall()
            
            runs = []
            for row in rows:
                run = dict(row)
                run['config'] = json.loads(run['config']) if run['config'] else {}
                run['result'] = json.loads(run['result']) if run['result'] else {}
                runs.append(run)
            return runs
