"""Database initialization factory."""

import os
from attractor.db.base import StorageBackend

def get_db() -> StorageBackend:
    """Factory to get the configured database backend."""
    target = os.environ.get("ATTRACTOR_DB", "sqlite").lower()
    
    if target == "mongodb":
        from attractor.db.mongodb import MongoBackend
        conn_str = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
        return MongoBackend(connection_string=conn_str)
    else:
        # Default to SQLite
        from attractor.db.sqlite import SQLiteBackend
        db_path = os.environ.get("SQLITE_PATH", "attractor_runs.db")
        return SQLiteBackend(db_path=db_path)
