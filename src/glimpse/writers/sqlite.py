# src/glimpse/writers/sqlite.py

import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional
from ..config import Config
from .base import BaseWriter

class SQLiteWriter(BaseWriter):
    """SQLite backend for trace storage with thread-safe operations."""
    
    def __init__(self, config: Config):
        self._config = config
        self._db_path = config.params.get("db_path", None) or "glimpse_traces.db"
        self._connection_lock = threading.Lock()
        self._local_storage = threading.local()
        
        # Ensure directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self._initialize_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local_storage, 'connection'):
            self._local_storage.connection = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                timeout=30.0  # 30 second timeout
            )
            # Enable WAL mode for better concurrency
            self._local_storage.connection.execute("PRAGMA journal_mode=WAL")
            # Enable foreign keys
            self._local_storage.connection.execute("PRAGMA foreign_keys=ON")
            
        return self._local_storage.connection
    
    def _initialize_database(self):
        """Create tables and indexes if they don't exist."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS trace_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id TEXT NOT NULL UNIQUE,
            call_id TEXT NOT NULL,
            trace_id TEXT,
            level TEXT NOT NULL,
            function TEXT NOT NULL,
            args TEXT,
            stage TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            result TEXT,
            duration_ms REAL,
            exception TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Indexes for common query patterns
        CREATE INDEX IF NOT EXISTS idx_entry_id ON trace_entries(entry_id);
        CREATE INDEX IF NOT EXISTS idx_call_id ON trace_entries(call_id);
        CREATE INDEX IF NOT EXISTS idx_trace_id ON trace_entries(trace_id);
        CREATE INDEX IF NOT EXISTS idx_function ON trace_entries(function);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON trace_entries(timestamp);
        CREATE INDEX IF NOT EXISTS idx_stage ON trace_entries(stage);
        CREATE INDEX IF NOT EXISTS idx_level ON trace_entries(level);
        
        -- Composite indexes for common query combinations
        CREATE INDEX IF NOT EXISTS idx_call_stage ON trace_entries(call_id, stage);
        CREATE INDEX IF NOT EXISTS idx_trace_function ON trace_entries(trace_id, function);
        """
        
        with self._connection_lock:
            conn = self._get_connection()
            try:
                conn.executescript(schema_sql)
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise RuntimeError(f"Failed to initialize SQLite database: {e}")
    
    def write(self, entry: Any) -> None:
        """Write a single trace entry to SQLite database."""
        # Convert dataclass to dict if needed
        if hasattr(entry, "__dict__"):
            entry_dict = entry.__dict__
        elif isinstance(entry, dict):
            entry_dict = entry
        else:
            raise TypeError("Log entry must be a dict or dataclass")
        
        # Prepare SQL insertion
        insert_sql = """
        INSERT INTO trace_entries (
            entry_id, call_id, trace_id, level, function, args, 
            stage, timestamp, result, duration_ms, exception
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # Extract values in correct order
        values = (
            entry_dict.get('entry_id'),
            entry_dict.get('call_id'),
            entry_dict.get('trace_id'),
            entry_dict.get('level'),
            entry_dict.get('function'),
            entry_dict.get('args'),
            entry_dict.get('stage'),
            entry_dict.get('timestamp'),
            entry_dict.get('result'),
            entry_dict.get('duration_ms'),
            entry_dict.get('exception')
        )
        
        conn = self._get_connection()
        try:
            conn.execute(insert_sql, values)
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            raise RuntimeError(f"Failed to write trace entry to SQLite: {e}")
    
    def flush(self) -> None:
        """Flush any pending writes to disk."""
        if hasattr(self._local_storage, 'connection'):
            try:
                self._local_storage.connection.commit()
            except sqlite3.Error as e:
                # Log error but don't raise - flush should be resilient
                print(f"Warning: Failed to flush SQLite connection: {e}")
    
    def close(self) -> None:
        """Close database connections and clean up resources."""
        if hasattr(self._local_storage, 'connection'):
            try:
                self._local_storage.connection.close()
                delattr(self._local_storage, 'connection')
            except sqlite3.Error as e:
                print(f"Warning: Error closing SQLite connection: {e}")
    
    # Additional utility methods for querying (optional but useful)
    
    def get_traces_by_call_id(self, call_id: str) -> list:
        """Get all trace entries for a specific call_id (START/END/EXCEPTION)."""
        query_sql = """
        SELECT * FROM trace_entries 
        WHERE call_id = ? 
        ORDER BY timestamp
        """
        
        conn = self._get_connection()
        try:
            cursor = conn.execute(query_sql, (call_id,))
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to query traces by call_id: {e}")
    
    def get_traces_by_trace_id(self, trace_id: str) -> list:
        """Get all trace entries for a specific trace_id (distributed tracing)."""
        query_sql = """
        SELECT * FROM trace_entries 
        WHERE trace_id = ? 
        ORDER BY timestamp
        """
        
        conn = self._get_connection()
        try:
            cursor = conn.execute(query_sql, (trace_id,))
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to query traces by trace_id: {e}")
    
    def get_slow_functions(self, min_duration_ms: float = 100.0, limit: int = 50) -> list:
        """Get slowest function calls above threshold."""
        query_sql = """
        SELECT function, AVG(duration_ms) as avg_duration, COUNT(*) as call_count,
               MAX(duration_ms) as max_duration, MIN(duration_ms) as min_duration
        FROM trace_entries 
        WHERE stage = 'END' AND duration_ms >= ?
        GROUP BY function
        ORDER BY avg_duration DESC
        LIMIT ?
        """
        
        conn = self._get_connection()
        try:
            cursor = conn.execute(query_sql, (min_duration_ms, limit))
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to query slow functions: {e}")
    
    def get_error_summary(self, hours: int = 24) -> list:
        """Get error summary for the last N hours."""
        query_sql = """
        SELECT function, exception, COUNT(*) as error_count
        FROM trace_entries 
        WHERE stage = 'EXCEPTION' 
        AND datetime(timestamp) >= datetime('now', '-{} hours')
        GROUP BY function, exception
        ORDER BY error_count DESC
        """.format(hours)
        
        conn = self._get_connection()
        try:
            cursor = conn.execute(query_sql)
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to query error summary: {e}")
    
    def cleanup_old_traces(self, days: int = 30) -> int:
        """Remove trace entries older than specified days. Returns count of deleted rows."""
        delete_sql = """
        DELETE FROM trace_entries 
        WHERE datetime(timestamp) < datetime('now', '-{} days')
        """.format(days)
        
        conn = self._get_connection()
        try:
            cursor = conn.execute(delete_sql)
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count
        except sqlite3.Error as e:
            conn.rollback()
            raise RuntimeError(f"Failed to cleanup old traces: {e}")