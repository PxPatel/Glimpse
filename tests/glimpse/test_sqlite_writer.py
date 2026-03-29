"""Tests for SQLiteWriter — specifically BUG-02: trailing comma in CREATE TABLE."""
import os
import tempfile
import pytest

from glimpse.config import Config
from glimpse.writers.sqlite import SQLiteWriter
from glimpse.writers.logentry import LogEntry
from datetime import datetime


def make_config(db_path: str) -> Config:
    return Config(dest="sqlite", params={"db_path": db_path}, env_override=False)


class TestSQLiteWriterInit:
    """BUG-02: SQLiteWriter.__init__ must not raise due to SQL syntax errors."""

    def test_init_does_not_raise(self, tmp_path):
        """SQLiteWriter(config) with a temp db_path should not raise any exception."""
        db_path = str(tmp_path / "test.db")
        cfg = make_config(db_path)
        writer = SQLiteWriter(cfg)
        writer.close()

    def test_trace_entries_table_exists(self, tmp_path):
        """After __init__, the trace_entries table must exist in the database."""
        import sqlite3

        db_path = str(tmp_path / "test.db")
        cfg = make_config(db_path)
        writer = SQLiteWriter(cfg)
        writer.close()

        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trace_entries'"
        )
        row = cursor.fetchone()
        conn.close()
        assert row is not None, "trace_entries table was not created"

    def test_write_inserts_row(self, tmp_path):
        """write() should insert a row; COUNT(*) returns 1 after a single write."""
        import sqlite3

        db_path = str(tmp_path / "test.db")
        cfg = make_config(db_path)
        writer = SQLiteWriter(cfg)

        entry = LogEntry(
            entry_id="e-001",
            call_id="c-001",
            trace_id="t-001",
            function="test_func",
            level="INFO",
            args="{}",
            stage="START",
            timestamp=datetime.utcnow().isoformat(),
        )
        writer.write(entry)
        writer.close()

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM trace_entries")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1, f"Expected 1 row after write(), got {count}"
