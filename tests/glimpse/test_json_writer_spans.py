import json
import os
import tempfile

import pytest

from glimpse.config import Config
from glimpse.span import Span, SpanEvent
from glimpse.writers.json import JSONWriter
from glimpse.writers.logentry import LogEntry
from datetime import datetime


@pytest.fixture
def writer_and_path():
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    tmp_path = tmp.name
    tmp.close()
    config = Config(params={"log_path": tmp_path})
    writer = JSONWriter(config)
    yield writer, tmp_path
    writer.close()
    os.unlink(tmp_path)


def _read_last_line(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return json.loads(lines[-1])


def _make_span(**kwargs) -> Span:
    defaults = dict(
        trace_id="trace-001",
        span_id="span-001",
        name="test-span",
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:00:01",
        status="ok",
        events=[],
        attributes={},
    )
    defaults.update(kwargs)
    return Span(**defaults)


def test_write_span_produces_parseable_json(writer_and_path):
    writer, path = writer_and_path
    span = _make_span()
    writer.write(span)
    writer.flush()
    record = _read_last_line(path)
    assert isinstance(record, dict)


def test_write_span_has_record_type_span(writer_and_path):
    writer, path = writer_and_path
    span = _make_span()
    writer.write(span)
    writer.flush()
    record = _read_last_line(path)
    assert record["record_type"] == "span"


def test_write_span_contains_all_span_fields(writer_and_path):
    writer, path = writer_and_path
    span = _make_span(attributes={"key": "value"})
    writer.write(span)
    writer.flush()
    record = _read_last_line(path)
    for field in ("trace_id", "span_id", "name", "start_time", "status", "events", "attributes"):
        assert field in record, f"Missing field: {field}"


def test_write_log_entry_has_record_type_log_entry(writer_and_path):
    writer, path = writer_and_path
    entry = LogEntry(
        entry_id=1,
        call_id="call-001",
        trace_id="trace-001",
        function="my_func",
        level="INFO",
        args="()",
        stage="call",
        timestamp=datetime(2026, 1, 1),
    )
    writer.write(entry)
    writer.flush()
    record = _read_last_line(path)
    assert record["record_type"] == "log_entry"


def test_write_span_via_write_span_method(writer_and_path):
    writer, path = writer_and_path
    span = _make_span()
    writer.write_span(span)
    writer.flush()
    record = _read_last_line(path)
    assert record["record_type"] == "span"
    assert record["name"] == "test-span"


def test_span_with_event_serialized_as_nested_dict(writer_and_path):
    writer, path = writer_and_path
    event = SpanEvent(name="my-event", timestamp="2026-01-01T00:00:00.500", attributes={"x": 1})
    span = _make_span(events=[event])
    writer.write(span)
    writer.flush()
    record = _read_last_line(path)
    assert len(record["events"]) == 1
    serialized_event = record["events"][0]
    assert isinstance(serialized_event, dict)
    assert serialized_event["name"] == "my-event"
    assert serialized_event["attributes"] == {"x": 1}
