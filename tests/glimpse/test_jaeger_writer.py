import sys
from unittest.mock import MagicMock, patch

import pytest

from glimpse.span import Span, SpanEvent
from glimpse.writers.jaeger import JaegerWriter


def _make_span(**kwargs) -> Span:
    defaults = dict(
        trace_id="aabbccdd11223344",
        span_id="1122334455667788",
        name="test-span",
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:00:01",
        status="ok",
        events=[],
        attributes={},
    )
    defaults.update(kwargs)
    return Span(**defaults)


@pytest.fixture
def mock_requests(monkeypatch):
    """Patch requests inside the JaegerWriter module after instantiation."""
    fake_requests = MagicMock()
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.text = ""
    fake_requests.post.return_value = fake_response
    return fake_requests


@pytest.fixture
def writer(mock_requests):
    with patch.dict("sys.modules", {"requests": mock_requests}):
        w = JaegerWriter()
    # Inject mock directly so calls after construction are captured
    w._requests = mock_requests
    return w, mock_requests


# ---------------------------------------------------------------------------
# _span_to_otlp shape
# ---------------------------------------------------------------------------

def test_otlp_payload_top_level_keys(writer):
    w, _ = writer
    span = _make_span()
    payload = w._span_to_otlp(span)
    assert "resourceSpans" in payload
    assert len(payload["resourceSpans"]) == 1


def test_otlp_payload_span_fields(writer):
    w, _ = writer
    span = _make_span(
        trace_id="aabbccdd",
        span_id="11223344",
        parent_span_id="00000000",
        name="my-op",
        status="ok",
        attributes={"env": "test"},
    )
    payload = w._span_to_otlp(span)
    otlp_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert otlp_span["traceId"] == "aabbccdd"
    assert otlp_span["spanId"] == "11223344"
    assert otlp_span["parentSpanId"] == "00000000"
    assert otlp_span["name"] == "my-op"
    assert otlp_span["kind"] == 1
    assert otlp_span["status"] == {"code": 1}
    assert otlp_span["attributes"] == [
        {"key": "env", "value": {"stringValue": "test"}}
    ]


def test_otlp_error_status_maps_to_code_2(writer):
    w, _ = writer
    span = _make_span(status="error")
    payload = w._span_to_otlp(span)
    otlp_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert otlp_span["status"] == {"code": 2}


def test_otlp_no_parent_span_id_is_empty_string(writer):
    w, _ = writer
    span = _make_span(parent_span_id=None)
    payload = w._span_to_otlp(span)
    otlp_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert otlp_span["parentSpanId"] == ""


def test_otlp_times_are_unix_nanoseconds(writer):
    w, _ = writer
    span = _make_span(
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:00:01",
    )
    payload = w._span_to_otlp(span)
    otlp_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert isinstance(otlp_span["startTimeUnixNano"], int)
    assert isinstance(otlp_span["endTimeUnixNano"], int)
    # end must be 1 second (1e9 ns) after start
    diff = otlp_span["endTimeUnixNano"] - otlp_span["startTimeUnixNano"]
    assert diff == int(1e9)


def test_otlp_missing_end_time_produces_zero(writer):
    w, _ = writer
    span = _make_span(end_time=None)
    payload = w._span_to_otlp(span)
    otlp_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert otlp_span["endTimeUnixNano"] == 0


def test_otlp_events_serialized(writer):
    w, _ = writer
    event = SpanEvent(
        name="exception",
        timestamp="2026-01-01T00:00:00.500000",
        attributes={"message": "oops"},
    )
    span = _make_span(events=[event])
    payload = w._span_to_otlp(span)
    otlp_events = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["events"]
    assert len(otlp_events) == 1
    assert otlp_events[0]["name"] == "exception"
    assert isinstance(otlp_events[0]["timeUnixNano"], int)
    assert otlp_events[0]["attributes"] == [
        {"key": "message", "value": {"stringValue": "oops"}}
    ]


# ---------------------------------------------------------------------------
# write_span success
# ---------------------------------------------------------------------------

def test_write_span_posts_to_endpoint(writer):
    w, mock_req = writer
    span = _make_span()
    w.write_span(span)
    mock_req.post.assert_called_once()
    call_kwargs = mock_req.post.call_args
    assert call_kwargs[0][0] == "http://localhost:4318/v1/traces"


def test_write_span_sends_content_type_header(writer):
    w, mock_req = writer
    w.write_span(_make_span())
    call_kwargs = mock_req.post.call_args
    headers = call_kwargs[1].get("headers", {})
    assert headers.get("Content-Type") == "application/json"


# ---------------------------------------------------------------------------
# non-2xx response: logs to stderr, does not raise
# ---------------------------------------------------------------------------

def test_write_span_non_2xx_logs_to_stderr(writer, capsys):
    w, mock_req = writer
    mock_req.post.return_value.status_code = 500
    mock_req.post.return_value.text = "Internal Server Error"
    w.write_span(_make_span())  # must not raise
    captured = capsys.readouterr()
    assert "500" in captured.err
    assert "JaegerWriter" in captured.err


def test_write_span_non_2xx_does_not_raise(writer):
    w, mock_req = writer
    mock_req.post.return_value.status_code = 503
    mock_req.post.return_value.text = "Unavailable"
    w.write_span(_make_span())  # must not raise


# ---------------------------------------------------------------------------
# connection error: logs to stderr, does not raise
# ---------------------------------------------------------------------------

def test_write_span_connection_error_logs_to_stderr(writer, capsys):
    w, mock_req = writer
    mock_req.post.side_effect = ConnectionError("refused")
    w.write_span(_make_span())  # must not raise
    captured = capsys.readouterr()
    assert "JaegerWriter" in captured.err


def test_write_span_connection_error_does_not_raise(writer):
    w, mock_req = writer
    mock_req.post.side_effect = OSError("network unreachable")
    w.write_span(_make_span())  # must not raise


# ---------------------------------------------------------------------------
# write() is a no-op
# ---------------------------------------------------------------------------

def test_write_is_noop(writer):
    w, mock_req = writer
    w.write({"anything": True})  # must not raise and must not call requests
    mock_req.post.assert_not_called()


# ---------------------------------------------------------------------------
# ImportError when requests is absent
# ---------------------------------------------------------------------------

def test_import_error_when_requests_missing():
    with patch.dict("sys.modules", {"requests": None}):
        with pytest.raises(ImportError, match="pip install glimpse\\[jaeger\\]"):
            JaegerWriter()
