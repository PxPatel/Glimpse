"""
Tests for glimpse.propagation — inject, extract, round-trip, and edge cases.

Active span state is managed manually with set_active_span / reset_active_span
so each test is fully isolated from the global ContextVar.
"""
import pytest

from glimpse.span import Span
from glimpse.context import set_active_span, reset_active_span
from glimpse.propagation import inject, extract
from glimpse.tracer import Tracer
from glimpse.config import Config

_START = "2026-01-01T00:00:00"


# ---------------------------------------------------------------------------
# inject — happy path
# ---------------------------------------------------------------------------

def test_inject_writes_traceparent_header():
    span = Span(trace_id="a" * 32, span_id="b" * 16, name="svc", start_time=_START)
    token = set_active_span(span)
    try:
        headers = {}
        inject(headers)
        assert headers["traceparent"] == f"00-{'a' * 32}-{'b' * 16}-01"
    finally:
        reset_active_span(token)


# ---------------------------------------------------------------------------
# inject — no active span (no-op)
# ---------------------------------------------------------------------------

def test_inject_no_active_span_is_noop():
    # Explicitly clear any active span
    token = set_active_span(None)
    try:
        headers = {}
        inject(headers)
        assert "traceparent" not in headers
    finally:
        reset_active_span(token)


# ---------------------------------------------------------------------------
# inject — preserves existing headers
# ---------------------------------------------------------------------------

def test_inject_preserves_existing_headers():
    span = Span(trace_id="a" * 32, span_id="b" * 16, name="svc", start_time=_START)
    token = set_active_span(span)
    try:
        headers = {"content-type": "application/json"}
        inject(headers)
        assert "traceparent" in headers
        assert "content-type" in headers
        assert headers["content-type"] == "application/json"
    finally:
        reset_active_span(token)


# ---------------------------------------------------------------------------
# extract — valid traceparent
# ---------------------------------------------------------------------------

def test_extract_valid_traceparent():
    headers = {"traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"}
    ctx = extract(headers)
    assert ctx is not None
    assert ctx["trace_id"] == "a" * 32
    assert ctx["parent_span_id"] == "b" * 16


# ---------------------------------------------------------------------------
# extract — no traceparent header
# ---------------------------------------------------------------------------

def test_extract_missing_header_returns_none():
    assert extract({}) is None


# ---------------------------------------------------------------------------
# extract — malformed traceparent (too short)
# ---------------------------------------------------------------------------

def test_extract_malformed_too_short_returns_none():
    assert extract({"traceparent": "bad-value"}) is None


# ---------------------------------------------------------------------------
# extract — malformed traceparent (wrong field lengths)
# ---------------------------------------------------------------------------

def test_extract_malformed_wrong_field_lengths_returns_none():
    assert extract({"traceparent": "00-tooshort-tooshort-01"}) is None


# ---------------------------------------------------------------------------
# extract — case-sensitive header key (exact match only)
# ---------------------------------------------------------------------------

def test_extract_case_sensitive_key():
    # This is a plain dict, not an HTTP framework. The key must be exactly
    # "traceparent" (lowercase). "Traceparent" is a different key and should
    # return None.
    headers = {"Traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"}
    assert extract(headers) is None


# ---------------------------------------------------------------------------
# round-trip: inject then extract produces continuous trace
# ---------------------------------------------------------------------------

def test_round_trip_inject_extract_continuity():
    upstream = Span(
        trace_id="c" * 32,
        span_id="d" * 16,
        name="upstream",
        start_time=_START,
    )

    # Sending side
    headers = {}
    token = set_active_span(upstream)
    try:
        inject(headers)
    finally:
        reset_active_span(token)

    # Receiving side
    ctx = extract(headers)
    assert ctx is not None
    assert ctx["trace_id"] == "c" * 32
    assert ctx["parent_span_id"] == "d" * 16

    # Downstream span continues the same trace
    child = Span(
        trace_id=ctx["trace_id"],
        span_id="e" * 16,
        name="downstream",
        start_time=_START,
        parent_span_id=ctx["parent_span_id"],
    )
    assert child.trace_id == "c" * 32          # same trace
    assert child.parent_span_id == "d" * 16    # upstream span is the parent


# ---------------------------------------------------------------------------
# tracer.inject and tracer.extract delegate to propagation module
# ---------------------------------------------------------------------------

def test_tracer_inject_and_extract_delegate():
    config = Config(dest="json", env_override=False)
    tracer = Tracer(config, writer_initiation=False)

    headers = {}
    with tracer.span("test"):
        tracer.inject(headers)

    assert "traceparent" in headers

    ctx = tracer.extract(headers)
    assert ctx is not None
    assert "trace_id" in ctx
    assert "parent_span_id" in ctx
