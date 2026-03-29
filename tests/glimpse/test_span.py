"""Tests for Span dataclass and ContextVar-based active span tracking."""
import pytest
from glimpse import Span, SpanEvent, get_active_span, set_active_span, reset_active_span


class TestSpanDefaults:
    def test_status_defaults_to_ok(self):
        span = Span(trace_id="t1", span_id="s1", name="test", start_time="2024-01-01T00:00:00")
        assert span.status == "ok"

    def test_events_defaults_to_empty_list(self):
        span = Span(trace_id="t1", span_id="s1", name="test", start_time="2024-01-01T00:00:00")
        assert span.events == []

    def test_attributes_defaults_to_empty_dict(self):
        span = Span(trace_id="t1", span_id="s1", name="test", start_time="2024-01-01T00:00:00")
        assert span.attributes == {}

    def test_parent_span_id_defaults_to_none(self):
        span = Span(trace_id="t1", span_id="s1", name="test", start_time="2024-01-01T00:00:00")
        assert span.parent_span_id is None

    def test_end_time_defaults_to_none(self):
        span = Span(trace_id="t1", span_id="s1", name="test", start_time="2024-01-01T00:00:00")
        assert span.end_time is None


class TestSpanEvent:
    def test_instantiation_with_name_and_timestamp(self):
        event = SpanEvent(name="cache_miss", timestamp="2024-01-01T00:00:00")
        assert event.name == "cache_miss"
        assert event.timestamp == "2024-01-01T00:00:00"
        assert event.attributes == {}


class TestActiveSpanContext:
    def test_get_active_span_returns_none_initially(self):
        # Reset to clean state first
        token = set_active_span(None)
        reset_active_span(token)
        assert get_active_span() is None

    def test_set_active_span_makes_span_accessible(self):
        span = Span(trace_id="t1", span_id="s1", name="test", start_time="2024-01-01T00:00:00")
        token = set_active_span(span)
        try:
            assert get_active_span() is span
        finally:
            reset_active_span(token)

    def test_reset_active_span_restores_none(self):
        span = Span(trace_id="t1", span_id="s1", name="test", start_time="2024-01-01T00:00:00")
        token = set_active_span(span)
        reset_active_span(token)
        assert get_active_span() is None

    def test_nested_span_context_restores_outer_span(self):
        outer = Span(trace_id="t1", span_id="s1", name="outer", start_time="2024-01-01T00:00:00")
        inner = Span(trace_id="t1", span_id="s2", name="inner", start_time="2024-01-01T00:00:01")

        outer_token = set_active_span(outer)
        assert get_active_span() is outer

        inner_token = set_active_span(inner)
        assert get_active_span() is inner

        reset_active_span(inner_token)
        assert get_active_span() is outer

        reset_active_span(outer_token)
        assert get_active_span() is None
