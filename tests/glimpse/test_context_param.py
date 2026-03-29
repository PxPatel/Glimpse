"""Tests for tracer.span(context=...) and tracer.async_span(context=...)."""
import pytest
from unittest.mock import MagicMock
from glimpse.tracer import Tracer
from glimpse.config import Config


def make_tracer():
    config = Config(dest="json", level="DEBUG",
                    max_field_length=200, enable_trace_id=True)
    t = Tracer(config, writer_initiation=False)
    t._on_span_end = MagicMock()
    return t


REMOTE_TRACE_ID = "a" * 32
REMOTE_SPAN_ID = "b" * 16
REMOTE_CTX = {"trace_id": REMOTE_TRACE_ID, "parent_span_id": REMOTE_SPAN_ID}


class TestSpanContextParam:
    def test_context_sets_trace_id(self):
        t = make_tracer()
        with t.span("op", context=REMOTE_CTX) as span:
            assert span.trace_id == REMOTE_TRACE_ID

    def test_context_sets_parent_span_id(self):
        t = make_tracer()
        with t.span("op", context=REMOTE_CTX) as span:
            assert span.parent_span_id == REMOTE_SPAN_ID

    def test_no_context_is_backward_compatible(self):
        t = make_tracer()
        with t.span("op") as span:
            # No remote context — span starts a fresh trace
            assert span.trace_id is not None
            assert span.parent_span_id is None

    def test_nested_span_still_links_to_parent(self):
        """Inner span with no context still links to in-process parent."""
        t = make_tracer()
        with t.span("outer", context=REMOTE_CTX) as outer:
            with t.span("inner") as inner:
                assert inner.parent_span_id == outer.span_id
                assert inner.trace_id == outer.trace_id

    def test_context_none_treated_as_no_context(self):
        t = make_tracer()
        with t.span("op", context=None) as span:
            assert span.parent_span_id is None


class TestAsyncSpanContextParam:
    @pytest.mark.asyncio
    async def test_async_context_sets_trace_id(self):
        t = make_tracer()
        async with t.async_span("op", context=REMOTE_CTX) as span:
            assert span.trace_id == REMOTE_TRACE_ID

    @pytest.mark.asyncio
    async def test_async_context_sets_parent_span_id(self):
        t = make_tracer()
        async with t.async_span("op", context=REMOTE_CTX) as span:
            assert span.parent_span_id == REMOTE_SPAN_ID

    @pytest.mark.asyncio
    async def test_async_no_context_backward_compatible(self):
        t = make_tracer()
        async with t.async_span("op") as span:
            assert span.trace_id is not None
            assert span.parent_span_id is None
