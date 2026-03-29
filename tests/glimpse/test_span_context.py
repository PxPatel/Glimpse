import pytest
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock
from glimpse.config import Config
from glimpse.tracer import Tracer
from glimpse.span import Span, SpanEvent
from glimpse.context import get_active_span, set_active_span, reset_active_span


@pytest.fixture
def config():
    return Config(dest="jsonl", level="INFO", max_field_length=100, env_override=False)


@pytest.fixture
def tracer(config):
    return Tracer(config, writer_initiation=False)


class TestSpanContextManager:

    def test_span_returns_span_with_name_and_times(self, tracer):
        with tracer.span("op") as span:
            assert isinstance(span, Span)
            assert span.name == "op"
            assert span.start_time is not None
            assert span.status == "ok"
            in_block_end_time = span.end_time
        # end_time is set after block exits
        assert span.end_time is not None
        assert in_block_end_time is None  # was None inside the block

    def test_nested_span_parent_linking(self, tracer):
        with tracer.span("outer") as outer:
            with tracer.span("inner") as inner:
                assert inner.parent_span_id == outer.span_id

    def test_active_span_restored_after_outer_block(self, tracer):
        with tracer.span("outer"):
            pass
        assert get_active_span() is None

    def test_exception_sets_error_status_and_event(self, tracer):
        with pytest.raises(ValueError):
            with tracer.span("failing") as span:
                raise ValueError("boom")
        assert span.status == "error"
        assert len(span.events) == 1
        event = span.events[0]
        assert event.name == "exception"
        assert event.attributes["exception.type"] == "ValueError"
        assert event.attributes["exception.message"] == "boom"

    def test_exception_propagates(self, tracer):
        with pytest.raises(RuntimeError, match="propagated"):
            with tracer.span("err"):
                raise RuntimeError("propagated")

    def test_active_span_restored_after_exception(self, tracer):
        with pytest.raises(Exception):
            with tracer.span("err"):
                raise Exception("test")
        assert get_active_span() is None


class TestAutoInstrumentationParentLinking:

    def test_handle_function_call_stores_parent_span_id(self, tracer, config):
        """_handle_function_call captures parent_span_id from active span."""
        from glimpse.policy import TracingPolicy
        policy = TracingPolicy(exact_modules=["__main__"], trace_depth=10)
        tracer._policy = policy

        # Build a minimal frame-like object using a real frame from sys._getframe
        # We'll use a mock frame that matches the policy module
        mock_frame = MagicMock()
        mock_frame.f_code.co_name = "my_func"
        mock_frame.f_globals = {"__name__": "__main__"}
        mock_frame.f_code.co_filename = "main.py"
        mock_frame.f_locals = {}
        mock_frame.f_code.co_varnames = ()
        mock_frame.f_code.co_argcount = 0

        # Set an active span
        span = Span(
            trace_id="trace123",
            span_id="span456",
            name="parent",
            start_time="2024-01-01T00:00:00",
        )
        token = set_active_span(span)
        try:
            tracer._handle_function_call(mock_frame)
            call_info = tracer._call_metadata.get(mock_frame)
            assert call_info is not None
            assert call_info["parent_span_id"] == "span456"
            assert call_info["trace_id"] == "trace123"
        finally:
            reset_active_span(token)
            tracer._call_metadata.clear()

    def test_handle_function_call_no_active_span(self, tracer, config):
        """Without active span, parent_span_id is None."""
        from glimpse.policy import TracingPolicy
        policy = TracingPolicy(exact_modules=["__main__"], trace_depth=10)
        tracer._policy = policy

        mock_frame = MagicMock()
        mock_frame.f_code.co_name = "my_func"
        mock_frame.f_globals = {"__name__": "__main__"}
        mock_frame.f_code.co_filename = "main.py"
        mock_frame.f_locals = {}
        mock_frame.f_code.co_varnames = ()
        mock_frame.f_code.co_argcount = 0

        # Ensure no active span
        assert get_active_span() is None

        tracer._handle_function_call(mock_frame)
        call_info = tracer._call_metadata.get(mock_frame)
        assert call_info is not None
        assert call_info["parent_span_id"] is None
        tracer._call_metadata.clear()
