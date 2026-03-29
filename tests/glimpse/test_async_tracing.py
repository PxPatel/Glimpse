import asyncio
import json
import os
import tempfile

import pytest

from glimpse.config import Config
from glimpse.context import get_active_span, set_active_span, reset_active_span
from glimpse.tracer import Tracer


def _read_spans(path):
    spans = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if obj.get("record_type") == "span":
                    spans.append(obj)
    return spans


@pytest.fixture
def tracer_and_path():
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    tmp_path = tmp.name
    tmp.close()
    config = Config(dest="jsonl", params={"log_path": tmp_path}, env_override=False)
    tracer = Tracer(config)
    yield tracer, tmp_path
    tracer._writer.close()
    os.unlink(tmp_path)


class TestTraceAsyncFunction:

    async def test_trace_async_function_produces_span(self, tracer_and_path):
        tracer, path = tracer_and_path

        @tracer.trace_async_function
        async def add(x, y):
            return x + y

        await add(1, 2)
        tracer._writer.flush()

        spans = _read_spans(path)
        assert len(spans) == 1
        span = spans[0]
        assert "add" in span["name"]
        assert span["start_time"] is not None
        assert span["end_time"] is not None

    async def test_trace_async_function_parent_linking(self, tracer_and_path):
        tracer, path = tracer_and_path

        @tracer.trace_async_function
        async def child():
            pass

        with tracer.span("parent") as parent_span:
            await child()

        tracer._writer.flush()

        spans = _read_spans(path)
        # Find the child span (not "parent")
        child_spans = [s for s in spans if "child" in s["name"]]
        assert len(child_spans) == 1
        assert child_spans[0]["parent_span_id"] == parent_span.span_id

    async def test_trace_async_function_exception_records_error(self, tracer_and_path):
        tracer, path = tracer_and_path

        @tracer.trace_async_function
        async def boom():
            raise ValueError("oops")

        with pytest.raises(ValueError):
            await boom()

        tracer._writer.flush()

        spans = _read_spans(path)
        assert len(spans) == 1
        span = spans[0]
        assert span["status"] == "error"
        events = span["events"]
        assert len(events) == 1
        event = events[0]
        assert event["name"] == "exception"
        assert event["attributes"]["exception.type"] == "ValueError"


class TestAsyncSpanContextManager:

    async def test_async_span_context_manager(self, tracer_and_path):
        tracer, path = tracer_and_path

        async with tracer.async_span("db-query") as span:
            assert span.name == "db-query"
            assert get_active_span() is span
            assert span.end_time is None  # not closed yet inside block

        assert span.end_time is not None
        tracer._writer.flush()

        spans = _read_spans(path)
        assert len(spans) == 1
        assert spans[0]["name"] == "db-query"

    async def test_context_survives_await(self, tracer_and_path):
        tracer, path = tracer_and_path

        async with tracer.async_span("persistent") as span:
            await asyncio.sleep(0)  # yield to event loop
            assert get_active_span() is span
