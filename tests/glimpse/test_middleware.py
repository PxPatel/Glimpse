"""Integration tests for GlimpseMiddleware using Starlette TestClient."""
import pytest
from unittest.mock import MagicMock
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from glimpse.tracer import Tracer
from glimpse.config import Config
from glimpse.middleware import GlimpseMiddleware


def make_tracer():
    config = Config(dest="json", level="DEBUG",
                    max_field_length=200, enable_trace_id=True)
    t = Tracer(config, writer_initiation=False)
    return t


def build_app(tracer):
    captured = []
    tracer._on_span_end = lambda span: captured.append(span)

    async def homepage(request: Request):
        # Create a child span inside the handler
        with tracer.span("handler-work") as s:
            pass
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/", homepage)])
    app.add_middleware(GlimpseMiddleware, tracer=tracer)
    return app, captured


REMOTE_TRACE_ID = "c" * 32
REMOTE_SPAN_ID = "d" * 16
VALID_TRACEPARENT = f"00-{REMOTE_TRACE_ID}-{REMOTE_SPAN_ID}-01"


class TestGlimpseMiddleware:
    def test_request_without_traceparent_succeeds(self):
        tracer = make_tracer()
        app, captured = build_app(tracer)
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200

    def test_request_without_traceparent_creates_spans(self):
        tracer = make_tracer()
        app, captured = build_app(tracer)
        client = TestClient(app)
        client.get("/")
        # Middleware root span + handler child span
        assert len(captured) == 2

    def test_request_with_traceparent_links_root_span(self):
        tracer = make_tracer()
        app, captured = build_app(tracer)
        client = TestClient(app)
        client.get("/", headers={"traceparent": VALID_TRACEPARENT})
        root_span = next(s for s in captured if s.name == "GET /")
        assert root_span.trace_id == REMOTE_TRACE_ID
        assert root_span.parent_span_id == REMOTE_SPAN_ID

    def test_handler_child_span_inherits_trace_id(self):
        tracer = make_tracer()
        app, captured = build_app(tracer)
        client = TestClient(app)
        client.get("/", headers={"traceparent": VALID_TRACEPARENT})
        child_span = next(s for s in captured if s.name == "handler-work")
        assert child_span.trace_id == REMOTE_TRACE_ID

    def test_handler_child_span_parent_is_middleware_root(self):
        tracer = make_tracer()
        app, captured = build_app(tracer)
        client = TestClient(app)
        client.get("/", headers={"traceparent": VALID_TRACEPARENT})
        root_span = next(s for s in captured if s.name == "GET /")
        child_span = next(s for s in captured if s.name == "handler-work")
        assert child_span.parent_span_id == root_span.span_id

    def test_malformed_traceparent_ignored_fresh_trace(self):
        tracer = make_tracer()
        app, captured = build_app(tracer)
        client = TestClient(app)
        client.get("/", headers={"traceparent": "not-valid"})
        root_span = next(s for s in captured if s.name == "GET /")
        # Malformed header -> extract() returns None -> fresh trace
        assert root_span.trace_id != REMOTE_TRACE_ID

    def test_span_name_includes_method_and_path(self):
        tracer = make_tracer()
        app, captured = build_app(tracer)
        client = TestClient(app)
        client.get("/")
        names = [s.name for s in captured]
        assert "GET /" in names
