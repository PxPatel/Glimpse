"""
Starlette/FastAPI middleware for automatic traceparent extraction.

This module is an optional dependency. Install with:
    pip install glimpse[starlette]
"""

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.types import ASGIApp
except ImportError as exc:
    raise ImportError(
        "glimpse[starlette] is required to use GlimpseMiddleware. "
        "Install it with: pip install glimpse[starlette]"
    ) from exc

from typing import Callable


class GlimpseMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware that extracts W3C traceparent from incoming requests
    and opens a root span for the duration of each request.

    Usage (FastAPI):
        app = FastAPI()
        app.add_middleware(GlimpseMiddleware, tracer=tracer)

    Usage (Starlette):
        app = Starlette()
        app.add_middleware(GlimpseMiddleware, tracer=tracer)

    Any tracer.span() calls inside route handlers will automatically
    inherit the remote parent — no header handling needed in handlers.
    """

    def __init__(self, app: ASGIApp, tracer) -> None:
        super().__init__(app)
        self._tracer = tracer

    async def dispatch(self, request: Request, call_next: Callable):
        headers = dict(request.headers)
        ctx = self._tracer.extract(headers)
        span_name = f"{request.method} {request.url.path}"

        async with self._tracer.async_span(span_name, context=ctx):
            response = await call_next(request)
        return response
