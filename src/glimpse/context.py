from contextvars import ContextVar
from typing import Optional

# Module-level var — one per process, works across sync and async
_active_span: ContextVar[Optional["Span"]] = ContextVar("glimpse_active_span", default=None)


def get_active_span() -> Optional["Span"]:
    """Return the currently active span, or None if no span is active."""
    return _active_span.get()


def set_active_span(span: Optional["Span"]):
    """Set the active span. Returns the Token needed to reset."""
    return _active_span.set(span)


def reset_active_span(token) -> None:
    """Reset the active span to its previous value using the token from set_active_span."""
    _active_span.reset(token)
