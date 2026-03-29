import re
from typing import Dict, Optional
from .context import get_active_span

_TRACEPARENT_HEADER = "traceparent"
_TRACEPARENT_VERSION = "00"
_TRACEPARENT_FLAGS = "01"  # sampled

# Regex: version(2hex)-trace_id(32hex)-parent_id(16hex)-flags(2hex)
_TRACEPARENT_PATTERN = re.compile(
    r"^[0-9a-f]{2}-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$"
)


def inject(headers: Dict[str, str]) -> None:
    """
    Inject the active span's trace context into `headers` as a W3C traceparent.

    Mutates `headers` in-place. No-op if no span is currently active.

    Format: 00-{trace_id_32hex}-{span_id_16hex}-01
    """
    span = get_active_span()
    if span is None:
        return
    headers[_TRACEPARENT_HEADER] = (
        f"{_TRACEPARENT_VERSION}-{span.trace_id}-{span.span_id}-{_TRACEPARENT_FLAGS}"
    )


def extract(headers: Dict[str, str]) -> Optional[Dict[str, str]]:
    """
    Extract trace context from `headers` and return a propagation context dict.

    Returns a dict with keys:
        - "trace_id":       32-hex str — the upstream trace ID
        - "parent_span_id": 16-hex str — the upstream span ID (becomes our parent)

    Returns None if the traceparent header is absent or does not match the
    expected format. Malformed values are silently ignored (returns None).
    """
    raw = headers.get(_TRACEPARENT_HEADER)
    if raw is None:
        return None
    match = _TRACEPARENT_PATTERN.match(raw.strip())
    if not match:
        return None
    return {
        "trace_id": match.group(1),
        "parent_span_id": match.group(2),
    }
