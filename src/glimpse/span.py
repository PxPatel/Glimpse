from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class SpanEvent:
    name: str
    timestamp: str  # ISO format string, e.g. datetime.utcnow().isoformat()
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    trace_id: str
    span_id: str
    name: str
    start_time: str  # ISO format string
    parent_span_id: Optional[str] = None
    end_time: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"  # "ok" or "error"
    events: List[SpanEvent] = field(default_factory=list)
