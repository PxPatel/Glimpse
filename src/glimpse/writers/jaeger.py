import sys
from datetime import datetime
from .base import BaseWriter


class JaegerWriter(BaseWriter):
    """
    Writer that sends spans to a Jaeger OTLP HTTP endpoint as JSON.

    Requires the `requests` library. Install via:
        pip install glimpse[jaeger]

    All export failures are caught and logged to stderr — never raised.
    """

    def __init__(self, endpoint: str = "http://localhost:4318/v1/traces"):
        self._endpoint = endpoint
        try:
            import requests
            self._requests = requests
        except ImportError:
            raise ImportError(
                "requests is required for Jaeger export: pip install glimpse[jaeger]"
            )

    def _span_to_otlp(self, span) -> dict:
        start_ns = int(datetime.fromisoformat(span.start_time).timestamp() * 1e9)
        end_ns = (
            int(datetime.fromisoformat(span.end_time).timestamp() * 1e9)
            if span.end_time
            else 0
        )

        events = [
            {
                "name": e.name,
                "timeUnixNano": int(
                    datetime.fromisoformat(e.timestamp).timestamp() * 1e9
                ),
                "attributes": [
                    {"key": k, "value": {"stringValue": str(v)}}
                    for k, v in e.attributes.items()
                ],
            }
            for e in span.events
        ]

        return {
            "resourceSpans": [
                {
                    "resource": {"attributes": []},
                    "scopeSpans": [
                        {
                            "scope": {"name": "glimpse"},
                            "spans": [
                                {
                                    "traceId": span.trace_id,
                                    "spanId": span.span_id,
                                    "parentSpanId": span.parent_span_id or "",
                                    "name": span.name,
                                    "kind": 1,
                                    "startTimeUnixNano": start_ns,
                                    "endTimeUnixNano": end_ns,
                                    "attributes": [
                                        {"key": k, "value": {"stringValue": str(v)}}
                                        for k, v in span.attributes.items()
                                    ],
                                    "status": {
                                        "code": 2 if span.status == "error" else 1
                                    },
                                    "events": events,
                                }
                            ],
                        }
                    ],
                }
            ]
        }

    def write_span(self, span) -> None:
        try:
            payload = self._span_to_otlp(span)
            response = self._requests.post(
                self._endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if not (200 <= response.status_code < 300):
                print(
                    f"Glimpse JaegerWriter: export failed ({response.status_code}): "
                    f"{response.text[:200]}",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(
                f"Glimpse JaegerWriter: export error: {exc}",
                file=sys.stderr,
            )

    def write(self, entry) -> None:
        # No-op: spans are the only meaningful output for Jaeger
        pass
