# Requirements: Glimpse

**Defined:** 2026-03-29
**Core Value:** Give developers easy, low-friction visibility into Python application execution — from function-level tracing to basic distributed trace graphs — without the ceremony of a full APM setup.

## v1 Requirements

### Bug Fixes

- [x] **BUG-01**: SQLite backend initializes correctly (fix `"sqllite"` → `"sqlite"` typo in logwriter.py)
- [x] **BUG-02**: SQLite schema creates successfully (fix trailing comma in CREATE TABLE statement)
- [x] **BUG-03**: Tracer releases all resources on stop (add `close()` call alongside `flush()` in `Tracer.stop()`)
- [x] **BUG-04**: Exception traces preserve original stack (fix re-raise from `raise Exception(str(e))` to bare `raise`)
- [x] **BUG-05**: Writer failures do not crash user code (wrap writer operations in try/except; log errors to stderr)

### Span Model

- [ ] **SPAN-01**: Library exposes a `Span` dataclass with `trace_id`, `span_id`, `parent_span_id`, `name`, `start_time`, `end_time`, `attributes`, `status`, and `events` fields
- [ ] **SPAN-02**: Active span context is tracked via `contextvars.ContextVar` so child spans automatically find their parent
- [ ] **SPAN-03**: User can create a span with a context manager (`with tracer.span("operation") as span:`)
- [ ] **SPAN-04**: Spans created by `sys.settrace` auto-instrumentation attach to the current active span as children
- [ ] **SPAN-05**: Span status is set to `"error"` and exception recorded when an exception propagates through the span
- [ ] **SPAN-06**: JSON writer outputs spans in a structured, human-readable format (separate from legacy LogEntry output)

### Async Support

- [x] **ASYNC-01**: User can trace an async function with `@tracer.trace_async_function` decorator
- [x] **ASYNC-02**: Async spans correctly propagate parent context across `await` boundaries using `ContextVar`
- [x] **ASYNC-03**: Async context manager works (`async with tracer.span("name"):`)
- [x] **ASYNC-04**: Existing synchronous tracing continues to work unchanged alongside async tracing

### Observability & Export

- [ ] **OBS-01**: Spans can be exported to a local Jaeger instance via HTTP (optional dependency: `requests`)
- [x] **OBS-02**: User can run Glimpse with Jaeger export by installing the `jaeger` extra (`pip install glimpse[jaeger]`)
- [x] **OBS-03**: Jaeger export failures are logged to stderr and do not raise exceptions in user code

## v2 Requirements

These are deferred — interesting but not needed for the core learning goals.

### Sampling

- **SAMP-01**: User can configure a sampling rate (e.g., trace 10% of calls) to reduce output volume
- **SAMP-02**: Sampler is pluggable (always-on, probabilistic, rate-limiting)

### Context Propagation (Cross-Process)

- **PROP-01**: Tracer can inject trace context into a dict/HTTP headers for cross-service propagation
- **PROP-02**: Tracer can extract trace context from incoming dict/HTTP headers to continue a distributed trace
- **PROP-03**: `tracer.span("name", context=ctx)` accepts extracted propagation context and automatically links the span to the remote parent (backward compatible — context is optional)
- **PROP-04**: FastAPI/Starlette middleware automatically extracts traceparent from incoming requests and sets up the active span context for the duration of the request

### PII Protection

- **PII-01**: User can configure field exclusions (e.g., exclude function arguments named `password`, `token`, `api_key`)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full OpenTelemetry SDK compatibility | Educational project, not a drop-in replacement |
| OTLP protobuf/gRPC export | Adds grpcio build complexity; HTTP JSON is sufficient |
| Automatic HTTP framework middleware | Too much magic; explicit spans are more educational |
| APM dashboard or UI | Export to Jaeger/files; don't build UI |
| Production-grade reliability guarantees | Hobby project scope |
| Metrics or log collection | Different domain; stay focused on tracing |
| Language support beyond Python | Single-language focus |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BUG-01 | Phase 1 | Complete |
| BUG-02 | Phase 1 | Complete |
| BUG-03 | Phase 1 | Complete |
| BUG-04 | Phase 1 | Complete |
| BUG-05 | Phase 1 | Complete |
| SPAN-01 | Phase 2 | Pending |
| SPAN-02 | Phase 2 | Pending |
| SPAN-03 | Phase 2 | Pending |
| SPAN-04 | Phase 2 | Pending |
| SPAN-05 | Phase 2 | Pending |
| SPAN-06 | Phase 2 | Pending |
| ASYNC-01 | Phase 3 | Complete |
| ASYNC-02 | Phase 3 | Complete |
| ASYNC-03 | Phase 3 | Complete |
| ASYNC-04 | Phase 3 | Complete |
| OBS-01 | Phase 4 | Complete |
| OBS-02 | Phase 4 | Complete |
| OBS-03 | Phase 4 | Complete |
| PROP-01 | Phase 5 | Complete |
| PROP-02 | Phase 5 | Complete |
