# Glimpse

**Function-level logging and tracing for Python.**

Inspired by **OpenTelemetry** and **Datadog**—this is a small, from-scratch take on traces and spans. Built to learn how the pieces fit together.

**Capabilities:** policy-driven tracing via `sys.settrace`, manual spans (JSONL / SQLite / **Jaeger** OTLP), W3C **`traceparent`** inject/extract, and optional **Starlette/FastAPI** middleware for a root span per request.

---

## What’s in the box

A short overview of what the library implements:

- Automatic tracing — `tracer.run()` instruments calls that match your policy (no per-function decorators required).
- Spans — `span()` / `async_span()` blocks with active-span context, parent IDs, and export where supported (including OTLP to Jaeger).
- Header propagation — inject/extract W3C `traceparent`; optional ASGI middleware attaches inbound headers to handler spans.
- Outputs — JSON/JSONL, SQLite, and Jaeger; you can combine destinations.
- Async — `trace_async_function` and async span context managers use `contextvars` for nesting.

---

## Installation

From the project / PyPI package name in `pyproject.toml`:

```bash
pip install glimpse
```

Optional extras:

```bash
pip install glimpse[jaeger]      # OTLP HTTP export to Jaeger (needs `requests`)
pip install glimpse[starlette]   # ASGI middleware (Starlette/FastAPI)
pip install glimpse[dev]        # pytest, pytest-asyncio (for contributors)
```

Editable install for development:

```bash
pip install -e ".[dev,jaeger,starlette]"
```

`glimpse.config` uses **`python-dotenv`** to load `.env` files. If you install from source, ensure `python-dotenv` is available (it should be listed as a runtime dependency in published packages).

---

## Quick start

```python
from glimpse.tracer import Tracer
from glimpse.config import Config
from glimpse.policy import TracingPolicy

config = Config(dest="jsonl", level="INFO", params={"log_path": "traces.jsonl"})
policy = TracingPolicy(exact_modules=["myapp"], package_trees=["myapp.services"])

tracer = Tracer(config, policy=policy)
tracer.run()

# Your application code runs here; matching calls are traced automatically.

tracer.stop()
```

Use a **policy** whenever you call `run()` (automatic tracing). For manual spans only, you can still construct `Tracer` with a config and use `span()` / `async_span()` without `run()`.

---

## Core features

### Automatic tracing (`sys.settrace`)

`tracer.run()` / `tracer.stop()` enable call-level logging for modules matched by `TracingPolicy` (depth limits, exact modules vs package trees, wildcards — unchanged from the policy model).

### Decorators

- **`@tracer.trace_function`** — Sync functions: classic START/END/EXCEPTION log lines with timing.
- **`@tracer.trace_async_function`** — Async functions: creates a **span** per invocation, nested under the active span when present.

### Spans (`Span`, `SpanEvent`)

Spans carry `trace_id`, `span_id`, optional `parent_span_id`, timing, status, attributes, and events (e.g. exceptions). The active span is stored in a **context variable** (`get_active_span` / `set_active_span` / `reset_active_span`).

```python
with tracer.span("checkout"):
    with tracer.span("validate_cart"):
        ...
# Completed spans are passed to writers that support `write_span` (e.g. JSONL, Jaeger).
```

Async equivalent:

```python
async with tracer.async_span("fetch_user"):
    ...
```

### Distributed tracing (W3C traceparent)

Module `glimpse.propagation` (re-exported from `glimpse`) implements:

- **`inject(headers)`** — Adds `traceparent` from the active span (mutates `headers` in place).
- **`extract(headers)`** — Returns `{"trace_id", "parent_span_id"}` or `None`.

On **`Tracer`**, use `tracer.inject(...)` and `tracer.extract(...)` for the same behavior.

Continue a remote trace in a manual span:

```python
ctx = tracer.extract(incoming_headers)  # dict-like, e.g. request headers
with tracer.span("handle_request", context=ctx):
    ...
```

### Starlette / FastAPI middleware (optional)

With `glimpse[starlette]`, wrap the app so each request runs inside `async_span` and **extracts** `traceparent` automatically:

```python
from glimpse.middleware import GlimpseMiddleware

app.add_middleware(GlimpseMiddleware, tracer=tracer)
```

Handlers can nest `tracer.span(...)` without copying headers by hand.

### Storage backends

| Destination | Role |
|-------------|------|
| `json` / `jsonl` | Append JSON lines; log lines and spans include a `record_type` field (`log_entry` vs `span`). |
| `sqlite` | Function-call log entries (schema oriented to `LogEntry`). |
| `jaeger` | OTLP HTTP JSON to a collector (default `http://localhost:4318/v1/traces`); requires `glimpse[jaeger]`. |

Multiple destinations:

```python
config = Config(dest=["jsonl", "jaeger"], params={
    "log_path": "out.jsonl",
    "jaeger_endpoint": "http://localhost:4318/v1/traces",
})
```

---

## Configuration

`Config` accepts destinations, level, trace IDs, path params, truncation, and env overrides. On import, it attempts to load **`.env`** / **`.env.<env>`** (see `config.py`) so variables like `GLIMPSE_DEST` can live in env files.

Common constructor options:

```python
config = Config(
    dest=["jsonl", "sqlite"],
    level="INFO",
    enable_trace_id=True,
    params={"log_path": "traces.jsonl", "db_path": "traces.db"},
    max_field_length=512,
)
```

### Environment variables

| Variable | Description |
|----------|-------------|
| `GLIMPSE_DEST` | Comma-separated list: `json`, `jsonl`, `sqlite`, `jaeger` (others in config may be reserved for future use). |
| `GLIMPSE_LEVEL` | e.g. `INFO`, `DEBUG`. |
| `GLIMPSE_TRACE_ID` | Truthy values enable trace correlation on log entries when applicable. |
| `GLIMPSE_*` | Non-core suffixes are folded into `params` (lower snake case), e.g. `GLIMPSE_LOG_PATH`, `GLIMPSE_JAEGER_ENDPOINT`. |

---

## Policy files

Policies still use `TracingPolicy` and optional `glimpse-policy.json` discovery. Example:

```json
{
  "version": "1.0",
  "name": "my_policy",
  "exact_modules": ["mylib", "requests"],
  "package_trees": ["myapp", "services"],
  "trace_depth": 5
}
```

Use `exact_modules` for exact module names only; `package_trees` for a package and its submodules. Wildcards are supported as documented in code/tests.

---

## Output examples

### JSONL log lines (function tracing)

Structured one JSON object per line with `stage` `START` / `END` / `EXCEPTION`, timing, etc.

### JSONL spans

Span records include `record_type: "span"` plus span fields (`trace_id`, `span_id`, `parent_span_id`, times, `status`, `events`, …).

### SQLite

SQL-friendly tables for **log entries** — useful for ad hoc queries on function-level traces (see writer schema in `writers/sqlite.py`).

---

## Architecture (conceptual)

```
Tracer ──► TracingPolicy ──► LogWriter ──► JSONWriter / SQLiteWriter / JaegerWriter
   │                            │
   │                            └── write_span(Span) where supported
   ├── span() / async_span()  ──► active span (contextvars)
   └── inject / extract       ──► W3C traceparent
```

- **Tracer** — `sys.settrace` automation, decorators, span managers, propagation helpers.
- **Span** — Dataclass model; completed spans go to writers that implement `write_span`.
- **LogWriter** — Multiplexes `write` and `write_span` to configured backends.
