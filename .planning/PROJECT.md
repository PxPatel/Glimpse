# Glimpse

## What This Is

Glimpse is a hobby Python library for low-instrumentation automatic function tracing. It captures function calls, arguments, return values, timings, and exceptions with zero code modification using `sys.settrace()`. Inspired by OpenTelemetry, AWS X-Ray, and Datadog, the goal is to evolve it into something more interesting — a lightweight distributed tracing system with spans and traces — as a learning project built by a developer exploring the observability space.

## Core Value

Give developers an easy, low-friction way to see what's happening inside their Python applications — from function-level tracing to basic distributed trace graphs — without the ceremony of a full APM setup.

## Requirements

### Validated

- ✓ Automatic function tracing via `sys.settrace()` — existing
- ✓ Manual tracing via `@trace_function` decorator — existing
- ✓ Trie-based policy filtering (package trees + exact modules) — existing
- ✓ Multi-destination writer fan-out (JSON/JSONL + SQLite backends) — existing
- ✓ Thread-safe ID generation (`{pid}-{instance}-{counter}`) — existing
- ✓ SQLite utility methods (slow function queries, error summaries, cleanup) — existing
- ✓ Environment-aware config with `GLIMPSE_*` env var prefix — existing
- ✓ Policy auto-discovery (walks directory tree for `glimpse-policy.json`) — existing
- ✓ SQLite backend selectable via `dest="sqlite"` (typo fixed) — Validated in Phase 01: bug-fixes
- ✓ SQLite schema creates without errors (trailing comma fixed) — Validated in Phase 01: bug-fixes
- ✓ Writer resource cleanup on `stop()` (close() added) — Validated in Phase 01: bug-fixes
- ✓ Exception type/traceback preserved through traced functions (bare raise) — Validated in Phase 01: bug-fixes
- ✓ Writer failures isolated from user code (try/except in write/flush) — Validated in Phase 01: bug-fixes
- ✓ Span data model (`Span`, `SpanEvent` dataclasses, parent/child via `parent_id`, `trace_id`) — Validated in Phase 02: span-model
- ✓ `tracer.span()` context manager with `ContextVar`-based active span tracking — Validated in Phase 02: span-model
- ✓ `tracer.async_span()` async context manager for asyncio function tracing — Validated in Phase 03: async-support
- ✓ `trace_async_function` decorator for async functions — Validated in Phase 03: async-support
- ✓ `JaegerWriter` OTLP HTTP exporter (`glimpse[jaeger]` optional extra) — Validated in Phase 04: jaeger-export
- ✓ W3C `traceparent` header inject/extract (`propagation.py`, ID lengths extended to W3C spec) — Validated in Phase 05: http-trace-propagation
- ✓ `context=` param on `span()`/`async_span()` for cross-process trace continuation — Validated in Phase 06: context-aware-span-api
- ✓ `GlimpseMiddleware` ASGI middleware for automatic FastAPI/Starlette span injection — Validated in Phase 06: http-middleware

### Active

None — all v1.0 milestone phases complete.

### Out of Scope

- Full OpenTelemetry SDK compatibility — this is educational, not a drop-in replacement
- Production-grade reliability guarantees — hobby project
- Language support beyond Python
- APM dashboard or UI — export to files/stdout, maybe Jaeger later
- Automatic HTTP framework middleware (Django, FastAPI) — ~~too much scope for v1~~ implemented as `GlimpseMiddleware` ASGI class (Phase 06)

## Context

This is a hobby project by a college student / new grad exploring the observability and distributed tracing space. The codebase already has a working synchronous tracing engine. The goal is to learn by building — understanding spans, trace propagation, and exporters by implementing them, not by wrapping an existing SDK.

**Current State:** All 6 phases of the v1.0 milestone complete. Full span/trace model implemented with async support, Jaeger OTLP export, W3C traceparent propagation, and ASGI middleware for FastAPI/Starlette. Next milestone not yet planned.

**Security notes (educational awareness):**
- Arguments captured verbatim — could log sensitive values; field truncation is only partial mitigation
- `cleanup_old_traces()` uses raw `.format(days)` in SQL — not injectable from current code path but worth fixing
- Policy files loaded without schema validation

## Constraints

- **Python version**: 3.9+ minimum (no f-string walrus operators etc.)
- **Scope**: Keep it fun and educational — don't recreate OpenTelemetry, just understand the concepts
- **Backwards compatible**: Existing `Tracer` API shouldn't break after bug fixes
- **Solo dev**: One person, hobby pace — don't over-engineer

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Fix bugs before new features | Leaks and broken backends undermine ability to test new distributed tracing work | Done — all bugs fixed in Phase 01 |
| Build span model from scratch | Learning opportunity — understand what OpenTelemetry solved | Done — `Span`/`SpanEvent` dataclasses with full parent/child/trace hierarchy |
| W3C TraceContext headers for propagation | Standard format, no extra deps, upgradeable later | Done — `propagation.py` with inject/extract; IDs extended to W3C spec lengths |
| Separate `async_span()` instead of overloading `span()` | Keeps sync/async explicit and simple | Done — Phase 03 |
| Lazy `requests` import in `JaegerWriter` | Module-level import must not fail if `requests` absent | Done — Phase 04 |
| `context=` param on span managers | Allows cross-process trace continuation without global mutation | Done — Phase 06 |
| `GlimpseMiddleware` as ASGI class | Compatible with FastAPI/Starlette; `starlette` optional dep | Done — Phase 06 |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-29 after Phase 06 completion (v1.0 milestone complete)*
