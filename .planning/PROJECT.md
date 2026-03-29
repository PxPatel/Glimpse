# Glimpse

## What This Is

Glimpse is a low-instrumentation Python library for automatic function-level tracing and logging. It captures function calls, arguments, return values, timings, and exceptions with zero code modification using `sys.settrace()`. The goal is to evolve it into a robust distributed tracing system — inspired by OpenTelemetry, AWS X-Ray, and Datadog — with full span/trace support for modern, distributed Python architectures.

## Core Value

Give developers deep visibility into Python application execution — from single-process function traces to distributed service call graphs — with minimal instrumentation effort.

## Requirements

### Validated

- ✓ Automatic function tracing via `sys.settrace()` — existing
- ✓ Manual tracing via `@trace_function` decorator — existing
- ✓ Trie-based policy filtering (package trees + exact modules) — existing
- ✓ Multi-destination fan-out writer (JSON/JSONL + SQLite backends) — existing
- ✓ Thread-safe ID generation (`{pid}-{instance}-{counter}`) — existing
- ✓ SQLite utility methods (slow functions, error summaries, cleanup) — existing
- ✓ Environment-aware config with `GLIMPSE_*` env var prefix — existing

### Active

- [ ] Fix SQLite backend typo (`"sqllite"` → `"sqlite"`) blocking initialization
- [ ] Fix SQLite schema trailing comma syntax error
- [ ] Add proper `close()` call in `Tracer.stop()` to prevent resource leaks
- [ ] Fix exception re-raise to preserve original traceback
- [ ] Add try/except around writer operations to prevent writer failures from crashing user code
- [ ] Distributed tracing: span model with parent-child relationships
- [ ] Distributed tracing: trace context propagation (W3C TraceContext / B3 headers)
- [ ] Async/await support (`asyncio` tracing)
- [ ] Exporters: OTLP (OpenTelemetry Protocol), Jaeger, Zipkin
- [ ] Sampling and rate limiting for high-frequency environments
- [ ] PII masking / field exclusion for sensitive argument capture

### Out of Scope

- Full OpenTelemetry SDK compatibility — Glimpse is inspired by OTEL, not a wrapper around it
- Language support beyond Python — single-language focus for now
- APM dashboard / UI — export to existing tools (Jaeger, Grafana Tempo, Datadog)
- Automatic framework instrumentation (Django, FastAPI middleware) — v2 consideration

## Context

The codebase currently exists with a functional synchronous tracing engine. Key known issues identified during audit:

**Blocking bugs:**
- `logwriter.py:24` — `"sqllite"` typo prevents SQLite backend from ever initializing
- `sqlite.py:55` — trailing comma in CREATE TABLE causes schema failure
- `tracer.py:stop()` — calls `flush()` but never `close()`, causing resource leaks
- Exception handler raises `Exception(string)` instead of re-raising, losing tracebacks

**Security concerns:**
- All function arguments captured verbatim (passwords, tokens, API keys land in logs)
- `cleanup_old_traces()` uses raw `.format()` in SQL — injectable if exposed as user input
- Policy/env files discovered by walking to filesystem root with no path limits

**Missing capabilities:**
- No async/await support (`sys.settrace` unreliable with asyncio)
- No distributed tracing (no span hierarchy, no context propagation)
- No write buffering — every entry is immediate I/O
- No PII masking

The intended evolution path mirrors OpenTelemetry's model: spans → traces → context propagation → exporters.

## Constraints

- **Compatibility**: Python 3.9+ minimum; no breaking changes to existing tracer API surface
- **Performance**: Tracing overhead must remain < 5% for typical applications
- **Zero-dependency core**: Core tracing engine should work without external dependencies; exporters can require extras
- **Backwards-compatible policy format**: `glimpse-policy.json` schema must remain compatible

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Fix bugs before new features | Broken SQLite backend and resource leaks undermine trust in data integrity | — Pending |
| W3C TraceContext for propagation | Industry standard, compatible with OTEL/Jaeger/Zipkin; avoids proprietary formats | — Pending |
| OTLP as primary export format | Broadest compatibility — works with Jaeger, Grafana Tempo, Datadog, Honeycomb | — Pending |
| Keep `sys.settrace` for sync, add `sys.monitoring` for async | Python 3.12+ offers `sys.monitoring` for lower-overhead tracing | — Pending |

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
*Last updated: 2026-03-29 after initialization*
