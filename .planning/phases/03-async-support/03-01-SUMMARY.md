---
phase: 03-async-support
plan: "01"
subsystem: tracer
tags: [async, spans, context-manager, decorator, contextvars]
dependency_graph:
  requires: [02-02, 02-03]
  provides: [async span context manager, async function decorator]
  affects: [tracer.py]
tech_stack:
  added: []
  patterns: [ContextVar token pattern for async, async context manager protocol]
key_files:
  created: []
  modified:
    - src/glimpse/tracer.py
decisions:
  - Separate async_span() method (not overloading span()) keeps sync/async explicit and simple
  - ContextVar token reset in finally block of trace_async_function ensures context is always restored
metrics:
  duration: ~10 minutes
  completed: 2026-03-29
---

# Phase 3 Plan 1: Async Decorator and Async Context Manager Summary

**One-liner:** Async span support via `_AsyncSpanContext` class and `trace_async_function` decorator using ContextVar token pattern for correct parent/child linking across await boundaries.

## What Was Built

Added async tracing support to `tracer.py` with two mechanisms:

1. `_AsyncSpanContext` — mirrors `_SpanContext` exactly but uses `__aenter__`/`__aexit__`. Returns a `Span` on entry, sets it as active via ContextVar, records end time and error events on exit, then resets the ContextVar token to restore prior context.

2. `tracer.async_span(name)` — factory method returning `_AsyncSpanContext` for use with `async with`.

3. `tracer.trace_async_function(func)` — decorator wrapping async functions. Creates a span, sets it active, awaits the wrapped function, captures exceptions as span events, and always resets context in `finally`.

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1 | Add _AsyncSpanContext class and async_span() method | 895c2cf |
| 2 | Add trace_async_function decorator | 895c2cf |

## Verification

- `pytest tests/glimpse/test_span_context.py` — 8 passed (sync tests unaffected)
- Manual smoke test: trace_async_function, async_span, and parent/child span linking all work correctly

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- `src/glimpse/tracer.py` modified: FOUND
- Commit 895c2cf: FOUND
