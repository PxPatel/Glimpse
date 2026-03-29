---
phase: "02"
plan: "02"
subsystem: span-model
tags: [spans, context-manager, auto-instrumentation, tracing]
dependency_graph:
  requires: [02-01]
  provides: [tracer.span context manager, parent span linking, _on_span_end routing]
  affects: [tracer.py, writers/base.py]
tech_stack:
  added: []
  patterns: [ContextVar token reset in __exit__, _SpanContext class-based context manager]
key_files:
  created: [tests/glimpse/test_span_context.py]
  modified: [src/glimpse/tracer.py, src/glimpse/writers/base.py]
decisions:
  - Class-based _SpanContext instead of @contextmanager generator for reliable token reset in __exit__
  - reset_active_span before _on_span_end to prevent leaked context if writer raises
  - write_span no-op on BaseWriter preserves backwards compatibility with all existing writers
metrics:
  duration: "~10 min"
  completed: "2026-03-29"
  tasks_completed: 3
  files_changed: 3
---

# Phase 02 Plan 02: Context Manager and Auto-Instrumentation Integration Summary

**One-liner:** Class-based `tracer.span()` context manager with ContextVar nesting, error recording, and auto-instrumentation parent span capture.

## What Was Built

Added `tracer.span("name")` as a context manager that:
- Creates a `Span` with correct `trace_id`, `span_id`, `start_time`, and `parent_span_id` from the active span
- Activates itself as the active span via `set_active_span()` and records the token
- On exit: sets `end_time`, records error event if an exception propagated, resets active span context, then calls `_on_span_end()`

Also wired auto-instrumentation (`_handle_function_call`) to capture `parent_span_id` and `trace_id` from the active span into `call_info` metadata, establishing the parent link for future Span emission.

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1 | Add `_SpanContext`, `tracer.span()`, `_on_span_end()`, `BaseWriter.write_span()` | 51451a8 |
| 2 | Store `parent_span_id`/`trace_id` in `_handle_function_call` `call_info` | 51451a8 |
| 3 | Create `tests/glimpse/test_span_context.py` (8 tests) | cbe3634 |

## Verification

- `pytest tests/glimpse/test_span_context.py -x` — 8 passed
- `pytest tests/glimpse/ -x` — 218 passed (full suite green)

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

- `src/glimpse/tracer.py` — exists with `_SpanContext` and `span()` method
- `src/glimpse/writers/base.py` — exists with `write_span()` no-op
- `tests/glimpse/test_span_context.py` — exists with 8 tests
- Commits 51451a8 and cbe3634 — verified in git log
