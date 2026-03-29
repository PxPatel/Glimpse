---
phase: "02"
plan: "01"
subsystem: span-model
tags: [span, dataclass, context, contextvars]
dependency_graph:
  requires: []
  provides: [Span, SpanEvent, get_active_span, set_active_span, reset_active_span]
  affects: [glimpse.__init__]
tech_stack:
  added: [contextvars.ContextVar]
  patterns: [dataclass-pure-data-container, token-based-context-reset]
key_files:
  created:
    - src/glimpse/span.py
    - src/glimpse/context.py
    - tests/glimpse/test_span.py
  modified:
    - src/glimpse/__init__.py
decisions:
  - "Span and SpanEvent are pure data containers — no methods added yet"
  - "ContextVar token semantics used for nesting: set returns token, reset restores previous value"
  - "Forward reference string 'Span' used in context.py to avoid circular import"
metrics:
  duration: "~5 minutes"
  completed: "2026-03-29"
  tasks_completed: 3
  files_changed: 4
---

# Phase 2 Plan 1: Span Dataclass and Active Context Tracking Summary

**One-liner:** Span and SpanEvent dataclasses with ContextVar-based active span tracking supporting token-based nesting.

## What Was Built

- `src/glimpse/span.py` — `Span` dataclass (trace_id, span_id, name, start_time, parent_span_id, end_time, attributes, status, events) and `SpanEvent` dataclass (name, timestamp, attributes). Pure data containers, no methods.
- `src/glimpse/context.py` — Module-level `ContextVar` with `get_active_span`, `set_active_span`, `reset_active_span` helpers. Token-based reset correctly restores outer spans in nested contexts.
- `src/glimpse/__init__.py` — Exports `Span`, `SpanEvent`, `get_active_span`, `set_active_span`, `reset_active_span` at the package level.
- `tests/glimpse/test_span.py` — 10 tests covering Span defaults, SpanEvent instantiation, and full context tracking lifecycle including nested context restore.

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Create Span and SpanEvent dataclasses | 55d2e22 | src/glimpse/span.py |
| 2 | Add ContextVar active span tracking | 0bd09a0 | src/glimpse/context.py, __init__.py |
| 3 | Write tests for span and context | c3a6f48 | tests/glimpse/test_span.py |

## Verification

- `pytest tests/glimpse/test_span.py -x` — 10 passed
- `from glimpse import Span, SpanEvent` — works
- `from glimpse import get_active_span, set_active_span, reset_active_span` — works
- Nested span context test confirms token-based reset restores outer span

## Deviations from Plan

None - plan executed exactly as written.

Tasks 1 and 2 were already partially drafted (files existed as untracked) before execution started. Verified contents matched the plan spec exactly and committed them as separate task commits.

## Self-Check: PASSED

- src/glimpse/span.py: FOUND
- src/glimpse/context.py: FOUND
- tests/glimpse/test_span.py: FOUND
- Commit 55d2e22: FOUND
- Commit 0bd09a0: FOUND
- Commit c3a6f48: FOUND
