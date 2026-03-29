---
phase: 06-context-aware-span-api-and-http-middleware
plan: "03"
subsystem: testing
tags: [tests, context-propagation, middleware, starlette]
dependency_graph:
  requires: [06-01, 06-02]
  provides: [test coverage for context param and GlimpseMiddleware]
  affects: [src/glimpse/tracer.py]
tech_stack:
  added: [httpx (test dependency for Starlette TestClient)]
  patterns: [MagicMock _on_span_end for span capture, Starlette TestClient for ASGI integration]
key_files:
  created:
    - tests/glimpse/test_context_param.py
    - tests/glimpse/test_middleware.py
  modified:
    - src/glimpse/tracer.py
decisions:
  - "Use Config(dest='json') not Config(output='json') — plan had wrong param name; fixed to match actual Config API"
  - "Auto-fixed trace_id inheritance bug in _SpanContext/_AsyncSpanContext — child spans now inherit trace_id from parent span rather than IDGenerator"
metrics:
  duration: "~15m"
  completed: "2026-03-29"
  tasks: 2
  files: 3
---

# Phase 6 Plan 3: Tests — Context Parameter and Middleware Summary

Context param and middleware integration tests written and passing; auto-fixed a trace_id inheritance bug where nested spans did not propagate the remote trace_id through the in-process span tree.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | test_context_param.py — unit tests for context param | 31f5e33 | tests/glimpse/test_context_param.py |
| 2 | test_middleware.py — integration tests for GlimpseMiddleware | 77ed181 | tests/glimpse/test_middleware.py |

## Verification

- `pytest tests/glimpse/test_context_param.py -v` — 8 passed
- `pytest tests/glimpse/test_middleware.py -v` — 7 passed
- `pytest tests/ -x -q` — 272 passed, 0 failed

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Config API mismatch — `output=` does not exist**
- **Found during:** Task 1 and 2 (test collection)
- **Issue:** Plan used `Config(output="json", filename="/dev/null")` but the real Config constructor uses `dest=` with no `filename` param
- **Fix:** Changed `make_tracer()` to use `Config(dest="json")` in both test files
- **Files modified:** tests/glimpse/test_context_param.py, tests/glimpse/test_middleware.py

**2. [Rule 1 - Bug] Child spans did not inherit trace_id from parent span**
- **Found during:** Task 1 (`test_nested_span_still_links_to_parent`) and Task 2 (`test_handler_child_span_inherits_trace_id`)
- **Issue:** `_SpanContext.__enter__` and `_AsyncSpanContext.__aenter__` used `IDGenerator.get_current_trace_id()` for child spans regardless of active parent span. When parent had a remote trace_id (from context=), child got a different auto-generated trace_id.
- **Fix:** When `parent is not None`, inherit `trace_id = parent.trace_id` before falling back to IDGenerator
- **Files modified:** src/glimpse/tracer.py
- **Commit:** dfce1f8

## Self-Check: PASSED

- tests/glimpse/test_context_param.py: EXISTS
- tests/glimpse/test_middleware.py: EXISTS
- Commits 31f5e33, 77ed181, dfce1f8: FOUND
