---
phase: 05-http-trace-propagation
plan: 01
subsystem: propagation
tags: [w3c-traceparent, inject, extract, http-propagation]
dependency_graph:
  requires: [context.py, span.py, ids.py]
  provides: [propagation.py, tracer.inject, tracer.extract]
  affects: [tracer.py, __init__.py]
tech_stack:
  added: []
  patterns: [W3C traceparent header format, pure-function propagation module]
key_files:
  created:
    - src/glimpse/propagation.py
  modified:
    - src/glimpse/tracer.py
    - src/glimpse/__init__.py
    - src/glimpse/common/ids.py
decisions:
  - "propagation.py is dependency-free from Tracer — imports only context.py, keeping the module graph clean"
  - "IDGenerator ID lengths extended to W3C spec (32-hex trace_id, 16-hex span_id) — required for valid traceparent headers; no existing tests depended on old length"
metrics:
  duration: "~10m"
  completed: "2026-03-29"
  tasks: 2
  files: 4
---

# Phase 5 Plan 01: inject() and extract() Implementation Summary

W3C traceparent inject/extract with pure propagation module and W3C-compliant ID generation.

## What Was Built

- `src/glimpse/propagation.py` — pure `inject(headers)` and `extract(headers)` functions; inject writes `00-{trace_id}-{span_id}-01` to the headers dict using the active span from ContextVar; extract parses and validates with regex, returns `{"trace_id", "parent_span_id"}` or None
- `Tracer.inject()` and `Tracer.extract()` — thin delegation wrappers added after `_on_span_end`
- `glimpse.__init__` — `inject` and `extract` added to public API and `__all__`
- `IDGenerator` — `new_trace_id()` extended to full 32-hex UUID hex, `new_call_id()` extended to 16-hex, both now W3C spec-compliant

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1 | Create propagation.py | eeecefd |
| 2 | Add Tracer methods + __init__ exports + fix ID lengths | 13b889e |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] IDGenerator produced 12-hex IDs, incompatible with W3C traceparent**
- **Found during:** Task 2 smoke test
- **Issue:** `new_trace_id()` returned 12-hex, `new_call_id()` returned 12-hex; W3C requires 32 and 16 respectively; inject() would produce invalid traceparent headers
- **Fix:** Extended `new_trace_id()` to `uuid.uuid4().hex` (32 hex) and `new_call_id()` to `uuid.uuid4().hex[:16]`
- **Files modified:** `src/glimpse/common/ids.py`
- **Commit:** 13b889e
- **Test impact:** All 247 existing tests still pass; no test enforced old ID length

## Validation Results

- `python -c "from glimpse.propagation import inject, extract; print('ok')"` — passes
- `python -c "from glimpse import inject, extract; print('ok')"` — passes
- Smoke test: traceparent matches `^00-[a-f0-9]{32}-[a-f0-9]{16}-01$` — passes
- Extract round-trip matches original span.trace_id and span.span_id — passes
- No-op when no active span — passes
- Returns None on missing/malformed traceparent — passes
- Full test suite: 247 passed

## Self-Check: PASSED

- `src/glimpse/propagation.py` — FOUND
- `src/glimpse/tracer.py` contains `def inject` — FOUND
- Commit eeecefd — FOUND
- Commit 13b889e — FOUND
