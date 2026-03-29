---
phase: 06-context-aware-span-api-and-http-middleware
plan: 02
subsystem: middleware
tags: [middleware, asgi, starlette, fastapi, traceparent]
dependency_graph:
  requires: [06-01]
  provides: [GlimpseMiddleware]
  affects: [src/glimpse/middleware.py, pyproject.toml]
tech_stack:
  added: [starlette optional dependency]
  patterns: [BaseHTTPMiddleware subclass, ImportError guard for optional deps]
key_files:
  created: [src/glimpse/middleware.py]
  modified: [pyproject.toml]
decisions:
  - starlette is an optional dependency with ImportError guard at module level
  - async_span used in dispatch() since BaseHTTPMiddleware.dispatch is async
metrics:
  duration: "5m"
  completed: "2026-03-29"
  tasks: 2
  files: 2
---

# Phase 6 Plan 02: GlimpseMiddleware for FastAPI/Starlette + pyproject.toml Extra Summary

**One-liner:** ASGI middleware with W3C traceparent extraction and root span per request using starlette as optional dependency.

## What Was Built

Created `src/glimpse/middleware.py` with `GlimpseMiddleware`, an ASGI middleware class that:
- Extracts `traceparent` from incoming request headers via `tracer.extract()`
- Opens a root span (`"METHOD /path"`) for the full request duration via `tracer.async_span()`
- Raises `ImportError` with install hint when starlette is not installed
- Added `starlette = ["starlette"]` to `[project.optional-dependencies]` in pyproject.toml

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1 | Create src/glimpse/middleware.py | a512a1b |
| 2 | Add starlette optional dep to pyproject.toml | 64905f6 |

## Deviations from Plan

None - plan executed exactly as written.

## Verification

- Import succeeds when starlette installed: confirmed
- ImportError with install hint when starlette missing: confirmed
- 257 tests pass, 0 failures

## Self-Check: PASSED
