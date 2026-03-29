---
phase: 04-jaeger-export
plan: "01"
subsystem: writers
tags: [jaeger, otlp, http-export, span-serialization]
dependency_graph:
  requires: [src/glimpse/writers/base.py, src/glimpse/span.py]
  provides: [src/glimpse/writers/jaeger.py]
  affects: []
tech_stack:
  added: [requests (optional runtime dep)]
  patterns: [lazy import in __init__ for optional dependency]
key_files:
  created:
    - src/glimpse/writers/jaeger.py
    - tests/glimpse/test_jaeger_writer.py
  modified: []
key_decisions:
  - "requests imported lazily inside __init__ so module-level import never fails — ImportError surfaces only on instantiation"
  - "write() is a deliberate no-op; LogEntry-style records have no meaning in OTLP context"
  - "Non-2xx responses treated as failures and logged to stderr alongside connection errors"
metrics:
  duration: "~10m"
  completed_date: "2026-03-29"
  tasks_completed: 1
  files_created: 2
  files_modified: 0
---

# Phase 4 Plan 1: JaegerWriter — OTLP HTTP Span Export Summary

**One-liner:** JaegerWriter serializes Span dataclasses into OTLP JSON ResourceSpans and POSTs them to a Jaeger HTTP endpoint, with all failures caught and printed to stderr.

## What Was Built

`JaegerWriter(BaseWriter)` in `src/glimpse/writers/jaeger.py`:

- `__init__(endpoint)` — stores endpoint, imports `requests` lazily, raises `ImportError` with pip hint if absent.
- `_span_to_otlp(span)` — converts a `Span` to OTLP JSON (ResourceSpans wrapper), including nanosecond timestamps, attribute key-value pairs, status code mapping (ok=1, error=2), and nested SpanEvent serialization.
- `write_span(span)` — calls `_span_to_otlp`, POSTs with `Content-Type: application/json`, checks 2xx range, logs non-2xx and connection errors to `sys.stderr` without raising.
- `write(entry)` — intentional no-op.

15 unit tests in `tests/glimpse/test_jaeger_writer.py` cover: OTLP payload shape, field values, timestamp math, error status code mapping, successful POST, non-2xx stderr logging, connection error stderr logging, no-raise guarantee, and ImportError on missing `requests`.

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check

- [x] `src/glimpse/writers/jaeger.py` — created
- [x] `tests/glimpse/test_jaeger_writer.py` — created
- [x] Commit `d6a0484` — verified
- [x] 15/15 tests passing

## Self-Check: PASSED
