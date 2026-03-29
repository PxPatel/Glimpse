---
phase: "02-span-model"
plan: "02-03"
subsystem: "writers"
tags: ["json", "span", "serialization", "output"]
dependency_graph:
  requires: ["02-01", "02-02"]
  provides: ["span-json-output"]
  affects: ["JSONWriter", "BaseWriter"]
tech_stack:
  added: []
  patterns: ["dataclasses.asdict for recursive serialization", "record_type tag for format discrimination"]
key_files:
  created:
    - tests/glimpse/test_json_writer_spans.py
  modified:
    - src/glimpse/writers/json.py
decisions:
  - "Use dataclasses.asdict instead of __dict__ for recursive nested dataclass serialization (SpanEvent inside Span.events)"
  - "Local import of Span inside write() to avoid circular import at module level"
  - "Legacy LogEntry records now also tagged with record_type=log_entry — additive, non-breaking"
metrics:
  duration: "~10 minutes"
  completed_date: "2026-03-29"
  tasks_completed: 3
  files_changed: 2
---

# Phase 02 Plan 03: JSON Writer Span Output Summary

**One-liner:** JSONWriter extended to serialize Span dataclasses with `record_type: "span"` tag using `dataclasses.asdict` for recursive nested field conversion.

## What Was Built

Extended `JSONWriter.write()` to use `dataclasses.asdict` for all dataclass entries, tagging spans as `"record_type": "span"` and legacy `LogEntry` records as `"record_type": "log_entry"`. Added `write_span()` override that routes through `write()`. This completes the full span output path: `_SpanContext.__exit__` → `_on_span_end` → `write_span` → `write` → JSONL file.

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1 | Replace JSONWriter.write() with dataclasses.asdict-based serialization + record_type tagging | ba136b9 |
| 2 | Add write_span() override to JSONWriter | ba136b9 |
| 3 | Create tests/glimpse/test_json_writer_spans.py with 6 tests | ba136b9 |

## Decisions Made

- `dataclasses.asdict` used over `__dict__` because it recursively converts nested dataclasses — critical for `SpanEvent` objects inside `Span.events`
- `from ..span import Span` placed as a local import inside `write()` to avoid circular import at module level
- `default=str` added to `json.dumps` to gracefully handle datetime and other non-JSON-serializable types
- `LogEntry` records now tagged with `record_type=log_entry` — additive change, does not break existing consumers

## Verification

- `pytest tests/glimpse/test_json_writer_spans.py -x` — 6 passed
- `pytest tests/glimpse/ -x` — 224 passed

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- `src/glimpse/writers/json.py` — FOUND
- `tests/glimpse/test_json_writer_spans.py` — FOUND
- commit ba136b9 — FOUND
