# Roadmap: Glimpse

## Overview

Glimpse evolves from a working synchronous function tracer into a lightweight distributed tracing library. Phase 1 fixes the broken plumbing so tests are trustworthy. Phase 2 introduces the Span model — the core distributed tracing concept. Phase 3 extends tracing to async code so modern Python applications are fully covered. Phase 4 adds Jaeger export so traces are visually explorable, closing the learning loop.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Bug Fixes** - Stabilize the existing tracer so all backends work and failures are safe
- [x] **Phase 2: Span Model** - Introduce the Span dataclass, active context tracking, and span output (completed 2026-03-29)
- [ ] **Phase 3: Async Support** - Extend tracing to async functions and await boundaries
- [ ] **Phase 4: Jaeger Export** - Export spans to a local Jaeger instance for visual trace exploration

## Phase Details

### Phase 1: Bug Fixes
**Goal**: The existing tracer works correctly — SQLite initializes, resources are released on stop, exceptions preserve their original tracebacks, and writer failures never crash user code
**Depends on**: Nothing (first phase)
**Requirements**: BUG-01, BUG-02, BUG-03, BUG-04, BUG-05
**Success Criteria** (what must be TRUE):
  1. A tracer configured with `"sqlite"` backend writes records to disk without initialization errors
  2. Calling `tracer.stop()` closes all file handles and DB connections (no resource leaks detectable via OS tooling)
  3. An exception raised inside a traced function preserves its original type and traceback when it propagates to the caller
  4. A writer that raises during output logs the error to stderr and allows the traced function to complete normally
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md — Fix sqlite typo (BUG-01), trailing comma in schema (BUG-02), and writer safety (BUG-05) in logwriter.py and sqlite.py
- [x] 01-02-PLAN.md — Fix resource leak on stop() (BUG-03) and exception re-raise destroying traceback (BUG-04) in tracer.py

### Phase 2: Span Model
**Depends on**: Phase 1
**Requirements**: SPAN-01, SPAN-02, SPAN-03, SPAN-04, SPAN-05, SPAN-06
**Success Criteria** (what must be TRUE):
  1. A `Span` object exists with `trace_id`, `span_id`, `parent_span_id`, `name`, `start_time`, `end_time`, `attributes`, `status`, and `events` fields
  2. `with tracer.span("operation") as span:` creates a span, sets it as the active context, and closes it on exit
  3. A child `with tracer.span("child"):` block created inside a parent span automatically links to the parent via `parent_span_id`
  4. Auto-instrumented functions (via `sys.settrace`) record the active span's `span_id` as `parent_span_id` in call metadata (full Span emission deferred to Phase 3)
  5. When an exception propagates through a span block, the span's `status` is set to `"error"` and the exception is recorded
  6. The JSON writer outputs completed spans in a structured, human-readable format distinct from the legacy LogEntry format
**Plans**: 3 plans

Plans:
- [ ] 02-01-PLAN.md — Span dataclass + ContextVar active span tracking (SPAN-01, SPAN-02)
- [ ] 02-02-PLAN.md — tracer.span() context manager + auto-instrumentation parent linking + error recording (SPAN-03, SPAN-04, SPAN-05)
- [ ] 02-03-PLAN.md — JSON writer span output with record_type tagging (SPAN-06)

Wave structure:
- Wave 1: 02-01 (Span dataclass + context — no dependencies)
- Wave 2: 02-02 (context manager + wiring — depends on 02-01)
- Wave 3: 02-03 (JSONWriter write_span override — depends on 02-01 and 02-02)

### Phase 3: Async Support
**Goal**: Async functions can be traced with the same span parent/child semantics as synchronous functions, without breaking existing sync tracing
**Depends on**: Phase 2
**Requirements**: ASYNC-01, ASYNC-02, ASYNC-03, ASYNC-04
**Success Criteria** (what must be TRUE):
  1. Decorating an async function with `@tracer.trace_async_function` produces a span for each call, visible in JSON output
  2. An `async with tracer.async_span("name"):` block creates and closes a span correctly in an async context
  3. Parent span context established before an `await` is still the active parent when execution resumes after the `await`
  4. Existing synchronous tracing tests pass unchanged after async support is added
**Plans**: 2 plans

Plans:
- [ ] 03-01-PLAN.md — `_AsyncSpanContext` class + `async_span()` method + `trace_async_function` decorator in tracer.py (ASYNC-01, ASYNC-02, ASYNC-03)
- [ ] 03-02-PLAN.md — Async tracing test suite + pytest-asyncio setup + sync regression guard (ASYNC-01, ASYNC-02, ASYNC-03, ASYNC-04)

Wave structure:
- Wave 1: 03-01 (implementation — no dependencies within phase)
- Wave 2: 03-02 (tests — depends on 03-01)

### Phase 4: Jaeger Export
**Goal**: Completed spans can be sent to a local Jaeger instance and appear as visual trace graphs in the Jaeger UI
**Depends on**: Phase 3
**Requirements**: OBS-01, OBS-02, OBS-03
**Success Criteria** (what must be TRUE):
  1. A trace created with `with tracer.span("root"):` appears in the Jaeger UI at `http://localhost:16686` after the program runs
  2. Installing `glimpse[jaeger]` is sufficient to enable Jaeger export; no manual dependency installation needed
  3. When the Jaeger endpoint is unreachable, the error is logged to stderr and the traced program continues without raising an exception
**Plans**: 2 plans

Plans:
- [ ] 04-01-PLAN.md — JaegerWriter class with OTLP HTTP export and stderr error handling (OBS-01, OBS-03)
- [ ] 04-02-PLAN.md — pyproject.toml jaeger extra, LogWriter registration, write_span fan-out, integration tests (OBS-02, OBS-03)

Wave structure:
- Wave 1: 04-01 (JaegerWriter — no dependencies)
- Wave 2: 04-02 (registration + packaging — depends on 04-01)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Bug Fixes | 0/2 | Not started | - |
| 2. Span Model | 3/3 | Complete   | 2026-03-29 |
| 3. Async Support | 1/2 | In Progress|  |
| 4. Jaeger Export | 0/2 | Not started | - |
