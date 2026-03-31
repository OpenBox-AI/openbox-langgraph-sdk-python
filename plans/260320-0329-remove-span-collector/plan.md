---
title: "Remove SpanCollector — SpanProcessor as sole activity context"
status: pending
priority: P1
branch: feat/otel-http-hook-spans
created: 2026-03-20
---

# Remove SpanCollector — SpanProcessor Only

## Context

User's architecture diagram shows each process (tool call) has:
- **SpanProcessor** — stores activity context, captures spans, sends to Core
- No SpanCollector

Current code has both SpanCollector (per-tool ContextVar registry) and SpanProcessor (OTel trace_id mapping). User removed SpanCollector conceptually. This plan removes it from code.

## Architecture (After)

```
on_tool_start
  └─ SpanProcessor.set_activity_context() + register_trace()

Tool executes httpx call
  └─ OTel hook → hook_governance._build_payload()
       └─ SpanProcessor.get_activity_context_by_trace() (with single-activity fallback)

on_tool_end
  └─ SpanProcessor.clear_activity_context()
```

No SpanCollector, no ContextVar registry, no pending hook coros, no abort registry.

## Phases

| # | Phase | File | Status |
|---|-------|------|--------|
| 1 | Remove SpanCollector imports from handler | `langgraph_handler.py` | pending |
| 2 | Remove SpanCollector from hook_governance | `hook_governance.py` | pending |
| 3 | Delete span_collector.py | `span_collector.py` | pending |
| 4 | Clean up __init__.py exports | `__init__.py` | pending |
| 5 | Verify compilation | all | pending |

## Phase 1: Remove SpanCollector from handler (`langgraph_handler.py`)

### Remove imports (line 47-56)
```python
# DELETE these:
from openbox_langgraph.span_collector import (
    SpanCollector,
    SpanHookContext,
    activate_span_collector,
    mark_activity_started,
    get_active_tool_hook_context,
    drain_spans,
    flush_pending_spans,
    enqueue_internal_span_hook,
)
```

### Remove from _RunBuffer (line 231)
```python
# DELETE:
span_collector: SpanCollector | None = None
```

### on_tool_start — remove SpanCollector activation
Keep only SpanProcessor registration. Remove:
- `activate_span_collector(event_run_id, hook_context=hook_ctx)`
- `buf.span_collector = collector`
- `SpanHookContext(...)` construction (only used by SpanCollector)

### _process_event — remove SpanCollector references
- Remove `mark_activity_started(gov_event.activity_id)` (line ~919)
- Remove `enqueue_internal_span_hook(...)` calls (started + completed spans)
- Remove `flush_pending_spans(event.run_id)` (line ~908)
- Remove `get_active_tool_hook_context()` usage in LLMCompleted handler (line ~825)
- Remove `SpanHookContext(...)` constructions for LLM span routing

### on_tool_end — remove SpanCollector cleanup
- Remove `drain_spans(buf.span_collector)` (line ~1286-1287)
- Remove internal span hook "completed" (`enqueue_internal_span_hook`)
- Keep SpanProcessor cleanup: `clear_activity_context()`

## Phase 2: Remove SpanCollector from hook_governance (`hook_governance.py`)

Remove the SpanCollector ContextVar lookup added earlier:
```python
# DELETE from _build_payload():
try:
    from openbox_langgraph.span_collector import get_active_tool_hook_context
    hook_ctx = get_active_tool_hook_context()
    ...
except ImportError:
    pass
```

SpanProcessor trace_id lookup becomes the ONLY path again (with single-activity fallback).

## Phase 3: Delete span_collector.py

```bash
rm openbox_langgraph/span_collector.py
```

## Phase 4: Clean up __init__.py

Remove any re-exports of SpanCollector, SpanHookContext, etc.

## Phase 5: Verify

```bash
python3 -c "import ast; ast.parse(open('openbox_langgraph/langgraph_handler.py').read())"
python3 -c "import ast; ast.parse(open('openbox_langgraph/hook_governance.py').read())"
```

## Risk

- Internal span hooks (started/completed) will be gone — Core won't see `function_call` hook_type spans. Only HTTP/DB/file OTel-based spans remain.
- `mark_activity_started` signal is gone — httpx hooks won't wait for ActivityStarted row in Core before sending span governance. Could cause race condition where span arrives before its parent row.
