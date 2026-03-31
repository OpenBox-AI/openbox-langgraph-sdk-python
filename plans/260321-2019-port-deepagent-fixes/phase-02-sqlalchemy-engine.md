# Phase 2: Add sqlalchemy_engine Parameter

**Priority:** high
**Status:** planned
**Effort:** small

## Overview

Expose `sqlalchemy_engine` kwarg in `create_openbox_graph_handler()` and thread it to `setup_opentelemetry_for_governance()`. Fixes DB governance not firing when engine created before handler.

## Related Code Files

- **Modify:** `openbox_langgraph/langgraph_handler.py`

## Changes

### 1. `OpenBoxLangGraphHandlerOptions` — add field
```python
sqlalchemy_engine: Any = None
```

### 2. `OpenBoxLangGraphHandler.__init__` — pass to OTel setup (line ~401)
```python
setup_opentelemetry_for_governance(
    ...,
    sqlalchemy_engine=opts.sqlalchemy_engine,
)
```

### 3. `create_openbox_graph_handler` — add explicit param
```python
def create_openbox_graph_handler(
    graph,
    *,
    ...,
    sqlalchemy_engine: Any = None,
    **handler_kwargs,
) -> OpenBoxLangGraphHandler:
```
Pass `sqlalchemy_engine` to `OpenBoxLangGraphHandlerOptions`.

## Success Criteria

- `create_openbox_graph_handler(graph, ..., sqlalchemy_engine=engine)` works
- Startup logs show `Instrumented: sqlalchemy (existing engine)` when engine passed
- All existing tests pass
