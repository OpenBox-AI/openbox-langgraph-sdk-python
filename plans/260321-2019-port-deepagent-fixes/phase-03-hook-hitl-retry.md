# Phase 3: Hook HITL Retry + Wrapped Error Unwrapping

**Priority:** high
**Status:** planned
**Effort:** medium

## Overview

When OTel hooks (httpx/DB/file) return REQUIRE_APPROVAL during graph execution, the error propagates up as `GovernanceBlockedError` (or wrapped by LLM SDKs as `APIConnectionError`). The handler should catch it, poll for approval, then retry.

## Architecture Challenge

The langgraph SDK uses `astream_events` — events are processed as they stream. There's no `awrap_tool_call` or `awrap_model_call` to wrap individual operations. Errors from hooks propagate through:

```
astream_events → LangGraph internal → tool/LLM execution → OTel hook → error
  → LangGraph catches internally → may re-raise or swallow
  → surfaces as exception in ainvoke()/astream_governed()
```

### Option A: Catch at ainvoke/astream_governed level
Catch the error after `astream_events` fails, poll, re-invoke the entire graph.

**Problem:** Re-invoking restarts from scratch. Without a checkpointer, all previous tool/LLM calls re-execute. With a checkpointer, it resumes from the last checkpoint — but the specific tool that triggered the hook re-runs.

### Option B: Add _extract_governance_blocked to hook_governance module
Move the unwrapping logic into `hook_governance.py` so it's shared across SDKs. But the retry still needs to happen at the handler level.

### Option C: Catch at _process_event level
The `_process_event` method handles each stream event. But by the time an error reaches here, the tool has already failed inside LangGraph's execution. We can't retry from `_process_event`.

### Recommended: Option A with checkpointer awareness

```python
async def ainvoke(self, input, *, config=None, **kwargs):
    try:
        return await self._ainvoke_internal(input, config=config, **kwargs)
    except Exception as exc:
        hook_err = _extract_governance_blocked(exc)
        if hook_err is None or hook_err.verdict != "require_approval":
            raise
        # Poll for approval
        await poll_until_decision(...)
        # Retry — with checkpointer this resumes, without it restarts
        return await self._ainvoke_internal(input, config=config, **kwargs)
```

## Related Code Files

- **Modify:** `openbox_langgraph/langgraph_handler.py` — `ainvoke()`, `astream_governed()`
- **Add:** `_extract_governance_blocked()` helper (port from deepagent SDK)

## Changes

### 1. Add `_extract_governance_blocked()` helper
Port from deepagent SDK's `middleware_hooks.py` — identical function.

### 2. Wrap `ainvoke()` with retry logic
Catch wrapped `GovernanceBlockedError(require_approval)`, poll, retry.

### 3. Wrap `astream_governed()` with retry logic
Same pattern but for streaming — on retry, yield from a fresh stream.

## Risk Assessment

- **Without checkpointer:** Retry re-executes everything from scratch — LLM calls are re-billed, tools re-run. This is acceptable for simple graphs but expensive for multi-step agents.
- **With checkpointer:** Retry resumes from last checkpoint — only the failed step re-runs. This is the expected production setup.
- **Document:** Users should use a checkpointer when hook-level HITL is active.

## Success Criteria

- `REQUIRE_APPROVAL` from OTel hooks during `ainvoke()` polls and retries
- Wrapped errors (APIConnectionError) also caught
- `astream_governed()` handles same case
- Non-governance errors propagate unchanged

## Unresolved Questions

1. Should we add `_extract_governance_blocked` to `openbox_langgraph` as a shared utility (used by both SDKs)?
2. For `astream_governed()`, do we need to yield partial results from the first attempt before retry, or discard and restart the stream?
