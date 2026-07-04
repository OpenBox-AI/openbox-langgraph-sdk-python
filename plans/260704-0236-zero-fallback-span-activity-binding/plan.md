# Zero-fallback span-to-activity binding for LangGraph tools

## Status: COMPLETE (2026-07-04)

All 4 phases implemented; 212 tests pass, ruff clean, no new mypy errors; code
review passed (concerns resolved). Report:
`plans/reports/implementation-260704-0241-zero-fallback-span-activity-binding-report.md`.

**Mechanism correction (verified empirically):** §2's "READ
`config["run_id"]`" is impossible — that field is `None` at the ToolNode seam
(the run id is minted inside `execute()`, after the wrapper). Implemented the
inverse: MINT a canonical id and WRITE it into `config["run_id"]` before
`execute()`, so `on_tool_start`/`ToolStarted.activity_id` and the bound
`activity_scope` share one id. Target invariant + acceptance criteria unchanged.

## Goal

Make every hook span that `openbox_core` evaluates from inside a LangGraph tool
map to the exact Activity row that represents that tool invocation.

Target invariant:

```text
LangGraph on_tool_start
  -> Core ActivityStarted
     workflow_id = W
     run_id = R
     activity_id = A
     activity_type = web_search

actual web_search execution
  -> HTTP/DB/file/function hook span
     workflow_id = W
     run_id = R
     activity_id = A
     activity_type = web_search

LangGraph on_tool_end
  -> Core ActivityCompleted
     workflow_id = W
     run_id = R
     activity_id = A
     activity_type = web_search
```

No guessed context is allowed. A hook span is attached only when the SDK can
prove the current `ActivityContext`.

## Current problem

The SDK currently has two different identities for one tool invocation:

1. `langgraph_handler.py::_map_event(on_tool_start)` sends lifecycle
   `ToolStarted -> ActivityStarted` using LangGraph's event `run_id` as
   `activity_id`.
2. `tool_activity_scope.py` binds an `ActivityContext` around the tool body, but
   it mints a fresh `uuid.uuid4().hex` because the deep `tool.func` /
   `tool.coroutine` boundary has already lost LangGraph's tool run id.

That gives hook spans the right `workflow_id`, `run_id`, and `activity_type`,
but not the exact lifecycle `activity_id`.

`FallbackContextStore` / `TraceContextRegistry` makes this harder to debug
because a missing exact context can turn into "single active" or "last
registered" context. That is useful as a safety net, but it violates the target
contract: exact or unbound, never guessed.

## Non-goals

- Do not reintroduce LangGraph-local hook payload builders.
- Do not move HTTP/DB/file/function instrumentation out of `openbox_core`.
- Do not use single-active or last-registered fallback correlation.
- Do not silently attach spans to a "best guess" activity.
- Do not require users to change their agent/tool code for normal `ToolNode`
  graphs.

## Ownership boundary

`openbox-langgraph-sdk-python` owns:

- deciding which LangGraph Activity is active
- binding the exact `ActivityContext` around framework execution
- translating base SDK verdicts into LangGraph-native exceptions
- LangGraph-specific tests proving exact tool/span correlation

`openbox-sdk-python` / `openbox_core` owns:

- HTTP/DB/file/function instrumentation
- hook event construction
- span serialization
- governance gate evaluation
- generic `ContextStore` behavior

## Design

### 1. Replace fallback store with plain exact store

In `core_runtime.py::create_core_runtime`, construct the runtime with the base
SDK's plain `ContextStore`, not `FallbackContextStore`.

Allowed context resolution paths:

1. `ContextStore.current_activity_context()` from `activity_scope`.
2. Exact `ContextStore.context_for_trace(trace_id)` only when this SDK explicitly
   registered a known parent trace for a known activity.

Forbidden paths:

1. single-active fallback
2. last-registered fallback
3. implicit "current latest activity" fallback

Keep hook runtime pinning. `LangGraphHookRuntime` does not guess; it pins the
context resolved at hook STARTED and reuses that same context at COMPLETED.

### 2. Bind at the LangGraph `ToolNode` request boundary

Stop using the deep `tool.func` / `tool.coroutine` wrapper as the primary path.
It is too late; the exact LangGraph run id is not available there.

Use LangGraph `ToolNode`'s tool-call wrapper seam instead:

- `ToolNode(..., wrap_tool_call=...)`
- `ToolNode(..., awrap_tool_call=...)`

For already-compiled graphs, patch each discovered `ToolNode` instance by
composing its existing `_wrap_tool_call` / `_awrap_tool_call` with an OpenBox
wrapper.

This wrapper receives:

- `ToolCallRequest.tool_call["name"]`
- `ToolCallRequest.tool_call["id"]`
- `ToolCallRequest.tool_call["args"]`
- `ToolCallRequest.runtime.config`
- the `execute(request)` callable that performs the real tool invocation

`ToolCallRequest.runtime.config["run_id"]` is the preferred canonical
`activity_id`, because `BaseTool.invoke()` forwards that same run id into
`BaseTool.run(..., run_id=...)`, which is what produces the LangChain
`on_tool_start` event that the handler later maps to `ActivityStarted`.

### 3. Build one canonical `ActivityContext`

Add a small LangGraph-owned binder, for example:

```text
openbox_langgraph/tool_activity_binding.py
```

Responsibilities:

- extract turn ids from `RunnableConfig["metadata"][TURN_METADATA_KEY]`
- extract canonical activity id from `RunnableConfig["run_id"]`
- build `ActivityContext` with:
  - `workflow_id`
  - `run_id`
  - `workflow_type`
  - `task_queue`
  - `activity_id`
  - `activity_type`
  - `activity_input`
  - `agent_name`
  - `session_id`
  - `multi_agent_session_id`
  - metadata: `tool_name`, `tool_type`, `tool_call_id`
- bind it with `activity_scope(ctx, store=runtime.context_store)`
- call the original ToolNode execute callable inside that scope
- guarantee reset in `finally` via `activity_scope`

The wrapper must not build or send hook payloads. It only supplies context.

### 4. No-context policy

If the ToolNode wrapper cannot prove the canonical activity id, it must not mint
a replacement id.

Default policy:

```text
log warning + execute unbound
```

Strict test/debug policy:

```text
raise OpenBoxConfigError or GovernanceBlockedError before executing unbound
```

Suggested config option:

```text
strict_activity_context: bool = False
```

The warning should include enough data to fix the missing binding:

- tool name
- tool_call_id
- workflow_id/run_id presence
- whether `config["run_id"]` was present
- graph/tool node class name

Do not fall back to a guessed activity.

### 5. Lifecycle `ActivityCompleted` should use the same id

Change `on_tool_end` mapping to close the same activity id used by
`on_tool_start`.

Current:

```text
ToolStarted.activity_id = event.run_id
ToolCompleted.activity_id = f"{event.run_id}-c"
```

Target:

```text
ToolStarted.activity_id = event.run_id
ToolCompleted.activity_id = event.run_id
```

The client dedup key already includes server event type, so
`(activity_id, ActivityStarted)` and `(activity_id, ActivityCompleted)` remain
distinct.

If Core currently requires the `-c` suffix, keep wire compatibility temporarily
but record the canonical started activity id in metadata, then remove the suffix
in a separate Core-compatible change. The preferred model is one activity id for
the activity lifecycle.

### 6. Exact trace registration remains allowed, fallback does not

Using the base `ContextStore.register_trace(trace_id, ctx)` is acceptable only
when this SDK creates or owns a parent span for that exact activity.

Allowed:

```text
known parent span trace_id -> known ActivityContext
```

Forbidden:

```text
unknown child trace -> single active context
unknown child trace -> last registered context
```

For normal tools, `activity_scope` should be the primary mechanism. Exact trace
registration can remain a secondary path for boundaries where ContextVars cannot
propagate but the SDK owns the parent trace.

## Implementation plan

### Phase 1: Add exact ToolNode binding

Files:

- `openbox_langgraph/tool_activity_scope.py` or new
  `openbox_langgraph/tool_activity_binding.py`
- `openbox_langgraph/langgraph_handler.py`

Work:

1. Replace `wrap_graph_tools()`'s deep `tool.func` / `tool.coroutine` wrapping
   with ToolNode wrapper composition.
2. Preserve existing user-provided `wrap_tool_call` / `awrap_tool_call` by
   composing:

   ```text
   OpenBox wrapper
     -> existing wrapper, if present
       -> ToolNode execute
   ```

3. Build `ActivityContext.activity_id` from
   `request.runtime.config.get("run_id")`.
4. Bind `activity_scope(ctx, store=core_runtime.context_store)` around the
   entire `execute(request)` call.
5. Include `tool_call_id` in metadata, but do not use it as the primary
   activity id.
6. Delete the minted `uuid.uuid4().hex` tool activity id path.

### Phase 2: Remove fallback correlation from the active runtime

Files:

- `openbox_langgraph/core_runtime.py`
- `openbox_langgraph/fallback_context_store.py`
- `openbox_langgraph/trace_context_registry.py`
- tests that assert fallback behavior

Work:

1. Build the runtime with `openbox_core.context.ContextStore`.
2. Stop importing/constructing `FallbackContextStore`.
3. Stop using `TraceContextRegistry` as an active fallback ladder.
4. Keep exact `ContextStore.register_trace` / `unregister_trace` only where the
   SDK explicitly owns a trace id for a known activity.
5. Retire or rewrite tests that expect single-active/last-registered fallback.

### Phase 3: Align tool completion ids

Files:

- `openbox_langgraph/langgraph_handler.py`
- golden fixtures under `tests/golden/`

Work:

1. Change `ToolCompleted.activity_id` from `f"{event_run_id}-c"` to
   `event_run_id`.
2. Confirm `GovernanceClient._is_duplicate()` still permits one started and one
   completed event for the same activity id.
3. Regenerate/update golden fixtures.

### Phase 4: Make unbound hooks obvious

Files:

- `openbox_langgraph/config.py`
- `openbox_langgraph/tool_activity_binding.py`
- tests

Work:

1. Add `strict_activity_context` option, default `False`.
2. In default mode, warn once per tool invocation when exact binding cannot be
   built.
3. In strict mode, raise before executing unbound tool work.
4. Add metrics/log counters for:
   - exact tool binding success
   - missing turn metadata
   - missing tool run id
   - hook skipped because no bound context

## Validation tests

### Required new tests

1. **Exact tool span activity id**

   Build a real ToolNode graph with a `web_search` or probe tool that performs a
   real hookable operation. Assert the hook event sent by base instrumentation
   has:

   ```text
   workflow_id == ToolStarted.workflow_id
   run_id == ToolStarted.run_id
   activity_id == ToolStarted.activity_id
   activity_type == ToolStarted.activity_type
   ```

2. **No minted uuid**

   Assert no hook `ActivityContext.activity_id` is a new uuid unrelated to the
   LangGraph `on_tool_start` run id.

3. **No fallback guessing**

   Create two active tool contexts and trigger a hook without a bound context.
   Assert it does not attach to either one by single-active/last-registered
   fallback.

4. **Concurrent tool calls**

   Run two tool calls in the same ToolNode concurrently. Each tool performs a
   hookable operation. Assert each hook maps to its own activity id.

5. **Existing ToolNode wrappers compose**

   Build a ToolNode with user-provided `wrap_tool_call` and `awrap_tool_call`.
   Assert the user wrapper still runs and OpenBox binding still wraps the actual
   execute call.

6. **Tool completion id parity**

   Assert `ActivityStarted.activity_id == ActivityCompleted.activity_id` for one
   tool invocation, unless Core compatibility forces a temporary suffix.

7. **Strict missing-context mode**

   Force missing `config["run_id"]`. Assert strict mode raises before executing
   the tool; default mode logs and executes unbound.

### Existing tests to update

- `tests/test_activity_context_binding.py`
- `tests/test_core_context_binding.py`
- `tests/test_context_fallback_task_and_thread.py`
- `tests/test_hook_context_pinning.py`
- golden lifecycle fixtures for tool started/completed ids

## Expected behavior after implementation

For a tool:

```python
@tool
def web_search(query: str) -> str:
    return httpx.get("https://example.com/search", params={"q": query}).text
```

Core should receive:

```text
ActivityStarted(web_search, activity_id=A)
ActivityStarted(hook_trigger=true, spans=[http started], activity_id=A)
ActivityStarted(hook_trigger=true, spans=[http completed], activity_id=A)
ActivityCompleted(web_search, activity_id=A)
```

There should be no log line saying a hook resolved through single-active or
last-registered fallback, because those paths no longer exist.

## Risks

1. ToolNode internals are semi-private after graph compilation. The plan should
   prefer documented constructor wrapper seams, but compiled graphs may still
   require patching `_wrap_tool_call` / `_awrap_tool_call`.
2. Older LangGraph versions may not expose `ToolCallRequest` or ToolNode wrapper
   seams. If this SDK supports those versions, add a version-gated fallback to
   `BaseTool.run/arun` wrapping, not `tool.func` wrapping.
3. User-created threads inside a tool may not inherit ContextVars. Without
   fallback guessing, spans from those threads are intentionally unbound unless
   the user or base SDK explicitly propagates context.
4. Changing `ToolCompleted.activity_id` may expose Core assumptions about the
   old `-c` suffix. Verify with Core/golden tests before shipping.

## Acceptance criteria

- No active use of `FallbackContextStore` in handler-owned runtimes.
- No active single-active or last-registered span/activity fallback.
- Hook spans emitted inside normal LangGraph tools map to the exact lifecycle
  `ActivityStarted.activity_id`.
- Concurrent tool calls cannot cross-map spans.
- Started and completed hook stages stay pinned to the same activity.
- Tool lifecycle started/completed ids are aligned, or a documented temporary
  compatibility exception remains.
- Full tests pass.

## Suggested validation command

```bash
UV_CACHE_DIR=/private/tmp/openbox-langgraph-uv-cache \
  uv run pytest \
  tests/test_activity_context_binding.py \
  tests/test_core_context_binding.py \
  tests/test_hook_context_pinning.py \
  tests/test_lifecycle_golden_parity.py \
  -q
```
