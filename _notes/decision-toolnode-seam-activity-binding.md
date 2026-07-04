---
type: decision
date: 2026-07-04
tags: [langgraph, governance, activity-context, hooks]
status: active
---

# Tool activity binding at the ToolNode seam (write config["run_id"], not read)

How a hook span fired inside a LangGraph tool maps to the exact Activity row for
that tool invocation. Zero-fallback: exact or unbound, never guessed.

## The gotcha that shaped the design

The obvious approach — READ the tool run id from
`ToolCallRequest.runtime.config["run_id"]` inside a `ToolNode.wrap_tool_call` /
`awrap_tool_call` wrapper — **does not work**. That field is `None` at the
wrapper seam: LangChain mints the tool run id INSIDE `execute()`
(`BaseTool.arun`), AFTER the wrapper runs. Verified empirically (config keys at
the seam: `callbacks, configurable, metadata, recursion_limit, tags` — no
`run_id`).

The working approach is the INVERSE: mint a canonical uuid in the wrapper, WRITE
it into `request.runtime.config["run_id"]`, THEN call `execute()`. `execute`
closes over the SAME `config` dict (`config = tool_runtime.config` in
`ToolNode._arun_one`), and `BaseTool` forwards `config["run_id"]` into the run
it starts — so `on_tool_start` (hence the handler's `ToolStarted.activity_id`)
carries our id, and it equals the `activity_scope` we bound. One id for the tool
lifecycle and every hook span inside it.

See `openbox_langgraph/tool_activity_binding.py`.

## Why the ToolNode seam (not deep `tool.func`/`tool.coroutine`)

The deep func boundary is too late — LangChain already minted the run id above
it, so a wrapper there can't make its bound id equal the lifecycle id. The
ToolNode `wrap_tool_call`/`awrap_tool_call` seam runs BEFORE the run id exists,
which is exactly what lets us write it.

## Verified facts (probes)

- `activity_scope` bound at the seam propagates into async tool bodies AND sync
  tool bodies (which run via `run_in_executor` — the executor copies the context
  at submit time, which is inside our bound scope).
- Concurrent tool calls in one ToolNode each get their own `runtime.config` dict
  and their own bound scope → zero cross-contamination. (This also fixed a prior
  CRITICAL bug where a module-level `_TURN` ContextVar cross-contaminated
  concurrent turns.)
- `on_tool_start.run_id == on_tool_end.run_id`, so tool `ActivityCompleted` can
  share the started `activity_id` (dropped the old `-c` suffix; Core / the
  client dedup key discriminate started vs completed by event_type, not a suffix).
- Compiled-graph ToolNodes are patched by setting `_wrap_tool_call`/
  `_awrap_tool_call` on the instance (read at call time), composing any
  user-provided wrapper INSIDE the bound scope.

## Zero-fallback consequences

- Runtime now uses a plain `openbox_core.context.ContextStore` (not the deleted
  `FallbackContextStore`). Resolution: ContextVar tier (bound at the seam),
  then EXACT trace-id tier (an OTel span this SDK registered for a known
  activity). No single-active / last-registered guessing — that ladder was
  removed from `TraceContextRegistry` (`resolve()` + `ContextMissMetrics` gone).
- Work running where neither tier reaches (e.g. a raw thread a tool spawns) is
  intentionally UNBOUND, not guessed.
- The adapter's post-approval abort sweep reaches the per-runtime registry via a
  dynamic `store.registry` attribute that `create_core_runtime` publishes.

## strict_activity_context (test/debug aid)

`config.strict_activity_context=True` raises `OpenBoxConfigError` before running
a tool that can't be bound (no governed-turn metadata). Sharp edge: the raise is
thrown inside the `ToolNode` call, so a ToolNode with the default
`handle_tool_errors=True` turns it into an error `ToolMessage` (visible, tool
still not run) rather than propagating — use `handle_tool_errors=False` to
hard-fail. Default (`False`) logs once and runs unbound.

Related: [[decision-base-only-hook-runtime]]
