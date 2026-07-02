---
type: decision
date: 2026-07-03
tags: [instrumentation, hooks, openbox-core, migration]
status: active
---

# Base openbox_core is the only hook runtime

## What
Removed all legacy in-repo hook governance from the LangGraph SDK. Deleted
`hook_governance.py`, `http_governance_hooks.py`, `file_governance_hooks.py`,
`db_governance_hooks.py`. Hook span creation, payload shape, gate evaluation,
span serialization, and enforcement are now owned entirely by the base
`openbox_core` instrumentation (`InstrumentationManager` + base `HookRuntime`).

The LangGraph SDK now only:
- Adapts LangGraph lifecycle events → base SDK lifecycle (`core_events.py`).
- Builds/registers/pins `ActivityContext` (`activity_context_binding.py`,
  `trace_context_registry.py`, `langgraph_hook_runtime.py`). `LangGraphHookRuntime`
  is context-only — it pins the started-stage `ActivityContext` and delegates
  every hook behavior to the base `HookRuntime`; it builds no payloads.

## Why
Two divergent hook implementations (legacy in-repo + base) were a
double-governance / last-writer-wins hazard and a maintenance burden. One
runtime = one source of truth for hook payload shape and enforcement.

## Key behavior changes
- `use_core_instrumentation` now defaults **True**. Setting it `False` fails
  fast (`OpenBoxConfigError` from `create_core_runtime`) — no legacy fallback.
- Injected-`client` handlers (`_core_runtime is None`) are **lifecycle-only**:
  no core runtime, no hook instrumentation, no bridging OTel span created in
  `_map_event`. Lifecycle governance still flows through the injected client.
- `setup_opentelemetry_for_governance` kept as an exported **deprecated shim
  that raises** (public-surface stability; `otel_setup.py`).
- `@traced` / `create_span` (`tracing.py`) are now plain OTel span helpers with
  NO inline governance evaluation.
- Handler `_map_event` still creates a bridging OTel span + `register_activity`
  (base trace→activity correlation) gated on `should_dual_write(core_runtime)`;
  the legacy `WorkflowSpanProcessor.register_trace/set/clear_activity_context`
  calls are gone. `WorkflowSpanProcessor`/`WorkflowSpanBuffer` stay exported but
  the handler no longer instantiates them (`self._span_processor` is always None).

## Correction: pathlib IS covered by base
An earlier draft of this note claimed `Path.read_text()`/`write_text()` were
NOT hooked (base patches `builtins.open` only). That is WRONG — verified
empirically: `io.open is builtins.open` (same object), and with an
`ActivityContext` bound, BOTH `open()` and `Path.read_text()` produce base hook
spans that resolve to the bound activity. No base change is needed for pathlib.

## ActivityContext binding around actual tool execution
Base hooks resolve context via `resolve_context(store, span)` → ContextVar tier
FIRST (`store.current_activity_context()`), trace-map second. To make hooks
inside a tool resolve to THAT tool deterministically (like Temporal's
`core_activity_scope`), the SDK binds the tool's `ActivityContext` on the
runtime's private store AROUND the real tool body.

Key constraint discovered: a LangChain **callback cannot** do this. LangChain
dispatches callbacks in an isolated context and `BaseTool.arun` runs the tool
body in a freshly `copy_context()`-ed child — a ContextVar set in
`on_tool_start` never reaches the tool body (verified empirically).

Turn-id channel — MUST be per-invocation, NOT a shared module ContextVar. A
first cut used one module ContextVar set in the consumer; code review caught
(and empirically reproduced) cross-contamination: two turns driven concurrently
on one handler saw each other's ids (last-writer-wins) → governance
misattribution. Fix: thread `{workflow_id, run_id}` through the turn's
`RunnableConfig["metadata"]` — `BaseTool.arun` binds that config on
`var_child_runnable_config` inside the tool body (readable via `ensure_config`,
copied into executor threads for sync tools), which is per-invocation and
concurrency-safe.

Implementation (`tool_activity_scope.py`):
- `turn_metadata(workflow_id, run_id)` → merged into `config["metadata"]` by the
  handler's `_governed_config` per turn.
- `wrap_graph_tools()` wraps every tool's `func`/`coroutine` in the compiled
  graph so the body runs inside `activity_scope(ctx, store)`; `ctx` built from
  the turn ids read via `ensure_config` + tool name/args. Idempotent,
  best-effort (custom BaseTools with no func/coroutine fall back to trace
  registration, logged).

Limitations (documented, acceptable):
- The tool's LangChain run_id is NOT exposed at the func boundary, so the hook
  `activity_id` is a minted uuid — `activity_type` (tool name) is what
  resolution keys on. langgraph_node/step/subagent enrichment isn't available
  at the func boundary either (only the lifecycle event / trace-registration
  backup has them).
- LLM (`llm_call`) has NO wrappable boundary — the model call lives inside
  opaque agent-node code, not a tool. It stays on trace registration (matches
  the task's "leave LLM on existing trace registration" fallback).
- Trace registration in `langgraph_handler._map_event` is kept as a best-effort
  BACKUP, no longer the primary mechanism.
