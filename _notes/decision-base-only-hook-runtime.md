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

## TODO — belongs in openbox-sdk-python, NOT here
`pathlib.Path.read_text()` / `Path.write_text()` produce NO governance span
because base file instrumentation patches `builtins.open` only. `Path.*` uses
`io.open` internally, which the base wrapper does not cover. Fix belongs in
base openbox_core file instrumentation (add `io.open` / pathlib coverage) —
do NOT re-add LangGraph-local file hooks to paper over it.

See [[arch-hook-governance-ownership]] (write if/when base coverage lands).
