# Code Review — Zero-fallback span-to-activity binding

Branch: `feat/openbox-core-base-sdk-migration`
Plan: `plans/260704-0236-zero-fallback-span-activity-binding/plan.md`
Reviewer: code-reviewer | 2026-07-04

## Verdict

**APPROVE with minor concerns.** All 7 acceptance criteria are met and verified against actual code + a real (non-mocked) test suite (212 passed). The core mechanism — mint-and-WRITE canonical id into `config["run_id"]` at the ToolNode seam — is correct: empirically confirmed that the written dict is the same object `execute` closes over, and that `on_tool_start.run_id == bound activity_id`. No new lint/type error types. Findings are documentation drift + one partially-defeated strict-mode guarantee; none block landing.

## Scope

- Files reviewed: `tool_activity_binding.py` (new), `core_runtime.py`, `trace_context_registry.py`, `langgraph_handler.py`, `config.py`, `activity_context_binding.py`, `core_adapter.py`, deleted `fallback_context_store.py` + `tool_activity_scope.py`, 3 golden fixtures, 8 test files.
- Verification: LangGraph `ToolNode._run_one`/`_arun_one`/`_execute_tool_sync` source introspected; mypy baseline diff (HEAD 30 → 27); ruff; full pytest.

## Acceptance criteria — all MET

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | No FallbackContextStore in handler runtimes | ✅ | `core_runtime.py:132` `ContextStore()`; file deleted; grep clean |
| 2 | No single-active/last-registered fallback | ✅ | `resolve()` removed; only `context_for_trace` (exact) + ContextVar tier remain |
| 3 | Hook spans map to exact `ActivityStarted.activity_id` | ✅ | `test_tool_body_resolves_its_own_activity_context` runs a REAL file op inside bound ctx; `test_bound_activity_id_equals_on_tool_start_run_id` asserts `bound == on_tool_start.run_id` |
| 4 | Concurrent tool calls cannot cross-map | ✅ | `test_concurrent_tool_calls_each_bind_their_own_activity` asserts `alpha != beta` with forced interleave |
| 5 | Started/completed pinned to same activity | ✅ | `LangGraphHookRuntime` unchanged; `test_hook_context_pinning.py` retained (now on plain `ContextStore`) |
| 6 | Tool lifecycle ids aligned (`-c` removed) | ✅ | `langgraph_handler.py:1559`; goldens updated; dedup key `(activity_id, event_type)` at `client.py:293` keeps started/completed distinct (`ToolStarted→ActivityStarted`, `ToolCompleted→ActivityCompleted` are distinct server types) |
| 7 | Full tests pass | ✅ | 212 passed, 1 warning |

Mandatory checks (a)–(e): all pass. (d) — the `config["run_id"]` write mutates the SAME dict `execute` closes over (verified in `ToolNode._run_one`: `config = tool_runtime.config` local aliases `request.runtime.config`; `execute` closes over that local; `_execute_tool_sync` passes it to `tool.invoke(call_args, config)`). Idempotency mark `_BIND_MARK` verified by `test_toolnode_bound_idempotently`. Both wrapper seams always installed → native-async path preserved (confirmed against `_arun_one`, which routes to `_wrap_tool_call` sync only when `_awrap_tool_call is None`). (e) — no new error TYPES; the change REMOVED `call-arg`/`attr-defined`/`assignment` errors.

## Findings (most severe first)

### H1 — Strict mode's "raise loudly" is defeated by ToolNode's default error handling (HIGH; PARTIALLY reproduces)

`strict_activity_context=True` raises `OpenBoxConfigError` in `_prepare_context` before the tool body runs (good — no unbound execution). BUT that raise happens inside `openbox_sync`/`openbox_async`, which `ToolNode._run_one`/`_arun_one` call inside a `try/except Exception`. With the **default** `handle_tool_errors` (truthy — LangGraph default is `_default_handle_tool_errors`), the `OpenBoxConfigError` is **caught and converted to `ToolMessage(status="error")`** — it does NOT propagate to the caller. The graph continues past it.

The strict-mode test only passes because it sets `handle_tool_errors=False` (test line 300, comment: "let the strict raise propagate, not become a ToolMessage"). So in a realistic default ToolNode, strict mode prevents unbound execution but its intended loud/observable failure is swallowed — a developer running the graph and expecting an exception will instead get a silent error ToolMessage.

- Actually reproduces with default ToolNode config (the common case). Only the documented test config (`handle_tool_errors=False`) surfaces the raise.
- Impact: strict mode is meant for "catch in test/debug rather than silently ungoverned." With default error handling it is caught but not loud — a weaker guarantee than the docstring (`config.py:99-109`) and plan §4 imply.
- Suggested fix: raise a `GraphBubbleUp`-derived exception (LangGraph re-raises those unconditionally per `_execute_tool_sync`) OR document explicitly that strict mode requires `handle_tool_errors=False` on the ToolNode. At minimum, note the limitation in `strict_activity_context`'s docstring.

### M1 — Stale comment + vestigial ordering logic in `trace_context_registry.py` (MEDIUM; does NOT affect correctness)

`register()` at lines 84-86 keeps an `OrderedDict` and re-inserts to "move to end" with the comment *"the last-registered fallback tier stays accurate on re-registration."* That fallback tier was removed in this change. `_by_trace` is now consumed ONLY by `sweep` (`list(self._by_trace.keys())` — order-irrelevant) and `unregister` (pop). The stored `ctx` value and the ordering are both dead: nothing reads most-recent, nothing reads the value. It could be a plain `set[int]`.

- Behavior is correct; this is misleading-comment + YAGNI cleanup. The comment also contradicts the module's own docstring (lines 13-14: "deliberately no ... last-registered guessing").
- Suggested: drop the OrderedDict → `set[int]`, delete the stale comment (or reduce to "set of this turn's trace keys, for sweep").

### M2 — Stale docstring in `langgraph_hook_runtime.py:10-12` (MEDIUM; file unchanged, doc now wrong)

The module docstring still describes resolution as "a trace-lookup fallback ladder ... exact → single-active → last-registered." That ladder no longer exists. The pinning fix itself is still valid and still needed (a plain-`ContextStore` trace map can still re-resolve to a later activity between started/completed — proven by `test_completed_reuses_pinned_context_despite_store_drift`), so this is doc-only drift, not a logic bug.

- Suggested: update the "why this exists" paragraph to describe drift over an exact trace map, dropping the removed ladder.

### L1 — LLM span/token leak if `on_llm_end`/`on_llm_error` never fires (LOW; theoretical)

`_start_and_register_llm_trace` does `otel_context.attach()` + `start_span()` and stores the handle; cleanup is in `on_llm_end`/`on_llm_error`. If neither callback fires (e.g. cancelled stream), the span stays un-ended and the token un-detached. Mitigations already present: (1) the trace REGISTRATION is swept at turn-exit via `_cleanup_turn → registry.sweep(workflow_id)` (all 3 invoke paths), so no unbounded store growth; (2) the callback handler is per-turn and GC'd, so the un-detached token becomes moot when its task ends. Residual cost is a minor telemetry gap (un-ended `llm.call` span), not a governance or memory bug.

- Does not reproduce in tests (they always drive end/error). Real cancelled-stream case only.

### L2 — Cross-context OTel detach for LLM span not proven under real callback dispatch (LOW; needs real-LLM validation)

`on_chat_model_start` attaches the `llm.call` span context to the current task; `on_llm_end` detaches. Tests exercise this in ONE coroutine (detach succeeds, context restored — `test_pre_llm_callback...` line 277). Under real LangChain callback dispatch, if start/end run in different contextvars.Contexts, `otel_context.detach(token)` logs an OTel error (does not raise) and may not restore the origin context. This is the LLM-trace-registration feature (a real ADDITION beyond the prompt's "-c + wire binder" summary), so flagging it for real-LLM smoke validation.

- Governance correctness unaffected (the registered trace resolves the right activity regardless); this is context-hygiene only.

## Non-issues verified (audit trail)

- `-c` removal safe: dedup keys on `(activity_id, server_event_type)`; started/completed are distinct server types. Goldens (`tool_completed`, `subagent_tool_completed`, `ordering_tool_call_turn`) updated; LLM goldens correctly UNCHANGED (LLM `-c` intentionally kept at `langgraph_handler.py:1158`).
- No public-contract break: `FallbackContextStore`/`ContextMissMetrics`/`wrap_graph_tools`/`tool_activity_scope` were never in `__init__.__all__` and are not importable post-change (verified). No lingering imports of deleted modules.
- Abort-sweep reaches registry in production: `create_core_runtime:142` sets `store.registry = get_trace_registry(runtime)`; `core_adapter.reset_after_approval:180` reads it via `getattr`; adapter's `_store` IS that store. E2E covered by `test_reset_after_approval_clears_the_turns_abort_marks` + the retry test.
- Native-async preserved; user-wrapper composition works (`test_existing_toolnode_wrappers_still_run_inside_the_bound_scope`, asserts order + bound id). Sync-only-user-wrapper-on-async-path superseding is documented in the binder docstring — acceptable (running a sync wrapper on an async path was already degraded).
- Turn-context reset after execution (`test_...:478` — store resolves None post-turn).
- mypy 30→27, ruff clean, no new error types.

## Recommended actions

1. **H1** — Either document that `strict_activity_context` requires `handle_tool_errors=False`, or raise via a `GraphBubbleUp` subclass so ToolNode re-raises it. Cheap doc fix at minimum. (Blocking only if strict mode is a shipped guarantee; informational if test/debug-only.)
2. **M1** — Simplify `_by_trace` to `set[int]`, delete the stale "last-registered" comment.
3. **M2** — Update `langgraph_hook_runtime.py` docstring to drop the removed ladder.
4. **L2** — Smoke-test the LLM-trace path against a real streaming model to confirm start/end detach hygiene.

## Unresolved questions

1. Is `strict_activity_context` intended as a shipped production guarantee, or test/debug only? Determines whether H1 is blocking. (The docstring frames it as "caught in test/debug" — leaning informational — but "raise before executing the tool" over-promises for default ToolNodes.)
2. The LLM-trace-registration callback path is a substantial feature not described in the review prompt's change summary (which said "Phase 3 `-c` removal + wire binder"). Was it intended to land in THIS change, or did it arrive with an earlier commit on the branch? (It is well-tested and correct; flagging only for scope awareness.)
