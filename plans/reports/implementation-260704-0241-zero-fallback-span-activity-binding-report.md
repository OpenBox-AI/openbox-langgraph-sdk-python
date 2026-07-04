# Implementation: zero-fallback span-to-activity binding

Plan: `plans/260704-0236-zero-fallback-span-activity-binding/plan.md`
Branch: feat/openbox-core-base-sdk-migration | Base: c723c90

## Outcome

All 4 phases done. 212 tests pass, ruff clean, mypy 27 errors (all pre-existing;
baseline HEAD = 30; zero new error types). Code review: DONE_WITH_CONCERNS →
all concerns resolved (below).

## Key deviation from plan (verified, not a scope change)

Plan §2 said READ the canonical activity id from
`ToolCallRequest.runtime.config["run_id"]`. Empirically that field is `None` at
the ToolNode wrapper seam — LangChain mints the tool run id INSIDE `execute()`,
AFTER the wrapper runs. Inverted the mechanism: MINT a canonical uuid, WRITE it
into `request.runtime.config["run_id"]` before `execute()`, so `on_tool_start`
(hence `ToolStarted.activity_id`) and the bound `activity_scope` share one id.
Same target invariant, same acceptance criteria. Proven by 4 probes:
config["run_id"] is None at seam; writing it makes on_tool_start use it;
activity_scope at the seam propagates into async bodies + sync executor threads
+ isolates concurrent calls; compiled-graph ToolNode instance patching + user
wrapper composition works. Also fixes the prior review's CRITICAL concurrency
bug (old `_TURN` module ContextVar cross-contaminated concurrent turns).

## Changes

- NEW `tool_activity_binding.py` — patches ToolNode `_wrap_tool_call`/
  `_awrap_tool_call` (composing any user wrapper); mint+write canonical id;
  bind `activity_scope`; no-context policy (default warn-once+unbound, strict
  raise `OpenBoxConfigError`). Replaces deleted `tool_activity_scope.py` (deep
  func-wrap, minted-but-unmatched uuid).
- `core_runtime.py` — plain `ContextStore` (was `FallbackContextStore`);
  publishes `store.registry = get_trace_registry(runtime)` so the adapter's
  post-approval abort sweep still reaches the registry.
- `trace_context_registry.py` — removed `resolve()` fallback ladder +
  `ContextMissMetrics`; `_by_trace` reduced OrderedDict→set (value/order dead
  after ladder removal). Kept register/unregister/sweep/clear_aborted.
- DELETED `fallback_context_store.py`.
- `langgraph_handler.py` — Phase 3: tool `completed_activity_id = event_run_id`
  (dropped `-c`); wired `bind_tools_activity_scope`. (LLM `-c` at ~L1158 left
  untouched — out of plan scope.)
- `config.py` — `strict_activity_context: bool = False`.
- `activity_context_binding.py` — `tool_call_id` param on build_activity_context.
- Tests: rewrote test_activity_context_binding.py (7 validation tests),
  test_context_fallback_task_and_thread.py, test_hook_approval_retry.py;
  swaps in test_hook_context_pinning/turn_isolation/core_context_binding.
- Goldens: 3 normalized/ordering fixtures (`-c` removed). Raw fixtures reverted
  (regeneration noise, not asserted).

## Code-review concerns — resolution

- H1 (strict raise swallowed by ToolNode default handle_tool_errors): NOT a bug
  — tool never runs unbound, error is visible (ToolMessage status=error). strict
  is a test/debug aid (plan §4). Documented the interaction + handle_tool_errors=False
  escape in config.py docstring. GraphBubbleUp rejected (semantically wrong —
  it's LangGraph's interrupt/control signal, not an error).
- M1 (vestigial OrderedDict in registry): fixed → `set[int]`.
- M2 (stale ladder docstring in langgraph_hook_runtime.py): fixed.
- L1/L2 (LLM-trace-registration span leak / cross-context detach): pre-existing
  WIP, not this change — see unresolved Q1.

## Acceptance criteria — all met

No active FallbackContextStore; no single-active/last-registered fallback (ladder
deleted); hook spans in tools map to exact ActivityStarted.activity_id;
concurrent tool calls isolated; started/completed ids aligned; pin preserved;
full tests pass.

## Unresolved questions

1. Working tree mixes this change with PRE-EXISTING uncommitted LLM-trace-
   registration WIP (`_GuardrailsCallbackHandler._start_and_register_llm_trace`
   in langgraph_handler.py + test_core_context_binding.py additions), from a
   prior session. Intertwined in one file. Include in the same commit, or
   separate? (User decision at commit time.)
2. Ship `strict_activity_context` as documented (test/debug aid, soft error
   under default ToolNode), or upgrade to hard-propagate always? Plan framed it
   test/debug; shipped as such.
