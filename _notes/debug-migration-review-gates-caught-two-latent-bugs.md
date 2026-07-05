---
type: debug
date: 2026-07-05
tags: [migration, langchain-core-callback, review-gates, langgraph, agentmiddleware]
status: active
---

# Migration session: review gates caught two latent bugs green tests missed

Session journal for the LangChain-Core callback-adapter migration (plan
`plans/260704-2345-langchain-core-callback-adapter/`). Architecture is in
[[decision-langchain-core-callback-owned-lifecycle]]; per-phase detail is in the
plan's phase Results notes. This note records only the process lessons.

## Outcome
Phases 0-5 landed, each behind a mandatory code-review + test gate. Final: langchain
130 tests, langgraph 240 tests, goldens byte-identical, ruff clean, mypy
clean/baseline-parity. Committed on `feat/openbox-core-base-sdk-migration`
(langchain `4baf804`, langgraph `6a19c7c`). Live-Core Demo 04 acceptance is the only
open item (needs a reachable Core + dashboard).

## The two bugs a passing suite hid (the reason the gates paid off)
1. **Middleware turn-state ContextVar didn't survive LangGraph's per-node tasks.**
   The AgentMiddleware rebuild stored per-run turn identity (workflow_id, run_id,
   pre-screen verdict) in a per-instance `ContextVar` set in `before_agent`. Unit
   tests called the hooks directly in one task and passed. But `create_agent().invoke()`
   runs each node (`model`, `tools`) as a SEPARATE task whose context is copied from
   the graph-invocation parent, not from `before_agent`'s task — so every real
   invocation raised `RuntimeError: no turn state bound`. **Lesson:** middleware hook
   state that must cross nodes has to ride the graph STATE (`state_schema` +
   `request.state`), never a ContextVar. An e2e test through the real `create_agent`
   is mandatory — direct-hook unit tests structurally cannot catch this.
2. **Tool-approval poll deferred to a server-echoed `identifier` over the tool's
   activity_id.** The shared lifecycle approval resolver preferred
   `result.raw["identifier"]` (correct as a DISPLAY id for hook blocks) as the HITL
   POLL key too. Core matches approvals on `activity_id`, and `poll_until_decision`
   is an unbounded `while True` — so if Core ever echoed an `identifier` on a tool
   REQUIRE_APPROVAL, `ainvoke` would hang forever. FakeCore never echoes it, so tests
   were green. **Lesson:** split the poll-key resolution (`_resolve_approval_activity_id`,
   activity_id-first) from the display-id resolution; guard with a test that injects a
   bogus `raw["identifier"]` rather than relying on the fake's default shape.

Both were found by adversarial code-review that traced the real dispatch, not by the
suite. A third (duplicate ActivityStarted from a missing cross-dispatch guard in the
LLM callback mixin) surfaced the same way during Phase 5.

## Reusable gotchas
- Editable-installing the langchain SDK into the langgraph repo makes its top-level
  `tests/` package shadow this repo's un-packaged `tests/` under pytest prepend mode →
  collection breaks. Fix: `addopts = "--import-mode=importlib"`.
- Golden `*.raw.json` carry non-deterministic ids/timestamps; regenerating them churns
  git without semantic change. The `*.normalized.json` are the real assertions — revert
  raw churn before committing.
- `BaseTool.run` ignores `config={"callbacks": [...]}`; it reads the `callbacks=` kwarg.
  ToolNode reaches it via the Runnable `invoke`/`ainvoke` which DO translate config
  callbacks — but a direct `.run(config=...)` in a test silently installs nothing.
