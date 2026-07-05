---
type: decision
date: 2026-07-05
tags: [langchain, callbacks, lifecycle, activity-bridge, governance, architecture]
status: active
---

# Tool and LLM lifecycle ownership moved to LangChain-Core callback (producer-side)

Tool call and LLM invocation lifecycle events (ActivityStarted/Completed, LLMStarted/Completed) are now emitted by a **producer-side LangChain-Core callback** (`OpenBoxLangChainCoreAsyncCallbackHandler` / `...SyncCallbackHandler`) instead of the consumer-side LangGraph event stream handler. This fixes Demo 04 span-ordering and correlation bugs.

## Why: The original consumer-side bug

Consuming lifecycle events from the v2 event stream (`on_tool_start`, `on_chat_model_start`) and immediately forwarding them to OpenBox created a race condition:

- A **tool hook span** (e.g., HTTP request inside the tool) fires and captures a `trace_id` → `activity_id` binding via `ActivityContext`.
- But the **tool lifecycle event** hasn't reached OpenBox Core yet (or arrives out of order).
- The hook span governance evaluates against a tool activity that doesn't exist in Core, or the binding is wrong.

Result: Demo 04 showed hook spans arriving BEFORE their parent tool ActivityStarted event, breaking the execution tree.

## The producer-side fix: LangChain-Core callback

The callback layer sits at the **point of generation**, not consumption:

1. **Tool lifecycle:** LangChain's `on_tool_start` callback fires INSIDE the tool execution (after the run_id is minted, before the tool body runs). The callback emits ActivityStarted directly to Core. When the tool body runs, hook spans see the binding immediately. When the tool completes, `on_tool_end` emits ActivityCompleted.

2. **LLM lifecycle:** LangChain's `on_llm_start` callback fires BEFORE the LLM call leaves the client. It emits LLMStarted to Core. Tool code inside the LLM invocation sees the binding. Completion uses the same activity_id (no `-c` suffix).

3. **Ownership contract via `ActivityBridge`:** The callback emits events with a per-event-type sent-flag (e.g., `tool_lifecycle_sent`, `llm_lifecycle_sent`). The langgraph stream-event layer checks these flags and **skips duplicate emission** (only the callback owns those event types).

## Architecture consequences

**openbox-langchain-sdk-python:**
- Pure LangChain-Core adapter on `openbox_core` (no `langgraph` dependency; optional `[agent]` extra only).
- Owns `ActivityBridge`, the core callback handlers, and `lifecycle_events` helpers.
- Optional `OpenBoxLangChainMiddleware` for `create_agent` integration.

**openbox-langgraph-sdk-python:**
- Depends on `openbox-langchain-sdk-python`; installs the callbacks for Phase 4 (tool lifecycle) and Phase 5 (LLM lifecycle).
- Keeps graph wrapping, ToolNode canonical-id binding, `ActivityContext`, stream-event governance for **graph/chain events only** (not tool/LLM).
- Duplicate suppression: stream events check `ActivityBridge` sent-flags; skips governance re-evaluation for callback-owned event types.
- Legacy `_GuardrailsCallbackHandler` retired; `llm_activity_map` / `llm_trace_map` removed.

**Governance semantics:**
- Tool/LLM lifecycle is **producer-owned** (guaranteed correct span nesting and ordering).
- Graph/chain events (WorkflowStarted/Completed) remain stream-event owned via langgraph.
- Stream events serve as **fallback telemetry** when the callback isn't installed (e.g., injected clients, subagent-gated handlers).
- Completions use `gate.aevaluate()` for telemetry (poll-and-continue; never retry-the-graph).
- Sync agents **fail-shut on REQUIRE_APPROVAL** (no working sync approval poller; callback handlers are `run_inline=True`).

## H11 event_run_id alias

For compatibility with H11 (HTTP/1.1 span correlation), the first LLM call in a workflow aliases its activity_id as `event_run_id` in governance events, enabling H11 to correlate completions on the same activity.

## Span ordering (verified)

- Tool ActivityStarted reaches Core before the tool body starts (callback tier).
- Hook spans fired by the tool body arrive with `trace_id` → `activity_id` binding to the tool's activity.
- Tool ActivityCompleted arrives after the tool body finishes.
- **No more out-of-order events; execution tree is correctly nested.**

Related: [[decision-toolnode-seam-activity-binding]]
