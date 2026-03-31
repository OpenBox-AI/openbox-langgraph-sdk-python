# Port DeepAgent Governance Fixes to LangGraph SDK

**Status:** planned
**Branch:** feat/otel-http-hook-spans
**Priority:** high
**Repo:** openbox-langgraph-sdk-python

## Overview

Port 4 governance fixes from the deepagent SDK to the langgraph SDK's `OpenBoxLangGraphHandler`. The langgraph SDK uses `astream_events` instead of middleware hooks, so the implementation differs but the gaps are identical.

## Phases

| # | Phase | Status | File |
|---|-------|--------|------|
| 1 | Remove hitl.enabled gate + hitl.skip_tool_types | planned | [phase-01](phase-01-remove-hitl-gates.md) |
| 2 | Add sqlalchemy_engine parameter | planned | [phase-02](phase-02-sqlalchemy-engine.md) |
| 3 | Add hook HITL retry + wrapped error unwrapping | planned | [phase-03](phase-03-hook-hitl-retry.md) |

## Gap Analysis

| Fix | DeepAgent SDK | LangGraph SDK |
|---|---|---|
| Remove `hitl.enabled` gate | Done (3 sites) | Needed (3 sites: lines 558, 845, 873) |
| Remove `hitl.skip_tool_types` | Done | Needed (2 sites: lines 846, 876) |
| `sqlalchemy_engine` param | Done | Needed (options + factory + __init__ OTel setup) |
| `_extract_governance_blocked()` | Done | Needed (new helper) |
| Hook HITL retry in tool handling | Done (awrap_tool_call) | Needed — but architecture is different (astream_events, not middleware) |
| Hook HITL retry in LLM handling | Done (awrap_model_call) | Needed — same gap |

## Architecture Difference

The langgraph SDK doesn't have `awrap_tool_call`/`awrap_model_call`. Instead, `_process_event()` handles all events from `astream_events`. Hook errors from OTel (httpx/DB/file) propagate up through the graph's internal execution and surface as exceptions in `ainvoke()` / `astream_governed()`.

The retry logic needs to be at the `ainvoke()`/`astream_governed()` level — catch the wrapped `GovernanceBlockedError`, poll, then re-invoke the graph. But this is different from the deepagent SDK where we retry a single tool/LLM call. Here, re-invoking the entire graph would restart from scratch.

**Alternative:** The hook-level retry may need to be at the `_process_event` level or deferred to the hook_governance module itself. This needs investigation during Phase 3.

## Dependencies

- Phases 1-2 are straightforward port — no architectural changes
- Phase 3 requires design investigation for the astream_events architecture
