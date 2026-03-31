# OpenBox LangGraph SDK — Codebase Summary

**Package:** openbox_langgraph (6,647 LOC) | **Version:** 0.1.0 | **License:** MIT

## Module Inventory

| Module | LOC | Purpose |
|--------|-----|---------|
| langgraph_handler.py | 1575 | Main handler: wraps graph, processes stream events, manages activity lifecycle, HITL polling |
| db_governance_hooks.py | 874 | Database intercept hooks (SQLAlchemy, asyncpg, psycopg2, pymongo, redis, MySQL, SQLite) |
| http_governance_hooks.py | 759 | HTTP intercept hooks (httpx, requests, urllib3, urllib) with started/completed stages |
| types.py | 483 | Verdict enum, event types, response parsing, serialization utilities |
| otel_setup.py | 481 | Instrumentation initialization, instrumentor registration, orchestration |
| file_governance_hooks.py | 407 | File I/O hooks (builtins.open, os.fdopen patching) |
| hook_governance.py | 377 | Unified hook-level governance evaluation with activity context resolution |
| client.py | 329 | GovernanceClient (async/sync HTTP client for OpenBox Core API) |
| tracing.py | 320 | @traced decorator, create_span() for manual span creation |
| config.py | 262 | GovernanceConfig, env var parsing, global singleton, validation |
| span_processor.py | 244 | WorkflowSpanProcessor (span processor mapping trace_id → activity context) |
| verdict_handler.py | 204 | Verdict enforcement (block → exception, halt → exception, require_approval → HITL) |
| __init__.py | 138 | Public API exports (45+ symbols) |
| errors.py | 106 | Exception hierarchy (OpenBoxError base + 9 specific exceptions) |
| hitl.py | 88 | Human-in-the-loop approval polling loop |

**Total:** 15 modules, 6,647 LOC avg 443 LOC/module

## Architecture Layers

### Layer 1: LangGraph Event Stream (langgraph_handler.py)
Wraps compiled graph and processes v2 event stream.

**Key Classes:**
- `OpenBoxLangGraphHandler` — drop-in replacement for compiled graph
- `OpenBoxLangGraphHandlerOptions` — config dataclass
- `_GuardrailsCallbackHandler` — LangChain callback for pre-LLM PII redaction
- `_RunBufferManager` — manages in-flight activity state (thread-safe ContextVar)
- `_RootRunTracker` — identifies outermost invocation vs subagents

**Key Methods:**
- `ainvoke()` / `astream()` / `batch()` — delegate to wrapped graph
- `_pre_screen_input()` — guardrails before stream starts (exceptions propagate)
- `_process_event()` — evaluate each v2 event inline during streaming
- `_map_event()` — convert LangGraph event → LangChainGovernanceEvent

**Data Flow:**
```
User calls governed.ainvoke(input, config)
  ↓
_pre_screen_input() [WorkflowStarted, LLMStarted, optional_start guardrails]
  → Evaluate with GovernanceClient
  → Enforce verdict (exceptions propagate to caller, no stream started)
  ↓
Stream v2 events from wrapped graph
  ↓
For each event:
  → _map_event() extracts activity details
  → _process_event() evaluates with GovernanceClient
  → enforce_verdict() applies verdict (BLOCK → exception, REQUIRE_APPROVAL → HITL)
  → LLM: _GuardrailsCallbackHandler.on_chat_model_start() redacts PII
  → Yield event to caller
  ↓
on_tool_start: register trace with SpanProcessor for hook lookup
on_tool_end: cleanup
```

### Layer 2: Hook Governance (http/db/file_governance_hooks.py + hook_governance.py)
Intercepts HTTP, database, file I/O at hook level — runs BEFORE operations execute.

**HTTP Hooks (http_governance_hooks.py — 759 LOC):**
- `httpx_send_hook()` — intercepts httpx.Client.send before request
- `requests_hooks()` — intercepts requests.Session.request
- `urllib3_hooks()` — intercepts urllib3 connection
- `urllib_hooks()` — intercepts urllib urlopen
- Body capture: patched httpx.Client.send bypasses stream consumption issue (instrumentation hook sees unconsumed stream)
- Stages: started (can block), completed (informational with response)

**Database Hooks (db_governance_hooks.py — 874 LOC):**
- SQLAlchemy event listener on before_execute
- asyncpg: wrapt wrapping for asyncpg.connection.execute()
- psycopg2/dbapi: CursorTracer patches cursor.execute()
- pymongo: CommandListener for before_command_started
- redis: native hook registration on redis.client.Redis
- MySQL/SQLite: auto-instrumented via opentelemetry-instrumentation-*

**File I/O Hooks (file_governance_hooks.py — 407 LOC):**
- TracedFile wrapper for open()
- os.fdopen() patching with platform-specific fd → path resolution
- Works with sync I/O (patches global builtins.open)

**Unified Hook Evaluation (hook_governance.py — 377 LOC):**
```python
async def evaluate_async(hook_payload):
  → SpanProcessor.get_activity_context_by_trace(trace_id)
  → Build governance payload with activity context
  → GovernanceClient.evaluate_raw() (no dedup, pre-built payloads)
  → Enforce verdict (BLOCK/HALT → GovernanceBlockedError)

def evaluate_sync(hook_payload):
  → Same, but uses GovernanceClient.evaluate_event_sync() (sync wrapper)
```

**Fallback Strategies:**
- Single-activity: if only one activity, assume all hooks belong to it
- Most-recent: if multiple activities, use most recently started

### Layer 3: Activity Context (span_processor.py + otel_setup.py)
Maps trace_id → (workflow_id, activity_id) for hook-level governance.

**WorkflowSpanProcessor:**
- Registers as SpanProcessor
- On span start: if trace_id not mapped, assume new activity
- On span end: preserve mapping for hook lookup
- ContextVar `_workflow_context` stores (workflow_id, activity_id, run_id, trace_id)

**Setup (otel_setup.py — 481 LOC):**
- `setup_opentelemetry_for_governance()` initializes instrumentation SDK if not already done
- Registers WorkflowSpanProcessor
- Auto-instruments: httpx, requests, urllib3, urllib, sqlalchemy, asyncpg, psycopg2, pymongo, redis, sqlite3, mysql
- Can pass existing SDK or engine for manual control

### Supporting Flow

```
User → governed.ainvoke()
  → Pre-screen guardrails
  → Stream events, evaluate inline
    → For tool calls:
      - Span created with trace_id
      - SpanProcessor.register_trace(trace_id → activity_id)
      - HTTP/DB/file hook fires
      - hook_governance.evaluate_sync/async()
      - SpanProcessor.get_activity_context_by_trace() returns (workflow_id, activity_id)
      - Build hook payload, evaluate with Core
      - Enforce verdict (BLOCK → GovernanceBlockedError bubbles up)
      - LLM SDK wraps error, handler unwraps via __cause__/__context__
```

## Key Types

### Verdict Enum (types.py)
```python
Verdict.ALLOW        # priority 0 — allowed
Verdict.CONSTRAIN    # priority 1 — allowed with constraints (logged)
Verdict.REQUIRE_APPROVAL  # priority 2 — wait for HITL approval
Verdict.BLOCK        # priority 3 — reject, raise exception
Verdict.HALT         # priority 4 — halt workflow, raise exception
```

Methods: `from_string()`, `priority`, `should_stop()`, `requires_approval()`, `highest_priority()`

### GovernanceConfig (config.py)
15+ parameters:
- `on_api_error` — "fail_open" | "fail_closed"
- `api_timeout` — seconds (auto-convert ms if >600)
- `send_*_event` flags — control event types sent
- `skip_*_types` — exclude chains/tools from governance
- `hitl` — HITLConfig (enabled, poll_interval_ms)
- `agent_name`, `task_queue` — metadata
- `tool_type_map` — classify tools as "http", "database", "builtin", "a2a"

### LangChainGovernanceEvent (types.py)
Event sent to OpenBox Core:
- `source` — "workflow-telemetry"
- `event_type` — "ChainStarted", "ToolStarted", "LLMStarted", etc.
- `workflow_id`, `run_id`, `activity_id` — unique identifiers
- `activity_type` — "chain", "tool", "llm_call"
- `activity_input`, `activity_output` — captured data
- `timestamp` — RFC3339 format

### GovernanceVerdictResponse (types.py)
Response from OpenBox Core:
- `verdict` — Verdict enum
- `reason` — human-readable explanation
- `redacted_input` — optional PII-redacted version
- `requires_hitl` — whether to poll approval queue

### Exception Hierarchy (errors.py)

```
OpenBoxError (base)
  ├─ OpenBoxAuthError — invalid API key
  ├─ OpenBoxNetworkError — unreachable Core
  ├─ OpenBoxInsecureURLError — HTTP on non-localhost
  ├─ GovernanceBlockedError — BLOCK/HALT verdict
  ├─ GovernanceHaltError — workflow-level HALT
  ├─ GuardrailsValidationError — guardrails fired
  ├─ ApprovalExpiredError — HITL approval expired
  ├─ ApprovalRejectedError — user rejected approval
  └─ ApprovalTimeoutError — HITL polling timed out
```

## Client & API Integration

### GovernanceClient (client.py — 329 LOC)
Async HTTP client for OpenBox Core API.

**Methods:**
- `evaluate_event(event)` → GovernanceVerdictResponse (async)
- `evaluate_event_sync(event)` → GovernanceVerdictResponse (sync wrapper for middleware hooks)
- `evaluate_raw(payload)` → dict (pre-built hook payloads, no dedup)
- `poll_approval(params)` → ApprovalResponse
- `validate_api_key()` — test connection on init

**Dedup Logic:**
- Per (workflow_id, run_id) pair (resets on new invocation)
- Key: (activity_id, event_type)
- Prevents duplicate events within a run

**Error Handling:**
- fail_open: returns None on network error, execution continues
- fail_closed: raises OpenBoxNetworkError, execution halts

## Test Infrastructure

**Unit Tests (tests/):**
- `test_governance_changes.py` — httpx hook behavior, subagent governance
- `test_telemetry_payload.py` — event schema validation
- `test_contextvars_propagation.py` — ContextVar async/sync behavior

**Test Agent (test-agent/):**
- Standalone LangGraph agent for end-to-end validation
- Tools: search_web (HTTP), write_report (in-memory)
- Requires: OPENBOX_URL, OPENBOX_API_KEY, OPENAI_API_KEY
- Tests: guardrails, policies, behavior rules, HITL

## Notable Patterns

1. **Dual async/sync:** GovernanceClient has both evaluate_event() and evaluate_event_sync() to handle async handlers and sync middleware hooks
2. **Pre-screen + stream:** Guardrails enforced before stream (exceptions propagate); stream events evaluated inline
3. **Tool classification:** `__openbox` metadata sentinel appended to activity_input enables Rego policies to detect tool types
4. **Trace bridging:** WorkflowSpanProcessor maps trace_id explicitly (LangGraph spawns tasks with new contexts)
5. **Hook guard:** Rego policies use `not input.hook_trigger` to prevent re-evaluation
6. **Lazy imports:** config.py uses urllib not httpx at module level (sandbox compatibility)
7. **Exception unwrapping:** _extract_governance_blocked() walks __cause__/__context__ chain (LLM SDKs wrap errors)

## Dependency Graph

```
langgraph_handler.py (entry point)
  ├─ config.py (GovernanceConfig)
  ├─ client.py (GovernanceClient)
  ├─ types.py (Verdict, events)
  ├─ verdict_handler.py (enforce_verdict)
  ├─ hitl.py (poll_until_decision)
  ├─ otel_setup.py (setup_opentelemetry_for_governance)
  ├─ span_processor.py (WorkflowSpanProcessor)
  ├─ hook_governance.py (evaluate_sync/async)
  │  ├─ http_governance_hooks.py
  │  ├─ db_governance_hooks.py
  │  └─ file_governance_hooks.py
  └─ tracing.py (@traced, create_span)

__init__.py (public exports)
  └─ all modules above

errors.py (exception hierarchy)
  └─ used by all modules
```

## Code Metrics

- **Total LOC:** 6,647
- **Largest Module:** langgraph_handler.py (1,575 LOC)
- **Average Module:** 443 LOC
- **Type Coverage:** 100% (strict mypy)
- **Test Coverage:** Unit tests for critical paths (client, verdict, hooks)
- **Code Quality:** ruff (100 char, py311), no linting issues

## Build & Testing

**Build Tool:** hatchling
**Package Manager:** uv

**Linting:**
```bash
ruff check openbox_langgraph/
```

**Type Checking:**
```bash
mypy openbox_langgraph/ --strict
```

**Testing:**
```bash
pytest tests/ -v --asyncio-mode=auto
```

**Entry Point:**
```python
from openbox_langgraph import create_openbox_graph_handler
governed = await create_openbox_graph_handler(graph=..., api_url=..., api_key=...)
```

## Recent Changes (from git log)

- feat: add evaluate_event_sync and sync mode SpanProcessor fallback
- feat: instrument os.fdopen() for file I/O governance spans
- fix: enable span-level governance for subagent tools
- feat: add HTTP governance hook spans with started/completed stages
- chore: clean up stale code

## Planned Work

See plans/ directory for active development:
- Remove SpanCollector (favor SpanProcessor only)
- Port DeepAgent SDK fixes (HITL gate removal, SQLAlchemy engine parameter)
- Hook-level HITL retry logic
