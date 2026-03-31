# OpenBox LangGraph SDK — Project Overview & PDR

## Project Purpose

**openbox-langgraph-sdk** is a Python SDK providing real-time governance and observability for LangGraph agents via OpenBox Core. It intercepts LangGraph v2 stream events, sends them to OpenBox's policy engine (OPA/Rego), and enforces verdicts (ALLOW/BLOCK/CONSTRAIN/REQUIRE_APPROVAL/HALT) — all without modifying your graph code.

**Problem Solved:** Agents operating in production need governance across tool calls, LLM invocations, database queries, file I/O, and HTTP requests. OpenBox centralizes policy enforcement, guardrails (PII, toxicity), and human-in-the-loop approval routing in a single policy engine.

**Key Value:**
- Zero graph changes required
- 5-tier graduated enforcement (allow → constrain → require approval → block → halt)
- Pre-screen guardrails before stream starts (exceptions propagate to caller)
- Hook-level governance for HTTP/DB/file operations
- Human approval queue for sensitive operations
- Full observability via built-in instrumentation

## Project Scope

### In Scope
- Wrapping any compiled LangGraph graph (no code changes needed)
- Processing LangGraph v2 event stream (on_chain_start/end, on_tool_start/end, on_chat_model_start/end)
- Governance evaluation for chain/tool/LLM events
- HITL approval polling and enforcement
- Hook-level governance for HTTP (httpx, requests, urllib3, urllib)
- Hook-level governance for databases (SQLAlchemy, asyncpg, psycopg2, pymongo, redis, MySQL, SQLite)
- Hook-level governance for file I/O (builtins.open, os.fdopen)
- Span tracking and activity context bridging
- PII redaction via guardrails
- Verdict enforcement (block → exception, halt → exception, require_approval → HITL)

### Out of Scope
- Custom policy language (delegated to OPA/Rego at OpenBox Core)
- Modifying LangGraph internals
- Framework-specific integrations (e.g., DeepAgents — separate `openbox-deepagent` package)
- Agent persistence or state management
- GUI dashboard (part of OpenBox Core)

## Functional Requirements

### FR1: Graph Wrapping
The SDK must wrap any compiled LangGraph graph and expose the same interface (ainvoke, astream, etc.) without requiring code changes.

**Acceptance Criteria:**
- `create_openbox_graph_handler()` accepts a graph, API URL, and API key
- Wrapped handler is a drop-in replacement for the original graph
- All LangGraph method signatures preserved (ainvoke, astream, batch, etc.)

### FR2: Event Stream Processing
The SDK must process all v2 stream events (on_chain_start, on_tool_start, on_chat_model_start, etc.) and map them to governance events.

**Acceptance Criteria:**
- Every event type maps to a LangChainGovernanceEvent
- Timestamps, activity IDs, activity types are extracted correctly
- Tool/chain/LLM inputs and outputs are captured
- Subagent classification (a2a tool_type) works correctly

### FR3: Governance Evaluation & Enforcement
The SDK must send governance events to OpenBox Core, receive verdicts, and enforce them.

**Acceptance Criteria:**
- evaluate_event() sends events to /api/v1/evaluate
- Verdicts are parsed and enforced synchronously
- BLOCK/HALT verdicts raise GovernanceBlockedError / GovernanceHaltError
- REQUIRE_APPROVAL triggers HITL polling
- CONSTRAIN is logged but execution continues
- Pre-screen verdicts propagate to caller; stream verdicts are enforced inline

### FR4: Hook-Level Governance
The SDK must intercept HTTP, database, and file I/O operations at the hook level and apply governance before execution.

**Acceptance Criteria:**
- HTTP hooks (httpx, requests, urllib3, urllib) intercept requests before sending
- Database hooks (SQLAlchemy, dbapi libs) intercept queries before execution
- File hooks (open, os.fdopen) intercept operations before execution
- All hooks can block, constrain, or allow operations
- Activity context is resolved via span trace_id → SpanProcessor mapping

### FR5: HITL Approval Queue
When a verdict requires approval, the SDK must poll OpenBox for a decision and wait for human input.

**Acceptance Criteria:**
- poll_until_decision() polls /api/v1/workflows/{workflow_id}/runs/{run_id}/activities/{activity_id}/approval
- Timeout and max_wait_ms are respected
- Approval rejection/expiration raises ApprovalRejectedError / ApprovalExpiredError
- Timeout raises ApprovalTimeoutError

### FR6: Configuration & Global State
The SDK must support both inline config and global environment variables.

**Acceptance Criteria:**
- GovernanceConfig dataclass holds 15+ configurable parameters
- Environment variables (OPENBOX_URL, OPENBOX_API_KEY, etc.) are parsed
- Config can be merged (partial override defaults)
- Global singleton supports initialize() and get_global_config()
- Tool type classification via tool_type_map

### FR7: Observability & Tracing
The SDK must use its built-in instrumentation to track spans and enable activity context resolution.

**Acceptance Criteria:**
- WorkflowSpanProcessor registers with the tracing SDK
- Spans are mapped from trace_id to (workflow_id, activity_id)
- Manual span creation via @traced decorator or create_span()
- Fallback strategies for single-activity or sync mode

### FR8: Error Handling & Debugging
The SDK must provide clear error messages and support debugging via OPENBOX_DEBUG.

**Acceptance Criteria:**
- Exception hierarchy: OpenBoxError base with 9+ specific exceptions
- Network errors are recoverable (fail_open vs fail_closed)
- API key validation on handler creation
- Debug output on OPENBOX_DEBUG=1

## Non-Functional Requirements

### NFR1: Performance
- Governance evaluation adds <100ms latency per event
- HTTP/DB/file hook evaluation adds <50ms per operation
- No unbounded memory growth (dedup resets per run)

### NFR2: Compatibility
- Python 3.11+
- LangGraph >= 0.2.0
- LangChain Core >= 0.3.0
- opentelemetry-api/sdk >= 1.20.0

### NFR3: Reliability
- fail_open mode allows operations if OpenBox is unreachable
- fail_closed mode blocks operations if OpenBox is unreachable
- Dedup prevents sending duplicate events within a run
- Exception unwrapping finds GovernanceBlockedError in wrapped exception chains

### NFR4: Security
- API key validation on initialization
- Insecure HTTP URLs rejected for non-localhost
- Bearer token auth with standard headers
- No credentials logged (lazy imports in config.py)

### NFR5: Code Quality
- Strict mypy: all type hints, no Any unless documented
- 100-char line limit (ruff)
- All modules under 1600 lines for maintainability
- Pre-commit linting: ruff + mypy

## Technical Constraints

1. **Python 3.11+** — Async-first, no legacy Python
2. **AsyncIO integration** — All governance calls are async; sync wrappers exist for middleware hooks
3. **Instrumentation mandatory** — span_processor.py requires instrumentation already active
4. **httpx/requests coexistence** — Both HTTP libs supported via separate hook modules
5. **Exception wrapping** — LLM SDKs wrap httpx errors; unwrap via __cause__/__context__
6. **LangGraph v2 events** — No support for v1 streaming
7. **Dedup scoped to run** — Resets per (workflow_id, run_id) pair, not global

## Dependencies

**Core:**
- httpx (async HTTP)
- langchain-core, langgraph (graph execution)
- opentelemetry-api, opentelemetry-sdk (tracing)
- opentelemetry-instrumentation-* (hook providers)

**Dev:**
- pytest, pytest-asyncio
- ruff (linting)
- mypy (type checking)

## Version & License

- **Version:** 0.1.0
- **License:** MIT
- **Python:** 3.11+
- **Status:** Beta (stable API, active development)

## Success Metrics

1. **Adoption:** SDK installed in production agents with zero graph modifications
2. **Governance:** 100% of tool/LLM/HTTP events evaluated and verdicts enforced
3. **HITL:** Approval queue fully functional with <2s polling latency
4. **Reliability:** <0.1% failure rate in fail_open mode; fail_closed mode blocks only on confirmed unreachable
5. **Observability:** All spans correctly traced and activity context resolved
6. **Developer Experience:** New developers integrate SDK in <5 minutes with Quickstart

## Roadmap

See `project-roadmap.md` for current phase, in-progress work, and planned features.
