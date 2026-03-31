# OpenBox LangGraph SDK — Comprehensive Architectural Analysis

**Report Date:** 2026-03-30 | **Package:** openbox-langgraph-sdk-python v0.1.0 | **License:** MIT

---

## Executive Summary

The OpenBox LangGraph SDK provides **3-layer real-time governance** for compiled LangGraph graphs:

1. **Event Stream Governance** — Intercepts LangGraph v2 events before/during execution
2. **Hook-Level Governance** — Blocks HTTP, DB, file I/O operations at kernel boundary
3. **Activity Context Mapping** — Links trace_id → activity_id for hook attribution

**Key distinctive feature:** Unlike traditional LLM middleware, OpenBox governs at 3 concurrent layers with unified verdict enforcement, enabling pre-execution blocking at hook level while preserving pre-screen guardrails for input validation.

---

## 1. Public API Surface

### Entry Point Function

```python
# openbox_langgraph/__init__.py
def create_openbox_graph_handler(
    graph: Any,
    *,
    api_url: str,                          # Required: "https://openbox.example.com"
    api_key: str,                          # Required: "obx_live_..." or "obx_test_..."
    governance_timeout: float = 30.0,      # HTTP timeout in seconds
    validate: bool = True,                 # Validate API key on startup
    enable_telemetry: bool = True,         # Reserved for future telemetry
    sqlalchemy_engine: Any = None,         # Optional: existing SQLAlchemy Engine
    **handler_kwargs: Any,                 # OpenBoxLangGraphHandlerOptions fields
) -> OpenBoxLangGraphHandler
```

**Internally calls:** `initialize(api_url, api_key, governance_timeout, validate)`

### Main Handler Class

```python
class OpenBoxLangGraphHandler:
    """Drop-in replacement for compiled LangGraph graph."""
    
    async def ainvoke(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]
    
    async def astream_governed(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]
    
    async def astream(...)  # Graph-compatible alias
    async def astream_events(...)  # Graph-compatible passthrough
    
    # Private methods (users don't call directly):
    async def _pre_screen_input(...) -> tuple[bool, GovernanceVerdictResponse | None]
    async def _process_event(...)
    def _map_event(...) -> tuple[LangChainGovernanceEvent, bool, bool, str]
```

### Configuration Class

```python
@dataclass
class OpenBoxLangGraphHandlerOptions:
    client: GovernanceClient | None = None
    on_api_error: str = "fail_open"                    # | "fail_closed"
    api_timeout: int = 30_000                          # milliseconds
    send_chain_start_event: bool = True
    send_chain_end_event: bool = True
    send_tool_start_event: bool = True
    send_tool_end_event: bool = True
    send_llm_start_event: bool = True
    send_llm_end_event: bool = True
    skip_chain_types: set[str] = field(default_factory=set)
    skip_tool_types: set[str] = field(default_factory=set)
    hitl: Any = None                                   # HITLConfig | dict | None
    session_id: str | None = None
    agent_name: str | None = None
    task_queue: str = "langgraph"
    use_native_interrupt: bool = False
    root_node_names: set[str] = field(default_factory=set)
    resolve_subagent_name: Callable[[LangGraphStreamEvent], str | None] | None = None
    sqlalchemy_engine: Any = None
    tool_type_map: dict[str, str] | None = None
```

### Exported Types

```python
# Verdicts
Verdict: StrEnum  # ALLOW, CONSTRAIN, REQUIRE_APPROVAL, BLOCK, HALT
verdict_from_string(value: str | None) -> Verdict
verdict_priority(v: Verdict) -> int
verdict_should_stop(v: Verdict) -> bool
verdict_requires_approval(v: Verdict) -> bool
highest_priority_verdict(verdicts: list[Verdict]) -> Verdict

# Events
WorkflowEventType: StrEnum  # WORKFLOW_STARTED, ACTIVITY_STARTED, etc.
LangGraphStreamEvent: @dataclass  # Raw v2 event from LangGraph
LangChainGovernanceEvent: @dataclass  # Event sent to OpenBox Core

# Configuration
GovernanceConfig: @dataclass
HITLConfig: @dataclass  # enabled, poll_interval_ms, skip_tool_types
DEFAULT_HITL_CONFIG: HITLConfig

# Responses
GovernanceVerdictResponse: @dataclass  # verdict, reason, guardrails_result, etc.
ApprovalResponse: @dataclass  # verdict, reason, expired
GuardrailsReason, GuardrailsResult: @dataclass

# Context
VerdictContext: Literal[...] # "chain_start", "tool_start", "llm_start", "tool_end", etc.
lang_graph_event_to_context(event_type: str, *, is_root: bool = False) -> VerdictContext

# Error Enforcement
enforce_verdict(
    response: GovernanceVerdictResponse,
    context: VerdictContext,
) -> VerdictEnforcementResult

# HITL
poll_until_decision(
    client: GovernanceClient,
    params: HITLPollParams,
    config: HITLConfig,
) -> None  # Blocks until decision, raises on rejection

# Initialization
initialize(
    api_url: str,
    api_key: str,
    governance_timeout: float = 30.0,
    validate: bool = True,
) -> None
get_global_config() -> _GlobalConfigState
merge_config(partial: dict[str, Any] | None = None) -> GovernanceConfig
```

### Exceptions

```python
OpenBoxError (base)
├─ OpenBoxAuthError — invalid API key
├─ OpenBoxNetworkError — Core unreachable
├─ OpenBoxInsecureURLError — HTTP on non-localhost
├─ GovernanceBlockedError(verdict, reason, policy_id=..., risk_score=...)
├─ GovernanceHaltError(reason, policy_id=..., risk_score=...)
├─ GuardrailsValidationError(reasons: list[str])
├─ ApprovalExpiredError
├─ ApprovalRejectedError
└─ ApprovalTimeoutError
```

---

## 2. Core Architecture: 3-Layer Governance

### Layer 1: Event Stream Governance

**File:** `langgraph_handler.py` (1,575 LOC)

**Responsibilities:**
- Wrap compiled LangGraph graph
- Intercept v2 event stream (on_chain_start, on_tool_start, on_chat_model_start, etc.)
- Pre-screen inputs with guardrails before stream starts
- Enforce verdicts inline as events are streamed

**Data Flow:**

```
ainvoke(input) / astream_governed(input)
    ↓
1. _pre_screen_input() [SignalReceived, WorkflowStarted, LLMStarted]
   └─ Send 2-3 governance events synchronously
   └─ Exceptions propagate to caller (block before stream)
    ↓
2. Start streaming v2 events (graph.astream_events())
    ↓
3. For each event:
   a) _map_event() → LangChainGovernanceEvent (extract activity details)
   b) _process_event() → enforce_verdict() (evaluate with Core)
   c) If REQUIRE_APPROVAL → poll_until_decision() (HITL loop)
   d) If BLOCK/HALT → raise exception (unwind stream)
   e) Yield event to caller
    ↓
4. Return final state or stream complete
```

**Key Insight:** Pre-screen runs BEFORE the stream starts. This ensures exceptions propagate to the caller, whereas callback exceptions (even with `raise_error=True`) are silently swallowed by LangGraph's graph runner.

### Layer 2: Hook-Level Governance

**Files:** `http_governance_hooks.py`, `db_governance_hooks.py`, `file_governance_hooks.py`, `hook_governance.py`

**Mechanism:** Intercept I/O at kernel boundary — before HTTP request, DB query, or file write executes.

#### HTTP Hooks (`http_governance_hooks.py`, 759 LOC)

**Supported Libraries:**
- **httpx** (async/sync) — primary
- **requests** — via OTel instrumentation hook
- **urllib3** — via HTTPConnectionPool hook
- **urllib** (stdlib) — via urlopen hook

**Hook Pattern:**

```python
# Example: httpx hook
async def httpx_send_hook(request, call_next):
    # Stage 1: started (BEFORE request)
    span = trace.get_current_span()
    payload = _build_http_span_data(
        span, request.method, request.url, stage="started",
        request_body=request.content, request_headers=...,
    )
    verdict = await hook_governance.evaluate_async(span, span_data=payload)
    
    if verdict == Verdict.BLOCK:
        raise GovernanceBlockedError("block", "URL blocked", request.url)
    
    # ALLOW: execute request
    response = await call_next(request)
    
    # Stage 2: completed (informational, can't block)
    payload = _build_http_span_data(
        span, ..., stage="completed", response_body=response.content, ...
    )
    await hook_governance.evaluate_async(...)  # fire-and-forget
    
    return response
```

**Body Capture Strategy:**
- **Problem:** OTel httpx instrumentation consumes the request.stream
- **Solution:** Patch `httpx.Client.send()` to capture `request.content` BEFORE instrumentation consumes the stream
- **Result:** Hooks see complete request/response bodies for governance evaluation

#### Database Hooks (`db_governance_hooks.py`, 874 LOC)

**Supported Libraries:**
- **SQLAlchemy** (all engines) — event listener on `before_execute`
- **asyncpg** (async PostgreSQL) — wrapt wrapper
- **psycopg2** (sync PostgreSQL) — CursorTracer patch
- **pymongo** (MongoDB) — CommandListener
- **redis** — native hook
- **MySQL, SQLite** — OTel auto-instrumentation

**Hook Pattern (example: asyncpg):**

```python
# Monkey-patch asyncpg.connection.execute with wrapt
@wrapt.decorator
async def patched_execute(wrapped, instance, args, kwargs):
    span = trace.get_current_span()
    query = args[0] if args else kwargs.get("query", "")
    
    # Stage 1: started (BEFORE query)
    payload = _build_db_span_data(
        span, db_system="postgresql", db_statement=query, stage="started"
    )
    verdict = await hook_governance.evaluate_async(span, span_data=payload)
    
    if verdict == Verdict.BLOCK:
        raise GovernanceBlockedError("block", "Query blocked", query)
    
    # ALLOW: execute
    result = await wrapped(*args, **kwargs)
    
    # Stage 2: completed (metadata only)
    payload = _build_db_span_data(
        span, ..., stage="completed", rowcount=len(result), ...
    )
    await hook_governance.evaluate_async(...)
    
    return result
```

#### File I/O Hooks (`file_governance_hooks.py`, 407 LOC)

**Mechanism:** Patch `builtins.open()` and `os.fdopen()`

**Hook Pattern:**

```python
def patched_open(path, mode="r", *args, **kwargs):
    span = trace.get_current_span()
    
    # Stage 1: started (BEFORE open)
    payload = _build_file_span_data(
        span, file_path=path, file_mode=mode, operation="open", stage="started"
    )
    verdict = hook_governance.evaluate_sync(span, span_data=payload)  # Sync
    
    if verdict == Verdict.BLOCK:
        raise GovernanceBlockedError("block", "File access denied", path)
    
    # ALLOW: open file (wrap with TracedFile)
    file_obj = TracedFile(original_open(path, mode, *args, **kwargs))
    return file_obj
```

**Path Resolution:**
- Regular paths: use `pathlib.Path`
- File descriptors (os.fdopen): use `os.readlink(f"/proc/self/fd/{fd}")` on Linux, `fcntl` on macOS/BSD

### Layer 3: Activity Context Mapping

**File:** `span_processor.py` (244 LOC) + `otel_setup.py` (481 LOC)

**Problem:** LangGraph spawns asyncio.Task for tool execution with NEW trace context. Hooks fire with different trace_id than the activity that spawned them.

**Solution:** WorkflowSpanProcessor explicitly maps trace_id → (workflow_id, activity_id)

**Registration Flow:**

```
on_tool_start event
    ↓
Create OTel span with trace_id="trace-123"
    ↓
SpanProcessor.register_trace("trace-123", workflow_id="wf-0", activity_id="act-1")
    ↓
Tool executes in Task with trace context
    ↓
HTTP hook fires (trace_id in context = "trace-123")
    ↓
hook_governance.evaluate_async()
    ├─ extract_trace_id_from_context() → "trace-123"
    ├─ SpanProcessor.get_activity_context_by_trace("trace-123")
    │  → Returns: (workflow_id="wf-0", activity_id="act-1")
    └─ Build hook payload with activity metadata
```

**Fallback Strategies (if trace_id not found):**

1. **Single-Activity Mode:** If only one activity in-flight → assume hook belongs to it
2. **Most-Recent Mode:** If multiple activities → use most recently started
3. **Sync Fallback:** For sync hooks (file I/O), store activity context in ContextVar

---

## 3. Configuration Model

### GovernanceConfig (Complete Parameter List)

```python
@dataclass
class GovernanceConfig:
    # Error Handling
    on_api_error: str = "fail_open"  # | "fail_closed"
    # - fail_open: Network error → return None → continue (optimistic)
    # - fail_closed: Network error → raise OpenBoxNetworkError → block (safe)
    
    api_timeout: float = 30.0  # seconds (auto-converted from ms if > 600)
    
    # Event Filtering (what to send to Core)
    send_chain_start_event: bool = True    # on_chain_start
    send_chain_end_event: bool = True      # on_chain_end (observation only)
    send_tool_start_event: bool = True     # on_tool_start
    send_tool_end_event: bool = True       # on_tool_end (observation only)
    send_llm_start_event: bool = True      # on_chat_model_start (guardrails)
    send_llm_end_event: bool = True        # on_chat_model_end (observation)
    
    # Skip Entire Tool/Chain Types
    skip_chain_types: set[str] = field(default_factory=set)
    # Example: {"ReActAgent", "SomeChain"} — never send governance events
    
    skip_tool_types: set[str] = field(default_factory=set)
    # Example: {"_check_tool_type", "internal_tool"} — never send governance events
    
    # HITL Configuration
    hitl: HITLConfig = field(default_factory=HITLConfig)
    # Fields: enabled (bool), poll_interval_ms (int), skip_tool_types (set)
    
    # Metadata
    session_id: str | None = None           # Optional session identifier
    agent_name: str | None = None           # "MyAgent" — sent in every event
    task_queue: str = "langgraph"           # Workflow queue name
    use_native_interrupt: bool = False      # Reserved for LangGraph native flow control
    
    # Root Node Classification
    root_node_names: set[str] = field(default_factory=set)
    # Example: {"start", "router"} — marks the outermost invocation
    
    # Tool Type Classification (for execution tree)
    tool_type_map: dict[str, str] = field(default_factory=dict)
    # Example: {
    #     "search_web": "http",          # HTTP tool
    #     "query_db": "database",        # Database tool
    #     "write_file": "builtin",       # Built-in
    #     "call_agent": "a2a",           # Agent-to-agent
    # }
    # Supported types: "http", "database", "builtin", "a2a", "custom"
    # Used by Core to build execution tree visualization
```

### Initialization & Global State

```python
# Synchronous initialization (safe to call at module level)
def initialize(
    api_url: str,                  # Required
    api_key: str,                  # Required (obx_live_* | obx_test_*)
    governance_timeout: float = 30.0,
    validate: bool = True,         # Ping /api/v1/auth/validate
) -> None:
    # 1. validate_url_security() — error if HTTP on non-localhost
    # 2. validate_api_key_format() — error if not obx_live_*/obx_test_*
    # 3. _global_config.configure(api_url, api_key, timeout)
    # 4. If validate=True: _validate_api_key_with_server() (sync via urllib)

# Global configuration singleton
_global_config: _GlobalConfigState = _GlobalConfigState()

def get_global_config() -> _GlobalConfigState:
    # Retrieve singleton (used by create_openbox_graph_handler)
    return _global_config
```

### Configuration Merging

```python
def merge_config(partial: dict[str, Any] | None = None) -> GovernanceConfig:
    # Start with DEFAULT_GOVERNANCE_CONFIG
    # Override with partial dict values
    # Convert api_timeout: if <= 600 treat as seconds, else milliseconds
    # Convert hitl dict → HITLConfig instance
    # Convert skip_*_types: accept list, tuple, or set
    # Return fully typed GovernanceConfig
```

---

## 4. Event Flow & Lifecycle

### WorkflowStarted → ActivityStarted → Spans → Verdicts → ActivityCompleted → WorkflowCompleted

#### Complete Event Sequence (Example: tool + HTTP call)

```
┌─────────────────────────────────────────────────────────────────────┐
│ ainvoke({"messages": [...]})                                        │
└────────────────┬────────────────────────────────────────────────────┘
                 │
        1. PRE-SCREEN PHASE
                 │
        ┌────────▼────────┐
        │ SignalReceived   │ (extract user prompt)
        │ event_type: "SignalReceived"
        │ signal_name: "user_prompt"
        │ signal_args: [user_text]
        └────────┬────────┘
                 │
        ┌────────▼────────────────┐
        │ WorkflowStarted          │
        │ event_type: "ChainStarted" (mapped to WorkflowStarted)
        │ workflow_id: "sess-abc-12345678"
        │ run_id: "sess-abc-run-87654321"
        │ activity_input: [user_input_dict]
        │ timestamp: RFC3339
        └────────┬────────────────┘
                 │ Evaluate with Core
                 │
        ┌────────▼──────────────────────────────┐
        │ _pre_screen_input() checks verdict    │
        │ • If BLOCK/HALT: raise exception ✗   │
        │ • If REQUIRE_APPROVAL: poll HITL      │
        │ • If ALLOW: continue ✓                │
        │ • Guardrails: if validation fails ✗  │
        └────────┬──────────────────────────────┘
                 │ (only if pass)
        ┌────────▼────────────────┐
        │ LLMStarted (pre-screen)  │
        │ activity_id: "run_id-pre"
        │ activity_type: "llm_call"
        │ prompt: extracted_user_text
        │ (for input guardrails)
        └────────┬────────────────┘
                 │ Evaluate
                 │ • Guardrails: redaction response if guardrails_result present
        │ • PII redaction: _GuardrailsCallbackHandler stores for later use
        └────────┬────────────────┘
                 │
        2. STREAM EVENTS PHASE
                 │
    ┌────────────▼──────────────┐
    │ on_chain_start (root)     │
    │ _map_event() extracts:    │
    │  - run_id (LangGraph)     │
    │  - name (chain name)      │
    │  - input (state dict)     │
    ├───────────┬───────────────┤
    │ SKIP?     │ (if root      │
    │ Sent by   │  already in   │
    │ pre-      │  pre-screen)  │
    │ screen    │               │
    └───────────┬───────────────┘
                 │
    ┌────────────▼──────────────────────────┐
    │ on_tool_start (e.g., "search_web")   │
    │ event: on_tool_start                 │
    │ name: "search_web"                   │
    │ run_id: uuid (LangChain callback)    │
    ├─────────────────────────────────────┤
    │ _map_event() extracts:               │
    │  - activity_id = run_id (UUID)      │
    │  - activity_type = "tool"           │
    │  - activity_input = [tool_input]    │
    │  - tool_name = "search_web"         │
    │  - tool_type = "http" (from map)    │
    └─────────────┬──────────────────────┘
                  │
    ┌─────────────▼──────────────────────────┐
    │ ToolStarted event sent to Core        │
    │ Evaluate with Core's policies         │
    ├──────────────────────────────────────┤
    │ Verdict options:                     │
    │ • ALLOW: continue                    │
    │ • BLOCK: raise GovernanceBlockedError│
    │ • HALT: raise GovernanceHaltError    │
    │ • REQUIRE_APPROVAL: poll HITL        │
    │ • CONSTRAIN: warn & continue         │
    └─────────────┬──────────────────────────┘
                  │ (if ALLOW)
    ┌─────────────▼──────────────┐
    │ OTel Span created         │
    │ trace_id = "trace-abc"    │
    │ span_name = "search_web"  │
    ├──────────────────────────┤
    │ SpanProcessor.register_  │
    │ trace("trace-abc" →      │
    │  (workflow_id, activity_ │
    │   id="uuid-tool"))       │
    └─────────────┬──────────────┘
                  │
    ┌─────────────▼──────────────────────┐
    │ Tool executes (search_web)        │
    │ Makes HTTP request:                │
    │ POST https://api.example.com/...   │
    └─────────────┬──────────────────────┘
                  │
    3. HOOK-LEVEL GOVERNANCE
                  │
    ┌─────────────▼──────────────────────────┐
    │ HTTP Hook fires (httpx_send)           │
    │ Stage: "started" (BEFORE request)      │
    ├──────────────────────────────────────┤
    │ Extract span context:                 │
    │  - span_id_hex = format(trace_id)    │
    │  - trace_id_hex = format(trace_id)   │
    │  - parent_span_id = None             │
    ├──────────────────────────────────────┤
    │ Build HTTP span_data:                │
    │  - span_id, trace_id, name           │
    │  - http_method = "POST"              │
    │  - http_url = request.url            │
    │  - request_body = request.content    │
    │  - request_headers = {...}           │
    │  - stage = "started"                 │
    └─────────────┬──────────────────────────┘
                  │
    ┌─────────────▼──────────────────────────┐
    │ hook_governance.evaluate_async()       │
    ├──────────────────────────────────────┤
    │ 1. Extract trace_id from span context│
    │ 2. SpanProcessor.get_activity_       │
    │    context_by_trace(trace_id)        │
    │    → Returns (workflow_id,           │
    │       activity_id="uuid-tool")       │
    │ 3. Build hook payload with activity  │
    │ 4. GovernanceClient.evaluate_raw()   │
    │    POST /api/v1/governance/evaluate  │
    │ 5. Parse response verdict            │
    └─────────────┬──────────────────────────┘
                  │
    ┌─────────────▼──────────────────────────┐
    │ Verdict from Hook evaluation          │
    │ • ALLOW: continue ✓                   │
    │ • BLOCK: raise GovernanceBlockedError │
    │   (bubbles through HTTP client → LLM  │
    │    SDK → handler catches & unwraps)   │
    └─────────────┬──────────────────────────┘
                  │
    ┌─────────────▼──────────────┐
    │ HTTP request proceeds      │
    │ Response received          │
    └─────────────┬──────────────┘
                  │
    ┌─────────────▼──────────────────────────┐
    │ HTTP Hook fires again                 │
    │ Stage: "completed" (AFTER response)   │
    │ Informational only (can't block)      │
    │ hook_governance.evaluate_async()      │
    │ (fire-and-forget)                     │
    └─────────────┬──────────────────────────┘
                  │
    ┌─────────────▼──────────────┐
    │ on_tool_end event          │
    │ _map_event() → ToolCompleted
    │ activity_output = result   │
    │ duration_ms = elapsed time │
    └─────────────┬──────────────┘
                  │ (observation only, no verdict)
    ┌─────────────▼──────────────┐
    │ on_chat_model_start        │
    │ LangGraph calls the LLM   │
    ├──────────────────────────┤
    │ _GuardrailsCallbackHandler│
    │ .on_chat_model_start():   │
    │ 1. Extract prompt from    │
    │    messages (human turns) │
    │ 2. Send LLMStarted event  │
    │ 3. If guardrails_result:  │
    │    mutate messages in-place│
    │    (PII redaction)         │
    └─────────────┬──────────────┘
                  │
    ┌─────────────▼──────────────┐
    │ LLM call executes          │
    │ (sees redacted input)      │
    └─────────────┬──────────────┘
                  │
    ┌─────────────▼──────────────┐
    │ on_chat_model_end          │
    │ _map_event() → LLMCompleted
    │ activity_output = response │
    │ token_counts               │
    │ model_name, has_tool_calls │
    └─────────────┬──────────────┘
                  │
    ┌─────────────▼──────────────────────────┐
    │ (more on_tool_start/end, etc.)        │
    └─────────────┬──────────────────────────┘
                  │
    ┌─────────────▼──────────────┐
    │ on_chain_end (root)        │
    │ _map_event() → WorkflowCompleted
    │ workflow_output = final    │
    │ state                      │
    └─────────────┬──────────────┘
                  │
    ┌─────────────▼──────────────┐
    │ Return final_output        │
    │ to caller                  │
    └─────────────────────────────┘
```

---

## 5. Verdict Handling: ALLOW/BLOCK/HALT/REQUIRE_APPROVAL

### Verdict Enum & Priority

```python
class Verdict(StrEnum):
    ALLOW = "allow"                    # Priority 0
    CONSTRAIN = "constrain"           # Priority 1
    REQUIRE_APPROVAL = "require_approval"  # Priority 2
    BLOCK = "block"                   # Priority 3
    HALT = "halt"                     # Priority 4 (highest)
```

**Aggregation:** When multiple hooks/events produce verdicts, **highest priority wins**.

### Enforcement Context Hierarchy

```python
VerdictContext = Literal[
    "chain_start",        # WorkflowStarted or on_chain_start → enforced
    "chain_end",          # WorkflowCompleted → OBSERVATION ONLY
    "tool_start",         # ToolStarted → enforced
    "tool_end",           # ToolCompleted → enforced (Behavior Rules)
    "llm_start",          # LLMStarted → enforced
    "llm_end",            # LLMCompleted → OBSERVATION ONLY
    "agent_action",       # AgentAction → enforced
    "agent_finish",       # AgentFinish → OBSERVATION ONLY
    "graph_node_start",   # on_chain_start (non-root) → enforced
    "graph_node_end",     # on_chain_end (non-root) → OBSERVATION ONLY
    "graph_root_start",   # on_chain_start (root) → enforced
    "graph_root_end",     # on_chain_end (root) → OBSERVATION ONLY
    "other",              # unknown → OBSERVATION ONLY
]
```

**Key:** Only START and TOOL/LLM END contexts enforce verdicts. Root/chain END events are observation-only (dashboards, logs, no enforcement).

### Enforcement Algorithm

```python
def enforce_verdict(
    response: GovernanceVerdictResponse,
    context: VerdictContext,
) -> VerdictEnforcementResult:
    """
    Raises exception or signals HITL requirement.
    Returns VerdictEnforcementResult(requires_hitl, blocked).
    """
    
    # 1. OBSERVATION-ONLY CONTEXTS → no enforcement
    if context in ("chain_end", "graph_root_end", "agent_finish", "other"):
        return VerdictEnforcementResult()
    
    verdict = response.verdict
    reason = response.reason
    policy_id = response.policy_id
    risk_score = response.risk_score
    
    # 2. HALT → highest priority, immediate exception
    if verdict == Verdict.HALT:
        raise GovernanceHaltError(
            reason or "Workflow halted by governance policy",
            policy_id=policy_id,
            risk_score=risk_score,
        )
    
    # 3. BLOCK → reject action
    if verdict == Verdict.BLOCK:
        raise GovernanceBlockedError(
            reason or "Action blocked by governance policy",
            policy_id,
            risk_score,
        )
    
    # 4. Guardrails validation → block if validation_passed == False
    if response.guardrails_result and not response.guardrails_result.validation_passed:
        gr = response.guardrails_result
        if gr.input_type == "activity_output":
            reasons = _get_guardrail_output_failure_reasons(gr.reasons)
        else:
            reasons = _get_guardrail_failure_reasons(gr.reasons)
        raise GuardrailsValidationError(reasons)
    
    # 5. REQUIRE_APPROVAL → poll HITL if applicable, else block
    if verdict == Verdict.REQUIRE_APPROVAL:
        if is_hitl_applicable(context):
            return VerdictEnforcementResult(requires_hitl=True)
        else:
            raise GovernanceBlockedError(
                reason or "Action requires approval but cannot be paused at this stage",
                policy_id,
                risk_score,
            )
    
    # 6. CONSTRAIN → warn and continue
    if verdict == Verdict.CONSTRAIN and reason:
        suffix = f" (policy: {policy_id})" if policy_id else ""
        warnings.warn(f"[OpenBox] Governance constraint: {reason}{suffix}", stacklevel=2)
        return VerdictEnforcementResult()
    
    # 7. ALLOW → continue
    return VerdictEnforcementResult()
```

### HITL Applicability

```python
def is_hitl_applicable(context: VerdictContext) -> bool:
    """Return True if HITL polling can pause execution at this context."""
    return context in (
        "tool_start",          # Pre-execution pause
        "tool_end",            # Post-execution pause (Behavior Rules)
        "llm_start",           # Pre-execution pause
        "llm_end",             # Post-execution pause
        "graph_node_start",    # Node-level pause
        "agent_action",        # ReAct-specific
    )
```

---

## 6. Hook Governance: Detailed Mechanics

### HTTP Governance Hooks

**Capture Strategy:**

```python
# Problem: OTel httpx instrumentation hooks consume the request.stream
# Solution: Patch httpx.Client.send to capture BEFORE instrumentation

class _patched_httpx_send:
    """Wraps httpx.Client.send to capture request.content."""
    
    async def __call__(self, client, request, stream=False, auth=None, follow_redirects=None, allow_redirects=None, cert=None, verify=None, proxies=None, timeout=None, **kwargs):
        # Capture request.content BEFORE instrumentation consumes stream
        try:
            request_body = request.content.decode("utf-8", errors="replace")
        except Exception:
            request_body = None
        
        # Store in ContextVar for the hook to retrieve
        _httpx_request_bodies.set((id(request), request_body))
        
        # Call original send (will be wrapped by OTel)
        response = await original_send(request, stream, auth, ...)
        
        # Capture response body
        try:
            response_body = response.content.decode("utf-8", errors="replace")
        except Exception:
            response_body = None
        
        return response
```

**Hook Stages:**

| Stage | Can Block? | Data Available | Use Case |
|-------|-----------|-----------------|----------|
| started | Yes | Request only | Blocking policy (URL allowlist, etc.) |
| completed | No | Request + response | Logging, analysis, extraction |

### Database Governance Hooks

**Library-Specific Patterns:**

| Library | Hook Point | Mechanism |
|---------|-----------|-----------|
| **SQLAlchemy** | `before_execute` event | Event listener on engine |
| **asyncpg** | `connection.execute()` | wrapt.decorator |
| **psycopg2** | `cursor.execute()` | OTel CursorTracer patch |
| **pymongo** | `CommandListener` | before_command_started |
| **redis** | Native hook | `response_hook` registration |
| **sqlite3** | OTel instrumentation | Auto-instrumented |

**Example: SQLAlchemy Hook**

```python
# In otel_setup.py
from sqlalchemy import event

def setup_sqlalchemy_hooks(engine, span_processor):
    @event.listens_for(engine, "before_cursor_execute")
    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        span = trace.get_current_span()
        
        # Stage: started
        payload = _build_db_span_data(
            span,
            db_system="sqlite" if "sqlite" in str(engine.url) else "postgresql",
            db_statement=statement,
            stage="started",
        )
        verdict = hook_governance.evaluate_sync(span, identifier=statement[:50], span_data=payload)
        
        if verdict.should_stop():
            raise GovernanceBlockedError("block", f"Query blocked: {statement[:50]}", statement)
    
    @event.listens_for(engine, "after_cursor_execute")
    def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        # Stage: completed (informational)
        span = trace.get_current_span()
        payload = _build_db_span_data(
            span,
            db_system=...,
            db_statement=statement,
            stage="completed",
            rowcount=cursor.rowcount,
        )
        hook_governance.evaluate_sync(span, span_data=payload)  # fire-and-forget
```

### Payload Building

**Unified Hook Payload Structure:**

```python
def _build_payload(
    span,                    # OTel Span
    span_data: dict,         # Hook-specific data
) -> dict[str, Any]:
    """
    Builds complete governance payload from span context + span_data.
    """
    span_id_hex, trace_id_hex, parent_span_id = extract_span_context(span)
    
    # Get activity context from SpanProcessor
    activity = span_processor.get_activity_context_by_trace(int(trace_id_hex, 16))
    
    if activity is None:
        # Fallback to single-activity or most-recent
        activity = span_processor.get_last_activity_context()
    
    payload = {
        # Activity identifiers
        "workflow_id": activity.workflow_id if activity else "unknown",
        "run_id": activity.run_id if activity else "unknown",
        "activity_id": activity.activity_id if activity else f"hook-{span_id_hex}",
        
        # Span details
        "span_id": span_id_hex,
        "trace_id": trace_id_hex,
        "parent_span_id": parent_span_id,
        
        # Hook type (merged from span_data)
        **span_data,
        
        # Signal to Rego policies that this is a hook (not stream event)
        "hook_trigger": True,
    }
    
    return payload
```

---

## 7. HITL: Human-in-the-Loop Approval Polling

### Polling Loop

**File:** `hitl.py` (88 LOC)

```python
async def poll_until_decision(
    client: GovernanceClient,
    params: HITLPollParams,
    config: HITLConfig,
) -> None:
    """Block until governance approves or rejects.
    
    Polls OpenBox Core indefinitely until a terminal verdict is received.
    The server controls approval expiration — the SDK does not impose a deadline.
    
    Resolves (returns None) when approval is granted (ALLOW verdict).
    Raises on rejection, expiry, or HALT/BLOCK verdict.
    """
    
    while True:
        # Sleep before first poll (give user time to see dashboard)
        await asyncio.sleep(config.poll_interval_ms / 1000.0)
        
        response = await client.poll_approval(
            ApprovalPollParams(
                workflow_id=params.workflow_id,
                run_id=params.run_id,
                activity_id=params.activity_id,
            )
        )
        
        if response is None:
            # API unreachable → keep polling (fail-open for HITL)
            continue
        
        if response.expired:
            raise ApprovalExpiredError(
                f"Approval expired for {params.activity_type} "
                f"(activity_id={params.activity_id})"
            )
        
        verdict = response.verdict
        
        if verdict == Verdict.ALLOW:
            return  # Approved
        
        if verdict_should_stop(verdict):  # HALT or BLOCK
            raise ApprovalRejectedError(
                verdict.reason or f"Approval rejected for {params.activity_type}"
            )
        
        # Still pending (REQUIRE_APPROVAL or CONSTRAIN) → keep polling
```

### Client API

```python
# In client.py
async def poll_approval(
    self,
    params: ApprovalPollParams,
) -> ApprovalResponse | None:
    """Poll the HITL queue for a decision.
    
    Called repeatedly by poll_until_decision() until a terminal verdict.
    
    Returns:
        ApprovalResponse(verdict, reason, expired)
        or None if network error (when fail_open).
    """
    try:
        response = await self._get_client().post(
            f"{self._api_url}/api/v1/governance/approval",
            headers=self._headers(),
            json={
                "workflow_id": params.workflow_id,
                "run_id": params.run_id,
                "activity_id": params.activity_id,
            },
        )
        
        if not response.is_success:
            if self._on_api_error == "fail_closed":
                raise OpenBoxNetworkError(...)
            return None  # fail_open
        
        data = response.json()
        return parse_approval_response(data)
    
    except OpenBoxNetworkError:
        raise
    except Exception as e:
        if self._on_api_error == "fail_closed":
            raise OpenBoxNetworkError(...) from e
        return None  # fail_open
```

### Integration with Handler

```python
# In langgraph_handler.py

async def ainvoke(...):
    ...
    try:
        async for event in self._graph.astream_events(...):
            await self._process_event(...)
    except GovernanceBlockedError as hook_err:
        if hook_err.verdict != "require_approval":
            raise
        
        # HITL Required — poll for decision
        _logger.info("[OpenBox] Polling HITL for approval...")
        await poll_until_decision(
            self._client,
            HITLPollParams(
                workflow_id=workflow_id,
                run_id=run_id,
                activity_id=activity_id,  # The activity that needs approval
                activity_type="http_hook",  # or "tool", "llm", etc.
            ),
            self._config.hitl,
        )
        
        # Approved — retry
        _logger.info("[OpenBox] Approval granted, retrying ainvoke...")
        final_output = await self._graph.ainvoke(input, config=cfg, **kwargs)
```

---

## 8. Client API: OpenBox Core Integration

### Endpoints

**File:** `client.py` (329 LOC)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/auth/validate` | GET | Validate API key on startup |
| `/api/v1/governance/evaluate` | POST | Send governance event, get verdict |
| `/api/v1/governance/approval` | POST | Poll HITL queue for approval decision |

### GovernanceClient

```python
class GovernanceClient:
    """Async HTTP client for OpenBox Core governance API."""
    
    def __init__(
        self,
        *,
        api_url: str,                  # Required
        api_key: str,                  # Required
        timeout: float = 30.0,         # seconds
        on_api_error: str = "fail_open",  # | "fail_closed"
    ):
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._on_api_error = on_api_error
        # Persistent clients (lazy-init)
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None
        # Dedup tracking (per workflow_id, run_id)
        self._dedup_run: tuple[str, str] | None = None
        self._dedup_sent: set[tuple[str, str]] = set()
    
    # ─────────────────────────────────────────────
    # Sync vs Async
    # ─────────────────────────────────────────────
    
    async def evaluate_event(
        self,
        event: LangChainGovernanceEvent,
    ) -> GovernanceVerdictResponse | None:
        """Async version — used by event stream & callbacks."""
        ...
    
    def evaluate_event_sync(
        self,
        event: LangChainGovernanceEvent,
    ) -> GovernanceVerdictResponse | None:
        """Sync version — used by sync middleware hooks (file I/O)."""
        # Uses self._get_sync_client() instead
        ...
    
    async def evaluate_raw(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """For pre-built hook payloads (no dedup, no mapping)."""
        ...
    
    async def poll_approval(
        self,
        params: ApprovalPollParams,
    ) -> ApprovalResponse | None:
        """Poll HITL queue."""
        ...
    
    async def validate_api_key(self) -> None:
        """Test connection on init (called by initialize())."""
        ...
```

### Request/Response Format

```python
# Request: governance/evaluate
{
    "source": "workflow-telemetry",
    "event_type": "WorkflowStarted" | "ToolStarted" | "LLMStarted" | ...,
    "workflow_id": "sess-abc-12345678",
    "run_id": "sess-abc-run-87654321",
    "workflow_type": "MyAgent",
    "task_queue": "langgraph",
    "timestamp": "2026-03-30T14:30:00.000Z",
    
    # Activity fields
    "activity_id": "uuid-tool-1",
    "activity_type": "tool" | "llm_call" | "chain",
    "activity_input": [tool_input_dict],
    "activity_output": tool_result_dict,
    
    # Optional
    "langgraph_node": "search_web",
    "langgraph_step": 2,
    "subagent_name": "SubAgent",
    "hook_trigger": False,  # True if from HTTP/DB/file hook
    
    # LLM-specific
    "llm_model": "gpt-4",
    "prompt": "user message text",
    "input_tokens": 150,
    "output_tokens": 200,
    "total_tokens": 350,
    
    # Hook-specific (if hook_trigger=True)
    "http_method": "POST",
    "http_url": "https://api.example.com/...",
    "request_body": "...",
    "response_body": "...",
    "db_statement": "SELECT * FROM users WHERE id = $1",
    "db_system": "postgresql",
    ...
}

# Response: governance/evaluate
{
    "verdict": "allow" | "constrain" | "require_approval" | "block" | "halt",
    "reason": "optional explanation",
    "policy_id": "policy-123",
    "risk_score": 0.75,
    "approval_id": "apr-xyz",                    # if require_approval
    "approval_expiration_time": "2026-03-30T14:35:00Z",
    "guardrails_result": {
        "input_type": "activity_input" | "activity_output",
        "validation_passed": False,
        "redacted_input": {"prompt": "..."},     # if validation_passed=False
        "reasons": [
            {"type": "pii", "field": "email", "reason": "Email detected"},
            ...
        ],
    },
    "constraints": [...],
    ...
}

# Request: governance/approval (HITL poll)
{
    "workflow_id": "sess-abc-12345678",
    "run_id": "sess-abc-run-87654321",
    "activity_id": "uuid-tool-1",
}

# Response: governance/approval
{
    "verdict": "allow" | "block" | "halt",
    "reason": "User approved",
    "expired": False,
    "approval_expiration_time": "2026-03-30T14:35:00Z",
}
```

### Deduplication

```python
def _is_duplicate(
    self,
    workflow_id: str,
    run_id: str,
    activity_id: str,
    event_type: str,
) -> bool:
    """Return True if this (activity_id, event_type) was already sent in this run.
    
    Hook events (evaluate_raw) are NOT checked — multiple hooks per activity
    are expected and valid.
    
    Resets automatically when (workflow_id, run_id) changes.
    """
    current_run = (workflow_id, run_id)
    
    # Reset dedup state on new run
    if self._dedup_run != current_run:
        self._dedup_run = current_run
        self._dedup_sent = set()
    
    # Check if duplicate
    key = (activity_id, event_type)
    if key in self._dedup_sent:
        return True  # Duplicate, skip
    
    self._dedup_sent.add(key)
    return False  # New, send it
```

---

## 9. Tracing: OpenTelemetry Setup & @traced Decorator

### OTel Instrumentation (`otel_setup.py`, 481 LOC)

```python
def setup_opentelemetry_for_governance(
    span_processor: WorkflowSpanProcessor,
    api_url: str,
    api_key: str,
    *,
    ignored_urls: list | None = None,
    instrument_databases: bool = True,
    db_libraries: set[str] | None = None,
    instrument_file_io: bool = False,
    sqlalchemy_engine: Any | None = None,
    api_timeout: float = 30.0,
    on_api_error: str = "fail_open",
) -> None:
    """
    Initialize OTel instrumentation for HTTP, database, and file I/O.
    
    This is called automatically by create_openbox_graph_handler(), but can
    also be called manually for more control.
    """
    
    # 1. Register WorkflowSpanProcessor with OTel provider
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(span_processor)
    
    # 2. Configure hook governance (sets up persistent HTTP clients)
    hook_governance.configure(
        api_url, api_key, span_processor,
        api_timeout=api_timeout, on_api_error=on_api_error,
    )
    
    # 3. Instrument HTTP libraries
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        RequestsInstrumentor().instrument(
            request_hook=_requests_request_hook,
            response_hook=_requests_response_hook,
        )
    except ImportError:
        pass
    
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument(
            request_hook=_httpx_request_hook,
            response_hook=_httpx_response_hook,
            async_request_hook=_httpx_async_request_hook,
            async_response_hook=_httpx_async_response_hook,
        )
    except ImportError:
        pass
    
    # 4. Instrument databases (conditional on db_libraries)
    if instrument_databases:
        db_gov.instrument_sqlalchemy(sqlalchemy_engine)
        db_gov.instrument_asyncpg()
        db_gov.instrument_psycopg2()
        db_gov.instrument_pymongo()
        db_gov.instrument_redis()
        db_gov.instrument_sqlite3()
        db_gov.instrument_mysql()
    
    # 5. Instrument file I/O (optional)
    if instrument_file_io:
        file_governance_hooks.setup_file_io_instrumentation()
```

### @traced Decorator (`tracing.py`, 320 LOC)

```python
@traced(
    name: str | None = None,                # Custom span name
    capture_args: bool = True,
    capture_result: bool = True,
    capture_exception: bool = True,
    max_arg_length: int = 2000,
)
def my_function(...):
    ...

# Usage:
@traced
def extract_data(query):
    return db.fetch(query)

@traced(name="custom-span", capture_result=False)
async def process_sensitive_data(data):
    return handle(data)
```

**Implementation:**

```python
def traced(
    _func: F | None = None,
    *,
    name: str | None = None,
    capture_args: bool = True,
    capture_result: bool = True,
    capture_exception: bool = True,
    max_arg_length: int = 2000,
) -> F | Callable[[F], F]:
    """Decorator to trace function calls as OpenTelemetry spans."""
    
    def decorator(func: F) -> F:
        span_name = name or func.__name__
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                tracer = _get_tracer()
                with tracer.start_as_current_span(span_name) as span:
                    # 1. Set metadata attributes
                    span.set_attribute("code.function", func.__name__)
                    span.set_attribute("code.namespace", func.__module__)
                    
                    # 2. Capture arguments
                    if capture_args:
                        for i, arg in enumerate(args):
                            span.set_attribute(
                                f"function.arg.{i}",
                                _safe_serialize(arg, max_arg_length),
                            )
                        for key, value in kwargs.items():
                            span.set_attribute(
                                f"function.kwarg.{key}",
                                _safe_serialize(value, max_arg_length),
                            )
                    
                    # 3. Governance: started stage (if hooks configured)
                    if hook_governance.is_configured():
                        started_sd = _build_traced_span_data(
                            span, func.__name__, func.__module__, "started",
                            args=_safe_serialize({"args": args, "kwargs": kwargs}),
                        )
                        await hook_governance.evaluate_async(
                            span, identifier=func.__name__, span_data=started_sd
                        )
                    
                    # 4. Execute function
                    _start = _time.perf_counter()
                    try:
                        result = await func(*args, **kwargs)
                        _dur_ms = (_time.perf_counter() - _start) * 1000
                        
                        # Capture result
                        if capture_result:
                            span.set_attribute(
                                "function.result",
                                _safe_serialize(result, max_arg_length),
                            )
                        
                        # Governance: completed stage
                        if hook_governance.is_configured():
                            completed_sd = _build_traced_span_data(
                                span, func.__name__, func.__module__, "completed",
                                duration_ms=_dur_ms,
                                result=_safe_serialize(result, max_arg_length) if capture_result else None,
                            )
                            await hook_governance.evaluate_async(
                                span, identifier=func.__name__, span_data=completed_sd
                            )
                        
                        return result
                    
                    except Exception as e:
                        if capture_exception:
                            span.set_attribute("error", True)
                            span.set_attribute("error.type", type(e).__name__)
                            span.set_attribute("error.message", str(e))
                        
                        # Governance: error stage
                        if hook_governance.is_configured():
                            error_sd = _build_traced_span_data(
                                span, func.__name__, func.__module__, "completed",
                                error=str(e),
                            )
                            await hook_governance.evaluate_async(
                                span, identifier=func.__name__, span_data=error_sd
                            )
                        
                        raise
            
            return async_wrapper  # type: ignore
        
        else:
            # Synchronous version (similar pattern)
            ...
    
    # Handle both @traced and @traced() syntax
    if _func is not None:
        return decorator(_func)
    return decorator
```

### Span Data Format

```python
# For @traced functions
{
    "span_id": "0123456789abcdef",
    "trace_id": "0123456789abcdef0123456789abcdef",
    "parent_span_id": "fedcba9876543210",
    "name": "extract_data",
    "kind": "INTERNAL",
    "stage": "started" | "completed",
    "start_time": 1711875000000000000,  # nanoseconds
    "end_time": 1711875000100000000,
    "duration_ns": 100000000,
    "attributes": {...},  # OTel span attributes
    "status": {"code": "UNSET" | "ERROR", "description": null | "error message"},
    "events": [],
    
    # Hook type identification
    "hook_type": "function_call",
    
    # Function-specific fields
    "function": "extract_data",
    "module": "my_package.data",
    "args": '{"args": [...], "kwargs": {...}}',
    "result": "...",
    "error": null | "error message",
}
```

---

## 10. Error Handling: Exception Types & Fail-Open/Fail-Closed

### Exception Hierarchy

```python
OpenBoxError (base Exception)
│
├─ OpenBoxAuthError
│  └─ Raised when API key is invalid or unauthorized
│     Context: initialize(), GovernanceClient.validate_api_key()
│
├─ OpenBoxNetworkError
│  └─ Raised when OpenBox Core is unreachable
│     Context: All API calls (when on_api_error="fail_closed")
│
├─ OpenBoxInsecureURLError
│  └─ Raised when HTTP is used on non-localhost URL
│     Context: initialize(), config validation
│
├─ GovernanceBlockedError(verdict, reason, policy_id, risk_score)
│  └─ Raised when verdict is BLOCK (or REQUIRE_APPROVAL on non-HITL context)
│     Attributes:
│       - verdict: "block" (str)
│       - identifier: URL, query, or path that triggered block
│       - policy_id: Policy that blocked
│       - risk_score: Risk score
│
├─ GovernanceHaltError(reason, policy_id, risk_score)
│  └─ Raised when verdict is HALT (workflow-level stop)
│     Attributes:
│       - verdict: "halt" (str constant)
│       - policy_id, risk_score
│
├─ GuardrailsValidationError(reasons: list[str])
│  └─ Raised when guardrails validation fails (input/output validation)
│     Attributes:
│       - reasons: list of human-readable failure reasons
│
├─ ApprovalExpiredError
│  └─ Raised when HITL approval window expired
│
├─ ApprovalRejectedError
│  └─ Raised when user explicitly rejects HITL approval
│
└─ ApprovalTimeoutError(max_wait_ms)
   └─ Raised when HITL polling exceeds max_wait_ms
       Attributes:
         - max_wait_ms: timeout in milliseconds
```

### Fail-Open vs Fail-Closed

**Configuration:**

```python
on_api_error: str = "fail_open"  # | "fail_closed"
```

**Behavior:**

| Error | fail_open | fail_closed |
|-------|-----------|-------------|
| Network timeout | Return None (skip eval) | Raise OpenBoxNetworkError |
| 5xx Server error | Return None | Raise OpenBoxNetworkError |
| 4xx Client error | Return None | Raise OpenBoxNetworkError |
| DNS resolution fail | Return None | Raise OpenBoxNetworkError |

**Semantics:**
- **fail_open:** Optimistic — assume ALLOW if Core unreachable (minimize disruption)
- **fail_closed:** Safe — block execution if Core unreachable (ensure governance)

**Use Cases:**
- **fail_open:** Production critical system (user experience priority)
- **fail_closed:** High-security system (compliance/audit priority)

### Exception Unwrapping

**Context:** LLM SDKs (OpenAI, Anthropic) wrap httpx errors as APIConnectionError.

```python
def _extract_governance_blocked(exc: Exception) -> GovernanceBlockedError | None:
    """Walk exception chain (__cause__ / __context__) to find wrapped GovernanceBlockedError.
    
    When an OTel hook raises GovernanceBlockedError inside httpx, the LLM SDK wraps it:
        GovernanceBlockedError
          ↓ (HTTPX catches)
        HTTPError
          ↓ (OpenAI SDK catches)
        APIConnectionError
        
    This function unwraps the chain to recover the original error.
    """
    cause: BaseException | None = exc
    seen: set[int] = set()
    
    while cause is not None:
        if id(cause) in seen:
            break  # Cycle detection
        seen.add(id(cause))
        
        if isinstance(cause, GovernanceBlockedError):
            return cause
        
        # Walk the chain
        cause = getattr(cause, '__cause__', None) or getattr(cause, '__context__', None)
    
    return None
```

---

## 11. Framework-Specific vs Reusable Components

### LangGraph-Specific Code

**Files:** `langgraph_handler.py`, `verdict_handler.py` (context classification)

**Components:**
- `OpenBoxLangGraphHandler` — LangGraph event stream wrapping
- `LangGraphStreamEvent` — v2 event parsing
- `_GuardrailsCallbackHandler` — LangChain callback for PII redaction
- `_RunBufferManager`, `_RootRunTracker` — LangGraph-specific lifecycle management
- `_map_event()` — v2 event → governance event conversion
- LangGraph routing logic (subagent detection, node naming, step numbering)

**Reusability:** Low — tightly coupled to LangGraph v2 event stream format.

### Reusable Components

**Files:** All other modules

| Module | Reusability | Can Use For |
|--------|------------|-----------|
| `config.py` | High | Any SDK initialization |
| `client.py` | High | Any governance client |
| `types.py` | High | Any event schema |
| `verdict_handler.py` | High | Any verdict enforcement |
| `hitl.py` | High | Any HITL polling |
| `otel_setup.py` | Medium | Any OTel instrumentation (with minor customization) |
| `hook_governance.py` | Medium | Any hook-level governance (HTTP, DB, file) |
| `http_governance_hooks.py` | High | Any HTTP governance (reusable for Temporal, FastAPI, etc.) |
| `db_governance_hooks.py` | High | Any DB governance (decoupled from LangGraph) |
| `file_governance_hooks.py` | High | Any file I/O governance (pure builtins patching) |
| `tracing.py` | High | Any @traced function instrumentation |
| `span_processor.py` | Medium | Any trace_id → activity mapping (with context refactor) |
| `errors.py` | High | Standard exception hierarchy |

### Candidates for SDK Extraction

```python
# Core governance layer (framework-agnostic)
├─ client.py
├─ config.py
├─ types.py
├─ verdict_handler.py
├─ hitl.py
├─ errors.py
└─ hook_governance.py

# Hook layers (framework-agnostic)
├─ http_governance_hooks.py
├─ db_governance_hooks.py
└─ file_governance_hooks.py

# Instrumentation (OTel-specific but reusable)
├─ otel_setup.py
├─ span_processor.py
└─ tracing.py

# Framework-specific layer (keep isolated)
└─ langgraph_handler.py
```

**Proposal:** A split package structure:
- `openbox-core-sdk` — generic governance (client, config, verdicts)
- `openbox-otel-sdk` — OTel instrumentation (hooks, tracing)
- `openbox-langgraph-sdk` — LangGraph wrapper (handler, callbacks)

---

## Summary Table: What's Where

| Capability | Module | LOC | Framework-Specific? |
|------------|--------|-----|-------------------|
| Entry point | langgraph_handler.py | 100 | Yes (LangGraph) |
| Event wrapping | langgraph_handler.py | 1000 | Yes (v2 events) |
| HTTP hooks | http_governance_hooks.py | 759 | No |
| DB hooks | db_governance_hooks.py | 874 | No |
| File hooks | file_governance_hooks.py | 407 | No |
| Hook evaluation | hook_governance.py | 377 | No |
| Verdict enforcement | verdict_handler.py | 204 | No |
| HITL polling | hitl.py | 88 | No |
| Governance client | client.py | 329 | No |
| Configuration | config.py | 262 | No |
| Types & enums | types.py | 483 | No |
| OTel setup | otel_setup.py | 481 | No |
| Span processor | span_processor.py | 244 | No |
| Tracing decorator | tracing.py | 320 | No |
| Exceptions | errors.py | 106 | No |
| Exports | __init__.py | 138 | No |
| **Total** | 15 modules | 6,647 | ~25% framework-specific |

---

## Appendix: Complete Example

```python
from openbox_langgraph import create_openbox_graph_handler
from langgraph.graph import StateGraph, START, END
import os

# 1. Define graph state
class State(TypedDict):
    messages: list[dict]

# 2. Define nodes
def search_node(state: State) -> State:
    # This HTTP call will be governed by hooks
    import httpx
    response = httpx.post("https://api.example.com/search", json={
        "query": state["messages"][-1]["content"]
    })
    return {
        **state,
        "messages": [*state["messages"], {"role": "assistant", "content": response.text}]
    }

def llm_node(state: State) -> State:
    # This LLM call will be governed by pre-screen guardrails
    from anthropic import Anthropic
    client = Anthropic()
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=state["messages"],
    )
    return {
        **state,
        "messages": [*state["messages"], {"role": "assistant", "content": response.content[0].text}]
    }

# 3. Build graph
graph = StateGraph(State)
graph.add_node("search", search_node)
graph.add_node("llm", llm_node)
graph.add_edge(START, "search")
graph.add_edge("search", "llm")
graph.add_edge("llm", END)
compiled_graph = graph.compile()

# 4. Wrap with governance
governed = create_openbox_graph_handler(
    graph=compiled_graph,
    api_url=os.environ["OPENBOX_URL"],
    api_key=os.environ["OPENBOX_API_KEY"],
    agent_name="MyAgent",
    hitl={"enabled": True, "poll_interval_ms": 5000},
    tool_type_map={"search": "http"},
)

# 5. Invoke
result = await governed.ainvoke(
    {"messages": [{"role": "user", "content": "Find papers on AI"}]},
    config={"configurable": {"thread_id": "session-1"}},
)
# Result:
# {
#     "messages": [
#         {"role": "user", "content": "..."},
#         {"role": "assistant", "content": "search result..."},
#         {"role": "assistant", "content": "LLM response..."},
#     ]
# }
```

---

**End of Report**

