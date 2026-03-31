# OpenBox LangGraph SDK — System Architecture

## Executive Summary

The SDK provides 3-layer real-time governance for LangGraph agents:

1. **Layer 1: Event Stream** (langgraph_handler.py) — Wraps graph, processes v2 events
2. **Layer 2: Hook Governance** (http/db/file hooks) — Intercepts HTTP/DB/file I/O before execution
3. **Layer 3: Activity Context** (span_processor.py) — Maps trace_id → activity for hooks

All verdicts (ALLOW/BLOCK/CONSTRAIN/REQUIRE_APPROVAL/HALT) flow through a unified enforcement path, with pre-screen guardrails blocking bad inputs before the stream starts.

## Architectural Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ User Code                                                       │
│ result = await governed.ainvoke(input, config)                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
        ┌────────────────────────────────────┐
        │ 1. Pre-Screen Guardrails           │
        │ (WorkflowStarted → LLMStarted)     │
        │ Exceptions propagate to caller     │
        └────────────────────────────────────┘
                         │
                    ✓ Pass or ✗ Raise
                         │
                         ↓ (if pass)
        ┌────────────────────────────────────┐
        │ 2. Stream v2 Events                │
        │ from wrapped LangGraph graph       │
        └────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
      on_chain_    on_tool_start   on_chat_model_
      start/end    /end            start/end
          │              │              │
          ├──────────────┴──────────────┤
          │                            │
          ↓                            ↓
   _map_event()              _GuardrailsCallbackHandler
   Convert to                (LLM PII redaction)
   LangChainGovernanceEvent        │
          │                        ↓
          │                  on_chat_model_start()
          │                  Evaluate LLMStarted
          │                  Redact PII in-place
          │                        │
          ├────────────────────────┤
          │                        │
          ↓                        ↓
    _process_event()         Trace Span
    Evaluate with Core        Created with
    Enforce verdict           trace_id
          │                        │
          │                        ↓
          │                SpanProcessor
          │                .register_trace()
          │                (activity_id ← trace_id)
          │                        │
          └────────────────┬───────┘
                           │
                    ┌──────┴──────┐
                    ↓             ↓
            ALLOW/CONSTRAIN  REQUIRE_APPROVAL
                    │             │
                    │             ↓
                    │      poll_until_decision()
                    │      Wait for HITL approval
                    │             │
                    ↓             ↓ (approval/rejection/timeout)
                 Yield         Raise or Continue
               Event to
               Caller
                    │
          ┌─────────┴──────────┐
          ↓                    ↓
    Tool executes        HTTP/DB/File
    (if applicable)      Operation fires
          │                    │
          │                    ↓
          │            hook_governance.evaluate()
          │            (sync or async)
          │                    │
          │                    ↓
          │            SpanProcessor
          │            .get_activity_context_by_trace()
          │                    │
          │                    ↓
          │            Build hook payload
          │            Evaluate with Core
          │                    │
          │         ┌──────────┴──────────┐
          │         ↓                     ↓
          │      ALLOW          BLOCK/HALT/REQUIRE_APPROVAL
          │         │                     │
          │         ↓                     ↓
          │     Operation         GovernanceBlockedError
          │     continues         (bubbles up through
          │         │              HTTP client)
          │         │                     │
          └─────────┼─────────────────────┤
                    │                     │
                    ↓                     ↓
                Operation            Exception
                completes           unwrapped by
                 success            handler
                    │                     │
                    └─────────────────────┤
                                          │
                                          ↓
                                    Raise to
                                    caller
```

## Layer 1: LangGraph Event Stream

### OpenBoxLangGraphHandler (langgraph_handler.py)

**Responsibility:** Wrap compiled LangGraph graph and process event stream.

**Architecture:**
```python
class OpenBoxLangGraphHandler:
    def __init__(self, graph, client, config, ...):
        self._graph = graph               # Original compiled graph
        self._client = client             # GovernanceClient
        self._config = config             # GovernanceConfig
        self._buffer_mgr = _RunBufferManager()  # Per-run state
        self._llm_activity_map = {}       # Callback UUID → activity_id mapping

    async def ainvoke(self, input, config=None):
        # 1. Pre-screen guardrails (WorkflowStarted, LLMStarted)
        # 2. If pass, stream events
        # 3. For each event, _process_event()
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `ainvoke()` | Single async execution; calls `_pre_screen_input()` then `astream_governed()` |
| `astream_governed()` | Yield v2 events after governance evaluation |
| `_pre_screen_input()` | Evaluate WorkflowStarted/LLMStarted guardrails before stream starts |
| `_process_event()` | Evaluate each stream event, enforce verdict |
| `_map_event()` | Convert v2 event → LangChainGovernanceEvent |

### Activity Lifecycle

Every tool/LLM call creates an activity with lifecycle:

```
on_tool_start
  ├─ Create activity_id (UUID or callback run_id)
  ├─ Span created with trace_id
  ├─ SpanProcessor.register_trace(trace_id → activity_id)
  └─ Store in _RunBufferManager for hook lookup

Tool executes
  └─ HTTP/DB/file hooks fire (details in Layer 2)

on_tool_end
  ├─ Activity marked as completed
  └─ SpanProcessor cleanup
```

### Event Mapping (v2 → Governance)

LangGraph v2 events are rich but not directly governance-focused. Mapping extracts key details:

```python
{
  "event": "on_tool_start",
  "data": {
    "input": "...",
    "tool": "search_web",
    "toolInput": {"query": "..."},
    "runId": "uuid-1",
    "parentRunId": "uuid-0",
  }
}

↓ _map_event()

LangChainGovernanceEvent(
  event_type="ToolStarted",
  workflow_id="uuid-0",  # Root run_id
  run_id="uuid-0",       # Per-invocation
  activity_id="uuid-1",  # Tool call UUID
  activity_type="tool",
  activity_name="search_web",
  activity_input=[{"query": "..."}],
  ...
)
```

### Pre-Screen Guardrails

Guardrails are evaluated BEFORE the stream starts, ensuring bad inputs are rejected immediately:

```
ainvoke(input)
  ├─ generate WorkflowStarted event
  ├─ Evaluate with Core
  ├─ If BLOCK/HALT: raise exception (doesn't start stream)
  │
  ├─ generate LLMStarted event (from initial messages)
  ├─ Evaluate with Core
  ├─ If BLOCK/HALT: raise exception
  │
  └─ If all pass: start streaming
```

**Why pre-screen?**
- LangGraph's graph runner swallows callback exceptions (even with raise_error=True)
- Pre-screen is the only reliable way to block before execution
- Stream events are enforcement (inline, not blocking pre-execution)

### Guardrails Callback Handler

LangChain's `on_chat_model_start()` callback fires BEFORE the LLM call, enabling PII redaction:

```python
class _GuardrailsCallbackHandler(AsyncCallbackHandler):
    async def on_chat_model_start(self, messages, **kwargs):
        # 1. Extract human-turn text from messages
        # 2. Evaluate with Core (guardrails)
        # 3. If redacted_input in response, mutate messages in-place
        # 4. Return (LLM now sees redacted messages)
```

**Why async callback?**
- Injected into config["callbacks"] so LangGraph propagates to every LLM node
- Must be async to evaluate with Core before LLM call
- Exceptions are caught by LangGraph (so not used for blocking here)

## Layer 2: Hook Governance

### HTTP Hooks (http_governance_hooks.py)

**Supported Libraries:**
- httpx (primary)
- requests
- urllib3
- urllib

**Interception Points:**

| Library | Hook | Stage |
|---------|------|-------|
| httpx | `client.send()` | started (before request) |
| requests | `Session.request()` | started |
| urllib3 | `HTTPConnectionPool._validate_conn()` | started |
| urllib | `urlopen()` | started |

**HTTP Hook Flow:**

```python
# User code: await httpx_client.post("https://api.example.com/data", json=...)

# httpx hook fires:
async def httpx_send_hook(request, call_next):
    payload = hook_governance.build_payload("http_request_started", request)
    verdict = await hook_governance.evaluate_async(payload)

    if verdict == Verdict.BLOCK:
        raise GovernanceBlockedError("block", "URL blocked by policy", request.url)

    # ALLOW: continue
    response = await call_next(request)  # Request executes

    # Completed stage (informational, can't block)
    payload = hook_governance.build_payload("http_response_completed", response)
    await hook_governance.evaluate_async(payload)  # Fire-and-forget

    return response
```

**Body Capture:**
- httpx instrumentation sees request.stream (consumed once)
- We patch `httpx.Client.send()` to capture request.content before stream consumption
- Response body is available in completed stage

**Stages:**
- **started:** Can block, has request only
- **completed:** Informational only, has request + response

### Database Hooks (db_governance_hooks.py)

**Supported Libraries:**
- SQLAlchemy (all engines: PostgreSQL, MySQL, SQLite)
- asyncpg (async PostgreSQL)
- psycopg2 (sync PostgreSQL)
- pymongo (MongoDB)
- redis
- MySQL, SQLite (via instrumentation)

**Hook Pattern:**

| DB | Hook | Mechanism |
|----|------|-----------|
| SQLAlchemy | `before_execute` | Event listener |
| asyncpg | `execute()` | wrapt wrapper |
| psycopg2 | `cursor.execute()` | CursorTracer patching |
| pymongo | `before_command_started` | CommandListener |
| redis | `send_command()` | Native hook |

**DB Hook Flow:**

```python
# User code: await asyncpg_conn.execute("SELECT * FROM users WHERE id = $1", user_id)

# asyncpg hook fires (wrapped execute method):
async def wrapped_execute(self, query, *args, **kwargs):
    payload = hook_governance.build_payload("db_query_started", query, args)
    verdict = await hook_governance.evaluate_async(payload)

    if verdict == Verdict.BLOCK:
        raise GovernanceBlockedError("block", "Query blocked", query)

    # ALLOW: execute
    result = await original_execute(self, query, *args, **kwargs)

    # Informational completed stage (can't block anymore)
    payload = hook_governance.build_payload("db_query_completed", query, result)
    await hook_governance.evaluate_async(payload)

    return result
```

**Query Capture:**
- SQL query string captured in started stage
- Result metadata (row count) captured in completed stage
- Full result set is NOT captured (too large, security risk)

### File I/O Hooks (file_governance_hooks.py)

**Supported Operations:**
- `open()` (builtins) — file creation, read, write
- `os.fdopen()` — file descriptor → file object

**Hook Pattern:**

```python
# User code: with open("/tmp/report.txt", "w") as f:
#               f.write(data)

# Patched builtins.open():
def patched_open(path, mode="r", *args, **kwargs):
    payload = hook_governance.build_payload("file_operation_started", path, mode)
    verdict = hook_governance.evaluate_sync(payload)  # Sync (file ops are blocking)

    if verdict == Verdict.BLOCK:
        raise GovernanceBlockedError("block", "File access denied", path)

    # ALLOW: open file
    file_obj = TracedFile(original_open(path, mode, *args, **kwargs))

    return file_obj

class TracedFile:
    def __enter__(self):
        return self

    def write(self, data):
        # Could add per-write governance if needed
        return self._file.write(data)
```

**Path Resolution:**
- For regular files: use pathlib.Path
- For file descriptors (os.fdopen): use os.readlink(f"/proc/self/fd/{fd}") on Linux, fcntl on macOS

## Layer 3: Activity Context (Span Processor)

### WorkflowSpanProcessor (span_processor.py)

**Responsibility:** Map trace_id → (workflow_id, activity_id) so hooks can find which activity they belong to.

**Why needed?**
- LangGraph spawns asyncio.Task for tool execution with new trace context
- Hook (httpx, DB) fires with different trace_id than the activity that spawned it
- Hooks need to know which activity (tool call) they belong to for governance payload
- SpanProcessor bridges this gap via explicit trace_id registration

### Context Flow

```
on_tool_start (activity_id = "uuid-tool-1")
  ├─ Create span with trace_id = "trace-123"
  ├─ SpanProcessor.register_trace("trace-123" → "uuid-tool-1")
  └─ Store in _workflow_context ContextVar

Tool executes in Task with trace context
  ├─ httpx.send() hook fires
  ├─ hook_governance.build_payload()
  └─ SpanProcessor.get_activity_context_by_trace("trace-123")
     returns: (workflow_id="...", activity_id="uuid-tool-1")

Hook payload includes activity_id="uuid-tool-1"
  ├─ Core's policies can see which tool made the request
  └─ Governance can be tool-specific (e.g., "search_web can only access api.example.com")
```

### Fallback Strategies

If trace_id not found in registry (shouldn't happen with proper instrumentation setup):

1. **Single-Activity Mode:** If only one activity is in-flight, assume hook belongs to it
2. **Most-Recent Mode:** If multiple activities, use the most recently started one

```python
def get_activity_context_by_trace(trace_id: str) -> WorkflowSpanBuffer | None:
    # Exact match
    if trace_id in self._trace_id_map:
        return self._trace_id_map[trace_id]

    # Fallback 1: single activity
    if len(self._in_flight) == 1:
        return next(iter(self._in_flight.values()))

    # Fallback 2: most recent
    if self._in_flight:
        return max(self._in_flight.values(), key=lambda b: b.started_at)

    return None
```

## Unified Hook Evaluation (hook_governance.py)

### Payload Building

All hooks (HTTP, DB, file) build a standardized payload:

```python
payload = {
    "workflow_id": "uuid-0",
    "run_id": "uuid-0",
    "activity_id": "uuid-tool-1",
    "hook_type": "http_request",  # http_request, db_query, file_operation
    "hook_stage": "started",       # started or completed
    "request": {
        "method": "POST",
        "url": "https://api.example.com/search",
        "headers": {...},
        "body": "...",
    },
    "hook_trigger": True,  # Signal to Rego policies
}
```

### Evaluation

```python
async def evaluate_async(payload):
    # 1. Get activity context via trace_id
    activity = span_processor.get_activity_context_by_trace(payload.get("trace_id"))

    # 2. Build complete payload with activity metadata
    full_payload = {
        **payload,
        "workflow_id": activity.workflow_id,
        "activity_id": activity.activity_id,
    }

    # 3. Call Core API
    client = GovernanceClient(...)
    verdict = await client.evaluate_raw(full_payload)

    # 4. Enforce verdict
    if verdict == Verdict.BLOCK or verdict == Verdict.HALT:
        raise GovernanceBlockedError(verdict, reason, identifier)

    return verdict

def evaluate_sync(payload):
    # Same, but uses evaluate_event_sync() to avoid asyncio.run() teardown issues
    return asyncio.run(evaluate_async(payload))  # Or task scheduling if event loop exists
```

## Verdict Enforcement (verdict_handler.py)

### Verdict Enum
```python
Verdict.ALLOW           # ✓ Continue execution
Verdict.CONSTRAIN       # ✓ Continue, but log constraint (future: add runtime constraints)
Verdict.REQUIRE_APPROVAL  # ⏸ Pause, poll HITL queue, wait for decision
Verdict.BLOCK           # ✗ Raise GovernanceBlockedError
Verdict.HALT            # ✗ Raise GovernanceHaltError (workflow-level)
```

### Enforcement Logic
```python
def enforce_verdict(verdict, context, client, config):
    if verdict.should_stop():  # BLOCK or HALT
        if verdict == Verdict.HALT:
            raise GovernanceHaltError(reason, ...)
        else:  # BLOCK
            raise GovernanceBlockedError(reason, ...)

    elif verdict.requires_approval():
        # Poll HITL queue
        if config.hitl.enabled:
            approval = poll_until_decision(client, context, config.hitl)
            if approval.approved:
                return  # Continue
            else:
                raise ApprovalRejectedError(...)

    # ALLOW or CONSTRAIN: continue
```

## Data Flow: End-to-End Example

```
User Code:
  result = await governed.ainvoke(
    {"messages": [{"role": "user", "content": "Search for AI papers"}]},
    config={"configurable": {"thread_id": "session-1"}}
  )

Handler:
  1. Generate WorkflowStarted event
  2. Evaluate with Core (guardrails: toxicity, PII)
  3. Result: ALLOW (pass pre-screen)
  4. Start streaming v2 events from graph

Event: on_tool_start (search_web tool)
  1. Create activity_id = "uuid-tool-1"
  2. Create span with trace_id = "trace-abc"
  3. Register: trace_id → activity_id
  4. Generate ToolStarted event
  5. Evaluate with Core
  6. Result: ALLOW
  7. Yield event to caller

Tool executes:
  httpx_client.post("https://api.example.com/search", json={"query": "..."})

HTTP Hook fires:
  1. Build payload with request details
  2. hook_governance.evaluate_async(payload)
     ├─ SpanProcessor.get_activity_context_by_trace("trace-abc")
     │  → Returns: (workflow_id, activity_id="uuid-tool-1")
     ├─ Build complete payload
     └─ GovernanceClient.evaluate_raw()
  3. Result: ALLOW
  4. Request proceeds to api.example.com
  5. Response received

HTTP Hook completed stage fires:
  1. Build payload with response details
  2. hook_governance.evaluate_async(payload) (fire-and-forget)
  3. Result: ALLOW (informational)

Tool completes:
  Event: on_tool_end
  1. Activity marked as completed
  2. SpanProcessor cleanup
  3. Yield event to caller

Stream continues (other events, other tools, LLM call, etc.)

ainvoke() returns final result to user
```

## Fault Tolerance

### Network Errors
- **fail_open:** Network error → return None → continue execution (optimistic)
- **fail_closed:** Network error → raise OpenBoxNetworkError → block execution (safe)

### Missing Activity Context
- SpanProcessor not yet initialized → fallback to single-activity or most-recent
- Ensures hooks can evaluate even in edge cases

### HITL Timeout
- If approval decision doesn't arrive within max_wait_ms → raise ApprovalTimeoutError
- User can catch and retry, or treat as rejection

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Pre-screen (2 events) | ~50-100ms | Network latency to Core |
| Per-event evaluation | ~50-100ms | Inline with streaming |
| HTTP hook evaluation | ~10-50ms | Smaller payload |
| DB hook evaluation | ~10-50ms | Query blocked before execution |
| File hook evaluation | <10ms | Sync, minimal payload |
| HITL polling (per tick) | ~100-500ms | Configurable poll_interval_ms |

**Total latency per invocation:** ~100-200ms for pre-screen + streaming overhead is minimal (pipelined with graph execution)

## Scalability Considerations

1. **No global state:** Dedup resets per (workflow_id, run_id) pair
2. **Persistent client:** httpx.AsyncClient reused across requests
3. **ContextVar safety:** No thread-local state, full async support
4. **Memory:** SpanProcessor ContextVar limited to current task
5. **Concurrency:** Multiple handler instances can run in parallel without contention
