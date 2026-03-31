# OpenBox LangGraph SDK Integration Guide

**Report Generated:** 2026-03-30  
**Scope:** Complete public API, core functionality, configuration, and integration patterns

---

## Executive Summary

The OpenBox LangGraph SDK provides real-time governance and observability for LangGraph agents by intercepting events at three layers:

1. **LangGraph Event Stream** (core) - Tool calls, LLM invocations, chain execution
2. **OTel HTTP Governance Hooks** - Outbound HTTP requests via httpx
3. **Database Governance Hooks** - SQLAlchemy database operations (optional)

A user integrates by wrapping their compiled graph with `create_openbox_graph_handler()` and calling the async invoke methods (`ainvoke`, `astream_governed`, `astream`). The handler transparently applies policies, guardrails, behavior rules, and HITL approval workflows without modifying agent code.

---

## 1. Public API (openbox_langgraph/__init__.py)

### Core Entry Point

```python
create_openbox_graph_handler(
    graph,
    api_url="https://...",
    api_key="obx_live_...",
    agent_name="MyAgent",
    # Optional configuration
    validate=True,
    on_api_error="fail_open",
    tool_type_map={"search_web": "http", "query_db": "database"},
    # ... more options (see config section)
)
```

Returns `OpenBoxLangGraphHandler` instance. See **langgraph_handler.py** for constructor details.

### Configuration & Initialization

```python
from openbox_langgraph import initialize, get_global_config, merge_config

# One-time global setup (synchronous, safe at module level)
initialize(
    api_url="https://core.openbox.ai",
    api_key="obx_live_...",
    governance_timeout=30.0,  # seconds (not ms)
    validate=True,
)

# Retrieve configured state
config = get_global_config()

# Merge partial config over defaults
full_config = merge_config({
    "on_api_error": "fail_open",
    "hitl": {"enabled": True, "poll_interval_ms": 5_000}
})
```

### Error Handling

All errors inherit from `OpenBoxError`:

| Error | When | Action |
|-------|------|--------|
| `OpenBoxAuthError` | Invalid API key | Check credentials at dashboard.openbox.ai |
| `OpenBoxNetworkError` | Core API unreachable | Depends on `on_api_error` setting |
| `OpenBoxInsecureURLError` | HTTP used for non-localhost | Use HTTPS |
| `GovernanceBlockedError` | Verdict is BLOCK | Execution halted by policy |
| `GovernanceHaltError` | Verdict is HALT | Workflow-level stop |
| `GuardrailsValidationError` | Guardrails validation failed | PII/content blocked |
| `ApprovalRejectedError` | HITL approval denied | User rejected the action |
| `ApprovalExpiredError` | Approval window closed | Re-run the action |
| `ApprovalTimeoutError` | HITL polling exceeded max_wait_ms | Action timed out |

### Verdict Types

```python
from openbox_langgraph import Verdict, verdict_from_string, verdict_priority

class Verdict(StrEnum):
    ALLOW = "allow"
    CONSTRAIN = "constrain"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK = "block"
    HALT = "halt"

# Priority ranking (for aggregation)
#   HALT (4) > BLOCK (3) > REQUIRE_APPROVAL (2) > CONSTRAIN (1) > ALLOW (0)
highest = max(verdicts, key=lambda v: v.priority)
```

### Other Exports

- **Governance event types:** `LangChainGovernanceEvent`, `GovernanceVerdictResponse`, `GuardrailsReason`, `GuardrailsResult`, `ApprovalResponse`
- **HITL:** `poll_until_decision`, `DEFAULT_HITL_CONFIG`, `HITLConfig`
- **OTel:** `setup_opentelemetry_for_governance`, `WorkflowSpanProcessor`, `create_span`, `traced`
- **Verdict utilities:** `verdict_from_string`, `verdict_priority`, `highest_priority_verdict`, `verdict_should_stop`, `verdict_requires_approval`
- **Type helpers:** `safe_serialize`, `rfc3339_now`, `parse_governance_response`, `parse_approval_response`
- **Event processing:** `lang_graph_event_to_context`, `enforce_verdict`, `is_hitl_applicable`, `VerdictContext`, `LangGraphStreamEvent`

---

## 2. How create_openbox_graph_handler() Works

### Location
`openbox_langgraph/langgraph_handler.py`

### Initialization Flow

```python
async def main():
    graph = create_react_agent(llm, tools=[...])
    
    governed = await create_openbox_graph_handler(
        graph=graph,
        api_url="https://...",
        api_key="obx_live_...",
        agent_name="MyAgent",
    )
```

**Behind the scenes:**

1. Validates API key format (`obx_live_*` or `obx_test_*`)
2. Validates URL security (HTTPS required for non-localhost)
3. Creates a `GovernanceClient` (persistent httpx.AsyncClient for pooling)
4. Sets up OTel HTTP governance hooks via `setup_opentelemetry_for_governance`
   - Instruments all httpx calls (outbound HTTP requests)
   - Routes each span through the governance engine
5. Returns `OpenBoxLangGraphHandler` wrapping the original graph

### Three Invocation Methods

#### `ainvoke(input, config=None, **kwargs)`

```python
result = await governed.ainvoke(
    {"messages": [{"role": "user", "content": "Search for..."}]},
    config={"configurable": {"thread_id": "session-abc"}},
)
```

- **Behavior:** Streams events via `astream_events`, applies governance inline, **returns final output dict**
- **On governance error:** Raises exception immediately (propagates to caller)
- **On REQUIRE_APPROVAL:** Automatically polls HITL, retries after approval
- **Return:** Final graph state (dict)

#### `astream_governed(input, config=None, **kwargs)`

```python
async for event in governed.astream_governed(
    {"messages": [...]},
    config={"configurable": {"thread_id": "session-abc"}},
):
    # Each event is a LangGraph state update chunk
    print(event)
```

- **Behavior:** Streams update chunks as they arrive, governance applied inline
- **Return:** Each iteration yields a dict (LangGraph state update)

#### `astream(input, config=None, **kwargs)` / `astream_events(..., version="v2")`

```python
async for event in governed.astream_events(
    input,
    config={"configurable": {"thread_id": "session-abc"}},
    version="v2"
):
    # Raw LangGraph event stream (on_tool_start, on_chain_end, etc.)
    print(event)
```

- **Behavior:** Governance runs transparently, raw events re-yielded
- **Backward compatibility:** Allows tooling like `langgraph dev` to work transparently

### Event Processing Pipeline

1. **Pre-screen phase** (`_pre_screen_input`):
   - Fires `SignalReceived` event with user prompt
   - Fires `WorkflowStarted` event
   - **Pre-screens first LLM call** with guardrails → raises exceptions immediately
   - Returns verdicts for PII redaction without duplicate events

2. **Stream phase** (via `astream_events`):
   - Injects `_GuardrailsCallbackHandler` into LangGraph config
   - Handler intercepts `on_chat_model_start` BEFORE LLM call
   - Applies PII redaction (in-place message mutation)
   - Main loop processes all LangGraph events (`on_tool_start`, `on_chain_end`, etc.)

3. **Verdict enforcement**:
   - **HALT** → raises `GovernanceHaltError` (workflow stops)
   - **BLOCK** → raises `GovernanceBlockedError` (tool/chain blocked)
   - **REQUIRE_APPROVAL** at start events → polls HITL, retries after approval
   - **CONSTRAIN** → logs warning, execution continues
   - **ALLOW** → execution continues

### ID Generation & Session Tracking

```python
thread_id = config["configurable"]["thread_id"]  # User-provided session ID

# Per-invocation (fresh on each ainvoke call)
_turn = uuid.uuid4().hex
workflow_id = f"{thread_id}-{_turn[:8]}"        # Logical session turn (sealed after WorkflowCompleted)
run_id = f"{thread_id}-run-{_turn[8:16]}"       # Execution attempt ID (distinct from workflow_id)
```

**Key:** OpenBox Core seals a workflow after `WorkflowCompleted` → reusing `workflow_id` causes `HALT: "fully attested and sealed"`. Fresh IDs per turn prevent this.

---

## 3. GovernanceConfig Options (config.py)

### Full Configuration Reference

```python
@dataclass
class GovernanceConfig:
    # Fail behavior
    on_api_error: str = "fail_open"              # "fail_open" | "fail_closed"
    api_timeout: float = 30.0                    # seconds (accepts ≤600s or ms > 600)
    
    # Event filtering
    send_chain_start_event: bool = True
    send_chain_end_event: bool = True
    send_tool_start_event: bool = True
    send_tool_end_event: bool = True
    send_llm_start_event: bool = True
    send_llm_end_event: bool = True
    
    # Exclude specific runnable types from telemetry
    skip_chain_types: set[str] = field(default_factory=set)
    skip_tool_types: set[str] = field(default_factory=set)
    
    # Human-in-the-loop
    hitl: HITLConfig = field(default_factory=HITLConfig)
    
    # Metadata
    session_id: str | None = None                # Attach to all events
    agent_name: str | None = None                # Agent identifier (e.g., "MyAgent")
    task_queue: str = "langgraph"                # Temporal task queue label
    
    # Execution tree
    use_native_interrupt: bool = False           # Reserved for future use
    root_node_names: set[str] = field(default_factory=set)
    tool_type_map: dict[str, str] = field(default_factory=dict)
```

### HITL Configuration

```python
@dataclass
class HITLConfig:
    enabled: bool = True
    poll_interval_ms: int = 5_000                # How often to check for approval
    skip_tool_types: set[str] = field(default_factory=set)
```

### Tool Type Mapping

```python
tool_type_map = {
    "search_web": "http",                        # Outbound HTTP call
    "query_db": "database",                      # Database query
    "write_file": "builtin",                     # File I/O
    "invoke_task": "a2a",                        # Agent-to-agent (subagent)
}
```

**Supported values:** `"http"`, `"database"`, `"builtin"`, `"a2a"`, `"custom"`

Default behavior:
- If tool is listed → use the mapped type
- If tool name matches subagent → auto-set to `"a2a"`
- Otherwise → bare tool name (no type prefix)

### Usage Example

```python
from openbox_langgraph import create_openbox_graph_handler

governed = await create_openbox_graph_handler(
    graph=my_graph,
    api_url="https://...",
    api_key="obx_live_...",
    agent_name="MyAgent",
    on_api_error="fail_closed",          # Enforce strict error handling
    api_timeout=60_000,                  # 60 seconds (or 60, interpreted as seconds if ≤ 600)
    send_chain_start_event=True,
    send_tool_start_event=True,
    skip_chain_types={"agent", "RunnableSequence"},
    hitl={
        "enabled": True,
        "poll_interval_ms": 5_000,
        "skip_tool_types": {"http"},     # Never ask for approval on HTTP tools
    },
    tool_type_map={
        "search_web": "http",
        "query_db": "database",
    },
)
```

---

## 4. Test-Agent Example (test-agent/agent.py)

### Setup

```bash
cd test-agent
cp .env.example .env
# Set: OPENBOX_URL, OPENBOX_API_KEY, OPENAI_API_KEY
uv run python agent.py
```

### Architecture

**Graph:**
- Built with `create_react_agent(llm, tools=[search_web, write_report])`
- Minimal agent: LLM selects between two tools

**Tools:**
- `search_web(query)` → HTTP GET to Wikipedia, returns search results
- `write_report(title, content, classification)` → writes to in-memory store

**Governance Integration:**

```python
governed = create_openbox_graph_handler(
    graph=graph,
    api_url=os.environ["OPENBOX_URL"],
    api_key=os.environ["OPENBOX_API_KEY"],
    agent_name="LangGraphTestAgent",
    validate=True,
    on_api_error="fail_open",
    tool_type_map={"search_web": "http"},
    skip_chain_types={"agent", "call_model", "RunnableSequence", "Prompt", "ChatPromptTemplate"},
    hitl={"enabled": True, "poll_interval_ms": 5_000, "max_wait_ms": 300_000},
)

thread_id = f"langgraph-test-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
await _repl(governed, thread_id=thread_id)
```

**REPL Loop:**

```python
while True:
    user_text = input("You: ").strip()
    result = await governed.ainvoke(
        {"messages": [{"role": "user", "content": user_text}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    msgs = result.get("messages") or []
    last = msgs[-1].content if msgs else ""
    print(f"\nAgent: {last}\n")
```

**Key Points:**
- Multi-turn session (reuses same `thread_id` across turns)
- Each turn generates fresh `workflow_id` + `run_id` (avoids sealing issues)
- Governance applies to every tool call and LLM invocation
- HITL polling enabled (5s poll, 300s max wait)
- HTTP tools labeled for specialized governance policies

---

## 5. Types & Verdict Handling

### Core Types

#### Verdict Enum

```python
from openbox_langgraph import Verdict

v = Verdict.ALLOW              # "allow" (default)
v = Verdict.CONSTRAIN          # "constrain" (warning)
v = Verdict.REQUIRE_APPROVAL   # "require_approval" (HITL)
v = Verdict.BLOCK              # "block" (immediate stop)
v = Verdict.HALT               # "halt" (workflow stop)

v.priority                     # int (0-4, higher = more restrictive)
v.should_stop()                # bool (True if BLOCK or HALT)
v.requires_approval()          # bool (True if REQUIRE_APPROVAL)
```

#### Governance Event Payload

```python
from openbox_langgraph import LangChainGovernanceEvent

event = LangChainGovernanceEvent(
    source="workflow-telemetry",
    event_type="WorkflowStarted" | "ActivityStarted" | "ActivityCompleted" | ...,
    workflow_id="session-abc-12345678",
    run_id="session-abc-run-87654321",
    workflow_type="LangGraphRun",
    task_queue="langgraph",
    timestamp="2026-03-30T14:23:45.123Z",
    
    # LangGraph routing
    langgraph_node="agent",
    langgraph_step=5,
    subagent_name=None,
    
    # Activity metadata
    activity_id="run-id-uuid",
    activity_type="tool_call" | "llm_call",
    activity_input=["arg1", "arg2"],
    activity_output={"result": "..."},
    
    # LLM fields
    llm_model="gpt-4o-mini",
    input_tokens=100,
    output_tokens=50,
    prompt="...",
    
    # Tool fields
    tool_name="search_web",
    tool_type="http",
    tool_input={"query": "..."},
    
    # Session tracking
    session_id="session-abc",
    
    # Telemetry
    start_time=1704067425123.0,  # ms since epoch
    end_time=1704067426500.0,
    duration_ms=1377.0,
    spans=[],  # OTel spans (HTTP, DB)
)

payload = event.to_dict()      # Serialize, omitting None values
```

#### Governance Response

```python
from openbox_langgraph import GovernanceVerdictResponse

response = GovernanceVerdictResponse(
    verdict=Verdict.ALLOW,
    reason="Passed policy checks",
    policy_id="pol_123",
    risk_score=0.1,
    
    # Guardrails (PII, content filtering)
    guardrails_result=GuardrailsResult(
        input_type="activity_input",
        redacted_input=[{"prompt": "[PII REDACTED]..."}],
        validation_passed=False,
        reasons=[
            GuardrailsReason(
                type="pii",
                field="prompt",
                reason="Email address detected",
            )
        ],
    ),
    
    # HITL
    approval_id="appr_456",
    approval_expiration_time="2026-03-30T14:30:00Z",
    
    # Metadata
    trust_tier="verified",
    alignment_score=0.95,
    constraints=[...],
)

response.action  # Backward compat: "continue" | "stop" | "require-approval" | ...
```

### Verdict Context & Enforcement

```python
from openbox_langgraph import (
    VerdictContext,
    enforce_verdict,
    lang_graph_event_to_context,
)

# Map LangGraph event to verdict context
context: VerdictContext = lang_graph_event_to_context(
    "on_tool_start",
    is_root=False,
)
# context = "tool_start"

# Enforce verdict
try:
    result = enforce_verdict(response, context)
    if result.requires_hitl:
        await poll_until_decision(...)
except GuardrailsValidationError as e:
    # PII/content failed validation
    print(f"Guardrails blocked: {e.reasons}")
except GovernanceBlockedError as e:
    # BLOCK verdict
    print(f"Blocked: {e}")
except GovernanceHaltError as e:
    # HALT verdict
    print(f"Halt: {e}")
```

**VerdictContext Values:**
- `"chain_start"`, `"chain_end"` - Generic chain lifecycle
- `"tool_start"`, `"tool_end"` - Tool invocation
- `"llm_start"`, `"llm_end"` - LLM call
- `"graph_node_start"`, `"graph_node_end"` - Individual graph node
- `"graph_root_start"`, `"graph_root_end"` - Root/outermost graph
- `"agent_action"`, `"agent_finish"` - Legacy ReAct agent events
- `"other"` - Unknown/unmapped

**Observation-only contexts** (no enforcement, telemetry only):
- `chain_end`, `graph_root_end`, `agent_finish`, `other`

**Enforceable contexts** (verdicts trigger actions):
- Start events: `*_start` → enforcement before execution
- Tool/LLM end: tool/llm_end → enforcement on output
- HITL applicable: Most except `chain_end`, `graph_root_end`

---

## Integration Checklist for Users

### 1. Setup
- [ ] Install: `pip install openbox-langgraph-sdk-python`
- [ ] Set environment: `OPENBOX_URL`, `OPENBOX_API_KEY`
- [ ] (Optional) Setup OTel instrumentation (auto-enabled by default)

### 2. Initialize SDK
```python
from openbox_langgraph import initialize

initialize(
    api_url=os.environ["OPENBOX_URL"],
    api_key=os.environ["OPENBOX_API_KEY"],
    governance_timeout=30.0,
    validate=True,  # Ping server on startup
)
```

### 3. Wrap Graph
```python
from openbox_langgraph import create_openbox_graph_handler

graph = create_react_agent(llm, tools=[...])
governed = await create_openbox_graph_handler(
    graph=graph,
    api_url=os.environ["OPENBOX_URL"],
    api_key=os.environ["OPENBOX_API_KEY"],
    agent_name="MyAgent",
)
```

### 4. Invoke with Thread ID
```python
result = await governed.ainvoke(
    {"messages": [{"role": "user", "content": "..."}]},
    config={"configurable": {"thread_id": "session-xyz"}},
)
```

### 5. Handle Errors
```python
try:
    result = await governed.ainvoke(...)
except GuardrailsValidationError as e:
    print(f"PII detected: {e.reasons}")
except (GovernanceBlockedError, GovernanceHaltError) as e:
    print(f"Governance blocked: {e}")
except ApprovalRejectedError:
    print("User rejected the action")
except ApprovalTimeoutError as e:
    print(f"Approval timed out after {e.max_wait_ms}ms")
```

### 6. (Optional) Configure Advanced Features
```python
governed = await create_openbox_graph_handler(
    graph=graph,
    api_url=...,
    api_key=...,
    # Fail-closed on API error
    on_api_error="fail_closed",
    
    # Skip telemetry for specific runnable types
    skip_chain_types={"agent", "RunnableSequence"},
    skip_tool_types={"builtin_tool"},
    
    # Enable HITL with custom polling
    hitl={
        "enabled": True,
        "poll_interval_ms": 3_000,
        "skip_tool_types": {"http"},
    },
    
    # Classify tools for governance
    tool_type_map={
        "search_web": "http",
        "query_db": "database",
        "write_file": "builtin",
    },
)
```

---

## Common Integration Patterns

### Pattern 1: Simple Synchronous Initialization

```python
# Module-level (safe to call at import time)
from openbox_langgraph import initialize
import os

initialize(
    api_url=os.environ["OPENBOX_URL"],
    api_key=os.environ["OPENBOX_API_KEY"],
    validate=True,  # Fails fast on bad credentials
)
```

### Pattern 2: Per-Request Custom Config

```python
from openbox_langgraph import create_openbox_graph_handler

async def create_agent():
    graph = create_react_agent(llm, tools=[...])
    return await create_openbox_graph_handler(
        graph=graph,
        api_url=os.environ["OPENBOX_URL"],
        api_key=os.environ["OPENBOX_API_KEY"],
        agent_name="MyAgent",
        on_api_error="fail_open",  # Don't break on API outages
        hitl={"enabled": True, "poll_interval_ms": 5_000},
    )
```

### Pattern 3: Multi-Turn Session (REPL)

```python
thread_id = f"session-{uuid.uuid4().hex}"
while True:
    user_input = input("You: ")
    result = await governed.ainvoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    # Response is the final state dict
    print(f"Agent: {result['messages'][-1].content}")
```

### Pattern 4: Streaming with Event Processing

```python
async for event in governed.astream_governed(
    {"messages": [...]},
    config={"configurable": {"thread_id": thread_id}},
):
    # Each event is a state update chunk
    if "messages" in event:
        for msg in event["messages"]:
            if hasattr(msg, "content"):
                print(msg.content, end="", flush=True)
```

### Pattern 5: Graceful Error Handling

```python
from openbox_langgraph import (
    GuardrailsValidationError,
    GovernanceBlockedError,
    ApprovalRejectedError,
    OpenBoxNetworkError,
)

try:
    result = await governed.ainvoke(input, config=config)
except GuardrailsValidationError as e:
    print(f"Content policy violation: {'; '.join(e.reasons)}")
    return "I can't help with that."
except GovernanceBlockedError as e:
    print(f"Action blocked: {e}")
    return "This action is not allowed."
except ApprovalRejectedError:
    print("User declined the action.")
    return "The action was not approved."
except OpenBoxNetworkError:
    # Core API is down — fail open (if configured)
    print("Governance unavailable, proceeding without checks")
    result = await graph.ainvoke(input)
```

---

## Debugging

### Enable Debug Output

```bash
OPENBOX_DEBUG=1 python agent.py
```

**Output includes:**
- All governance request payloads (JSON)
- All governance response verdicts
- All LangGraph v2 stream events
- Deduplication info

### Example Debug Output

```
[OpenBox Debug] governance request: {
  "source": "workflow-telemetry",
  "event_type": "WorkflowStarted",
  "workflow_id": "session-abc-12345678",
  "activity_input": [{"messages": [...]}],
  ...
}

[OBX_EVENT] on_tool_start          name='search_web'  node='agent'
[OBX_EVENT] on_tool_end            name='search_web'  node='agent'
[OBX_EVENT] on_chain_end           name='agent'       node='__root__'
```

---

## Unresolved Questions

1. **Framework extensions:** How does `openbox-deepagent` extend this handler for DeepAgents-specific subagent routing?
2. **OTel setup details:** What specific OTel exporters/processors are configured by `setup_opentelemetry_for_governance`?
3. **SQLAlchemy hooking:** How are database spans instrumented when `sqlalchemy_engine` is provided?
4. **Custom span attachment:** What's the pattern for `create_span` and `traced` decorators for tracing custom functions?
5. **Hook governance:** How do HTTP hooks (`hook_governance.py`, `http_governance_hooks.py`) work at the httpx instrumentation level?
6. **Behavioral Rules (AGE):** How do the AGE-evaluated behavior rules on ActivityCompleted verdicts flow through enforcement?

