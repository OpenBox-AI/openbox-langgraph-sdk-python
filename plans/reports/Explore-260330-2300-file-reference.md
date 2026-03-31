# OpenBox LangGraph SDK - File Reference

## Quick File Lookup

### Core Files

| File | Purpose | Key Exports |
|------|---------|------------|
| `openbox_langgraph/__init__.py` | Public API surface | `create_openbox_graph_handler`, `initialize`, all error types, verdict utilities |
| `openbox_langgraph/langgraph_handler.py` | Main handler wrapper & invocation | `OpenBoxLangGraphHandler`, `OpenBoxLangGraphHandlerOptions`, `create_openbox_graph_handler()` |
| `openbox_langgraph/config.py` | Configuration & initialization | `GovernanceConfig`, `initialize()`, `merge_config()`, `get_global_config()` |
| `openbox_langgraph/types.py` | Core types & enums | `Verdict`, `LangChainGovernanceEvent`, `GovernanceVerdictResponse`, `HITLConfig` |
| `openbox_langgraph/errors.py` | Custom exceptions | `OpenBoxError`, `GovernanceBlockedError`, `GuardrailsValidationError`, etc. |

### Integration & Execution

| File | Purpose | Key Functions |
|------|---------|---|
| `openbox_langgraph/client.py` | HTTP client for governance API | `GovernanceClient.evaluate_event()`, `build_auth_headers()` |
| `openbox_langgraph/verdict_handler.py` | Verdict enforcement logic | `enforce_verdict()`, `lang_graph_event_to_context()`, `is_hitl_applicable()` |
| `openbox_langgraph/hitl.py` | Human-in-the-loop polling | `poll_until_decision()`, `HITLPollParams` |

### Observability & Instrumentation

| File | Purpose | Key Functions |
|------|---------|---|
| `openbox_langgraph/otel_setup.py` | OTel initialization | `setup_opentelemetry_for_governance()` |
| `openbox_langgraph/span_processor.py` | OTel span processing | `WorkflowSpanProcessor` |
| `openbox_langgraph/tracing.py` | Custom span creation | `create_span()`, `traced` decorator |
| `openbox_langgraph/http_governance_hooks.py` | HTTP span hooks | Hook functions for httpx interception |
| `openbox_langgraph/db_governance_hooks.py` | Database span hooks | Hook functions for SQLAlchemy interception |
| `openbox_langgraph/hook_governance.py` | Hook-level verdict enforcement | Functions to block/constrain at span level |
| `openbox_langgraph/file_governance_hooks.py` | File I/O span hooks | Hook functions for file operations |

### Testing & Example

| File | Purpose | Content |
|------|---------|---------|
| `test-agent/agent.py` | Minimal working example | `create_react_agent` + 2 tools (search_web, write_report) wrapped with governance |
| `tests/test_governance_changes.py` | Integration tests | Tests for verdict enforcement, HITL, guardrails |
| `tests/test_telemetry_payload.py` | Telemetry tests | Tests for event payload construction |
| `tests/test_contextvars_propagation.py` | Context propagation tests | Tests for OTel context tracking |

## File Dependencies

```
User Code
  ↓
__init__.py (public API)
  ├─ langgraph_handler.py (OpenBoxLangGraphHandler)
  │  ├─ config.py (GovernanceConfig, initialize)
  │  ├─ client.py (GovernanceClient)
  │  ├─ types.py (Verdict, LangChainGovernanceEvent)
  │  ├─ verdict_handler.py (enforce_verdict)
  │  ├─ hitl.py (poll_until_decision)
  │  ├─ otel_setup.py (setup_opentelemetry_for_governance)
  │  └─ span_processor.py (WorkflowSpanProcessor)
  ├─ errors.py (exception hierarchy)
  ├─ config.py
  ├─ types.py
  └─ OTel hooks
     ├─ http_governance_hooks.py
     ├─ db_governance_hooks.py
     ├─ file_governance_hooks.py
     └─ hook_governance.py
```

## Reading Order

For understanding the SDK from scratch:

1. **Start:** `openbox_langgraph/__init__.py`
   - See all public exports
   - 130 lines, imports tell the story

2. **Configuration:** `openbox_langgraph/config.py`
   - How to initialize and configure
   - 265 lines, `GovernanceConfig` dataclass, validation logic

3. **Types:** `openbox_langgraph/types.py`
   - Core data structures
   - 489 lines, `Verdict`, `LangChainGovernanceEvent`, response types

4. **Handler:** `openbox_langgraph/langgraph_handler.py`
   - Main integration point (complex, 1000+ lines)
   - `OpenBoxLangGraphHandler` class
   - Three invocation methods: `ainvoke`, `astream_governed`, `astream_events`
   - Internal: `_GuardrailsCallbackHandler`, `_pre_screen_input`, `_process_event`

5. **Errors:** `openbox_langgraph/errors.py`
   - All exception types (115 lines)
   - Easy reference

6. **Verdict Handling:** `openbox_langgraph/verdict_handler.py`
   - How verdicts are enforced (204 lines)
   - `enforce_verdict()` function
   - Context mapping

7. **Example:** `test-agent/agent.py`
   - Concrete working example (173 lines)
   - Real-world integration pattern

## Key Code Sections

### Session ID Generation (langgraph_handler.py:625-633)

```python
thread_id = _extract_thread_id(config)  # User-provided, stable
_turn = uuid.uuid4().hex                # Fresh per ainvoke
workflow_id = f"{thread_id}-{_turn[:8]}"
run_id = f"{thread_id}-run-{_turn[8:16]}"
```

### Verdict Priority (types.py:41-49)

```python
@property
def priority(self) -> int:
    return {
        Verdict.ALLOW: 0,
        Verdict.CONSTRAIN: 1,
        Verdict.REQUIRE_APPROVAL: 2,
        Verdict.BLOCK: 3,
        Verdict.HALT: 4,
    }[self]
```

### Enforcement Logic (verdict_handler.py:138-203)

```python
def enforce_verdict(response, context) -> VerdictEnforcementResult:
    # 1. HALT (checked first)
    # 2. BLOCK
    # 3. Guardrails validation
    # 4. REQUIRE_APPROVAL (if HITL applicable)
    # 5. CONSTRAIN (warn)
    # 6. ALLOW (no-op)
```

### Pre-screen Execution (langgraph_handler.py:441-601)

```python
async def _pre_screen_input(self, input, workflow_id, run_id):
    # 1. SignalReceived (user prompt)
    # 2. WorkflowStarted
    # 3. LLMStarted (guardrails)
    # 4. enforce_verdict (exceptions propagate here)
    # 5. Poll HITL if needed
    # Returns (workflow_started_sent, pre_screen_response)
```

### Event Processing (langgraph_handler.py:833-899)

```python
async def _process_event(self, event, thread_id, workflow_id, run_id, ...):
    # 1. Map LangGraph event to governance event
    # 2. Skip if disabled in config
    # 3. Send to Core API
    # 4. Enforce verdict
    # 5. Poll HITL if REQUIRE_APPROVAL
```

## Size & Complexity

| File | Lines | Complexity | Purpose |
|------|-------|-----------|---------|
| `langgraph_handler.py` | 1068 | High | Core handler (event processing, enforcement) |
| `types.py` | 489 | Medium | Type definitions (straightforward dataclasses) |
| `config.py` | 265 | Low | Configuration (simple validation) |
| `verdict_handler.py` | 204 | Medium | Verdict logic (clear branching) |
| `client.py` | 270+ | Low-Med | HTTP client (straightforward async) |
| `errors.py` | 115 | Low | Exception definitions |
| `__init__.py` | 131 | Low | Imports & re-exports |
| `otel_setup.py` | 200+ | Medium | OTel instrumentation |
| `span_processor.py` | 150+ | Low-Med | Span processing |
| **Total** | ~3000 | - | Lean, focused SDK |

## For Each Use Case

### "How do I integrate?"
→ See `test-agent/agent.py` (example) + `config.py` (options)

### "What are the verdict types?"
→ `types.py:14-62` (Verdict enum)

### "How does enforcement work?"
→ `verdict_handler.py:138-203` (enforce_verdict function)

### "What errors can be raised?"
→ `errors.py` (all exception types)

### "How do I handle HITL?"
→ `hitl.py` + `verdict_handler.py:190-192` (is_hitl_applicable)

### "What gets sent to Core?"
→ `types.py:256-311` (LangChainGovernanceEvent)

### "How does PII redaction work?"
→ `langgraph_handler.py:78-226` (_GuardrailsCallbackHandler.on_chat_model_start)

### "How are session IDs tracked?"
→ `langgraph_handler.py:625-633` (workflow_id, run_id generation)

## Configuration Quick Reference

**File:** `config.py:56-84` (GovernanceConfig dataclass)

Most commonly customized:
- `on_api_error: "fail_open" | "fail_closed"`
- `hitl: {"enabled": bool, "poll_interval_ms": int}`
- `skip_chain_types: set[str]` (exclude internal chains)
- `tool_type_map: {"tool_name": "http|database|builtin|a2a"}`

All options documented in: `config.py:56-84` + `langgraph_handler.py:312-355`

