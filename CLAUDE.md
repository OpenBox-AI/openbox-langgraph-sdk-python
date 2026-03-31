# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Python SDK (`openbox-langgraph-sdk`) that adds real-time governance to LangGraph agents via OpenBox. It intercepts LangGraph v2 stream events (tool calls, LLM prompts, chain executions), sends them to OpenBox Core's policy engine, and enforces verdicts (ALLOW/BLOCK/CONSTRAIN/REQUIRE_APPROVAL/HALT) — all without modifying agent code.

## Commands

```bash
# Install dependencies
uv sync

# Run all tests
pytest

# Run single test file / pattern
pytest tests/test_governance_changes.py
pytest -k "test_httpx"

# Lint & format
ruff check openbox_langgraph/
ruff format openbox_langgraph/

# Type check
mypy openbox_langgraph/

# Debug mode (verbose governance request/response logging)
OPENBOX_DEBUG=1 pytest -xvs

# Run test agent (requires .env with OPENBOX_URL, OPENBOX_API_KEY, OPENAI_API_KEY)
cd test-agent && uv run python agent.py
```

## Architecture

The SDK has three governance layers that intercept operations at different levels:

### Layer 1: LangGraph Event Stream (langgraph_handler.py)
`OpenBoxLangGraphHandler` wraps a compiled LangGraph graph. It processes the v2 event stream (`on_chain_start/end`, `on_tool_start/end`, `on_chat_model_start/end`), sends governance events to OpenBox Core via `GovernanceClient`, and enforces verdicts. This is the main entry point — users call `create_openbox_graph_handler()` (sync function, returns handler immediately) to wrap their graph.

### Layer 2: Hook Governance (http/db/file_governance_hooks.py)
Intercepts low-level operations using built-in instrumentation:
- **HTTP** (`http_governance_hooks.py`): httpx request/response hooks with started/completed stages
- **Database** (`db_governance_hooks.py`): SQLAlchemy event hooks for query classification
- **File I/O** (`file_governance_hooks.py`): Monkey-patches `os.fdopen()` for file operation tracking

Each hook module creates spans and calls `hook_governance.py` → `evaluate_event()` to get verdicts. `hook_governance.py` is the bridge that resolves the current activity context from the SpanProcessor and calls `GovernanceClient`.

### Layer 3: Activity Context (span_processor.py + otel_setup.py)
`WorkflowSpanProcessor` is a custom SpanProcessor that maps `trace_id` → governance `activity_id`. This lets hook-level governance (Layer 2) find which governance activity a given HTTP/DB/file operation belongs to. `otel_setup.py` initializes all instrumentation and registers the span processor.

### Supporting Modules
- **client.py**: Async HTTP client (`GovernanceClient`) for OpenBox Core API with dedup and connection pooling
- **types.py**: Verdict enum, event types, response parsing, utility functions
- **config.py**: `GovernanceConfig` dataclass, env var parsing, global config registry
- **verdict_handler.py**: Verdict enforcement logic (block/halt/redact)
- **hitl.py**: Human-in-the-loop approval polling loop
- **errors.py**: Exception hierarchy rooted at `OpenBoxError`
- **tracing.py**: `@traced` decorator and `create_span()` for manual span creation

### Data Flow
```
User calls governed.ainvoke()
  → OpenBoxLangGraphHandler streams LangGraph events
    → GovernanceClient sends event to OpenBox Core
    → Verdict received → enforce_verdict()
    → Meanwhile, hooks intercept HTTP/DB/file ops
      → hook_governance.evaluate_event() with activity context from SpanProcessor
      → Additional verdicts enforced at operation level
```

## Tests

Three test files in `tests/`:
- `test_governance_changes.py` — core governance flow (event stream, verdicts, HITL, guardrails)
- `test_telemetry_payload.py` — hook payload structure and content
- `test_contextvars_propagation.py` — context variable propagation across async boundaries

Tests mock the OpenBox Core API; no live server needed. `pytest-asyncio` with `asyncio_mode = "auto"` means no `@pytest.mark.asyncio` decorator needed.

## Key Conventions

- **Python 3.11+**, async-first with sync fallbacks
- **Package manager**: `uv` (lockfile: `uv.lock`)
- **Build**: hatchling
- **Ruff config**: 100 char line length, rules: E/F/I/UP/B/C4/PIE/RUF
- **mypy**: strict mode
- Public API exported from `openbox_langgraph/__init__.py`
- The `test-agent/` directory is a standalone example agent for end-to-end validation, not part of the SDK package
