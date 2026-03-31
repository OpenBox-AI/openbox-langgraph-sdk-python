# openbox-langgraph-sdk-python

[![PyPI](https://img.shields.io/pypi/v/openbox-langgraph-sdk-python)](https://pypi.org/project/openbox-langgraph-sdk-python/)
[![Python](https://img.shields.io/pypi/pyversions/openbox-langgraph-sdk-python)](https://pypi.org/project/openbox-langgraph-sdk-python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Real-time governance and observability for [LangGraph](https://github.com/langchain-ai/langgraph) agents — powered by [OpenBox](https://openbox.ai).

**OpenBox** sits between your agent and the world. Every tool call, LLM prompt, HTTP request, database query, and file operation passes through a policy engine before it executes. You write policies in [Rego](https://www.openpolicyagent.org/docs/latest/policy-language/); OpenBox enforces them — blocking harmful actions, screening for PII, and routing sensitive operations to a human approver — all without changing your agent code.

---

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Configuration reference](#configuration-reference)
- [Governance features](#governance-features)
  - [Policies (OPA / Rego)](#policies-opa--rego)
  - [Guardrails](#guardrails)
  - [Human-in-the-loop (HITL)](#human-in-the-loop-hitl)
  - [Behavior Rules (AGE)](#behavior-rules-age)
  - [Tool classification](#tool-classification)
- [Hook governance](#hook-governance)
  - [HTTP hooks](#http-hooks)
  - [Database hooks](#database-hooks)
  - [File I/O hooks](#file-io-hooks)
  - [Custom function tracing](#custom-function-tracing)
- [Error handling](#error-handling)
- [Advanced usage](#advanced-usage)
- [Debugging](#debugging)
- [Contributing](#contributing)

---

## Architecture

The SDK has three governance layers that intercept operations at different levels:

```
Your code                 SDK (3 layers)                              OpenBox Core
──────────                ──────────────                              ────────────

governed.ainvoke()
  │
  ├─ Layer 1: LangGraph Event Stream (langgraph_handler.py)
  │    on_tool_start/end  ─────────────────────────────────────────→  Policy engine
  │    on_chat_model_start/end  ───────────────────────────────────→  Guardrails
  │    on_chain_start/end  ────────────────────────────────────────→  HITL queue
  │         ↑ enforce verdict (allow / block / redact / pause)
  │
  ├─ Layer 2: Hook Governance (http/db/file hooks)
  │    httpx/requests outbound calls  ─────────────────────────────→  Behavior Rules (AGE)
  │    SQL queries (psycopg2, asyncpg, pymongo, redis, SQLAlchemy) →  Per-operation policies
  │    File I/O (open, read, write)  ──────────────────────────────→  File access policies
  │         ↑ block before operation executes (started stage)
  │
  └─ Layer 3: Activity Context (span_processor.py)
       Maps trace_id → governance activity_id
       Links hook-level operations to the tool call that triggered them
```

**Layer 1** wraps your compiled LangGraph graph and intercepts the [v2 event stream](https://langchain-ai.github.io/langgraph/how-tos/streaming-events-from-within-tools/). It sends governance events (WorkflowStarted, ActivityStarted, etc.) to OpenBox Core and enforces verdicts.

**Layer 2** uses built-in instrumentation to intercept low-level operations (HTTP requests, DB queries, file I/O) made by your tools. Each operation is evaluated at two stages: `started` (can block) and `completed` (informational).

**Layer 3** maintains the mapping between traces and governance activities, so Layer 2 hooks know which tool call each operation belongs to.

**Zero graph changes required.** You keep writing LangGraph exactly as you normally would.

---

## Installation

```bash
pip install openbox-langgraph-sdk-python
```

Or with `uv`:

```bash
uv add openbox-langgraph-sdk-python
```

**Requirements:** Python 3.11+, `langgraph >= 0.2`, `langchain-core >= 0.3`

**Included instrumentation libraries:** The package includes built-in instrumentors for `httpx`, `requests`, `urllib3`, `psycopg2`, `asyncpg`, `mysql`, `pymysql`, `pymongo`, `redis`, `sqlalchemy`, and `sqlite3`. These are activated automatically when you create the handler.

---

## Quickstart

### 1. Get your API key

Sign in to [dashboard.openbox.ai](https://dashboard.openbox.ai), create an agent called `"MyAgent"`, and copy your API key (`obx_live_...` or `obx_test_...`).

### 2. Set environment variables

```bash
export OPENBOX_URL="https://core.openbox.ai"
export OPENBOX_API_KEY="obx_live_..."
```

### 3. Wrap your graph

```python
import os
import asyncio
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from openbox_langgraph import create_openbox_graph_handler

# Your existing agent — no changes needed
llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, tools=[search_web, write_file])

async def main():
    governed = create_openbox_graph_handler(
        graph=agent,
        api_url=os.environ["OPENBOX_URL"],
        api_key=os.environ["OPENBOX_API_KEY"],
        agent_name="MyAgent",  # must match the agent name in your dashboard
    )

    result = await governed.ainvoke(
        {"messages": [{"role": "user", "content": "Search for the latest AI papers"}]},
        config={"configurable": {"thread_id": "session-001"}},
    )
    print(result["messages"][-1].content)

asyncio.run(main())
```

That's it. Your agent now sends governance events to OpenBox on every tool call, LLM prompt, HTTP request, and database query.

### Try it locally (included test agent)

The repository includes a runnable LangGraph test agent under `test-agent/`.

It validates:

- **Guardrails** on LLM prompts
- **Policies** on tool invocations (BLOCK / REQUIRE_APPROVAL)
- **HITL** approval polling
- **Behavior Rules (AGE)** via `httpx` spans from `search_web`

See `test-agent/README.md` for setup and run instructions.

---

## Configuration reference

`create_openbox_graph_handler` accepts the following keyword arguments:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `graph` | `CompiledGraph` | **required** | Your compiled LangGraph graph |
| `api_url` | `str` | **required** | Base URL of your OpenBox Core instance |
| `api_key` | `str` | **required** | API key (`obx_live_*` or `obx_test_*`) |
| `agent_name` | `str` | `None` | Agent name as configured in the dashboard |
| `validate` | `bool` | `True` | Validate API key against server on startup |
| `on_api_error` | `str` | `"fail_open"` | `"fail_open"` (allow on error) or `"fail_closed"` (block on error) |
| `governance_timeout` | `float` | `30.0` | HTTP timeout in seconds for governance calls |
| `session_id` | `str` | `None` | Optional session identifier for multi-session agents |
| `task_queue` | `str` | `"langgraph"` | Task queue label attached to all governance events |
| `hitl` | `dict` | `{}` | Human-in-the-loop config (see [HITL](#human-in-the-loop-hitl)) |
| `tool_type_map` | `dict[str, str]` | `{}` | Map tool names to semantic types (see [Tool classification](#tool-classification)) |
| `skip_chain_types` | `set[str]` | `set()` | Chain node names to skip |
| `skip_tool_types` | `set[str]` | `set()` | Tool names to skip entirely |
| `send_chain_start_event` | `bool` | `True` | Send `WorkflowStarted` event |
| `send_chain_end_event` | `bool` | `True` | Send `WorkflowCompleted` event |
| `send_llm_start_event` | `bool` | `True` | Send `LLMStarted` event (enables prompt guardrails) |
| `send_llm_end_event` | `bool` | `True` | Send `LLMCompleted` event |
| `enable_telemetry` | `bool` | `True` | Enable hook governance (HTTP, DB, file I/O) |
| `sqlalchemy_engine` | `Engine` | `None` | SQLAlchemy Engine instance to instrument (if created before handler) |
| `resolve_subagent_name` | `Callable` | `None` | Hook for framework-specific subagent name detection |

---

## Governance features

### Policies (OPA / Rego)

Policies are written in [Rego](https://www.openpolicyagent.org/docs/latest/policy-language/) and configured in the OpenBox dashboard under your agent. The SDK sends an `ActivityStarted` event before every tool call; your policy decides what happens next.

**Fields available in `input`:**

| Field | Type | Description |
|---|---|---|
| `input.event_type` | `string` | `"ActivityStarted"` or `"ActivityCompleted"` |
| `input.activity_type` | `string` | Tool name (e.g. `"search_web"`) |
| `input.activity_input` | `array` | Tool arguments as a JSON array |
| `input.workflow_type` | `string` | Your `agent_name` |
| `input.workflow_id` | `string` | Session workflow ID |
| `input.trust_tier` | `int` | Agent trust tier (1–4) from dashboard |
| `input.hook_trigger` | `bool` | `true` when event is a hook-level re-evaluation |

**Example — block a restricted search term:**

```rego
package org.openboxai.policy

import future.keywords.if
import future.keywords.in

default result = {"decision": "CONTINUE", "reason": null}

restricted_terms := {"nuclear weapon", "bioweapon", "malware synthesis"}

result := {"decision": "BLOCK", "reason": "Restricted topic."} if {
    input.event_type == "ActivityStarted"
    input.activity_type == "search_web"
    not input.hook_trigger
    count(input.activity_input) > 0
    entry := input.activity_input[0]
    is_object(entry)
    some term in restricted_terms
    contains(lower(entry.query), term)
}
```

**Example — require approval for sensitive exports:**

```rego
result := {"decision": "REQUIRE_APPROVAL", "reason": "Data export requires sign-off."} if {
    input.event_type == "ActivityStarted"
    input.activity_type == "export_data"
    not input.hook_trigger
}
```

**Possible decisions:**

| Decision | Effect |
|---|---|
| `CONTINUE` | Tool executes normally |
| `BLOCK` | `GovernanceBlockedError` raised — tool does not execute |
| `REQUIRE_APPROVAL` | Agent pauses; human must approve or reject in dashboard |
| `HALT` | `GovernanceHaltError` raised — session terminated |

#### The `hook_trigger` guard

The SDK's hook layer intercepts outgoing HTTP requests, DB queries, and file operations made by your tools and sends additional governance events with `hook_trigger: true`.

**Always add `not input.hook_trigger`** to `BLOCK` and `REQUIRE_APPROVAL` rules to prevent them from double-firing on hook-level re-evaluations.

---

### Guardrails

Guardrails screen the content of LLM prompts and tool outputs. Configure them in the dashboard per agent.

| Type | What it detects |
|---|---|
| PII detection | Names, emails, phone numbers, SSNs, credit cards |
| Content filter | Harmful or unsafe content categories |
| Toxicity | Toxic language |
| Ban words | Custom word/phrase blocklist |
| Regex | Custom regex patterns |

When a guardrail fires on an LLM prompt:
- **PII redaction** — the prompt is automatically redacted before the LLM sees it
- **Content block** — `GuardrailsValidationError` is raised

---

### Human-in-the-loop (HITL)

When a policy returns `REQUIRE_APPROVAL`, the agent pauses and polls OpenBox for a human decision:

```python
governed = create_openbox_graph_handler(
    graph=agent,
    api_url=os.environ["OPENBOX_URL"],
    api_key=os.environ["OPENBOX_API_KEY"],
    agent_name="MyAgent",
    hitl={
        "enabled": True,
        "poll_interval_ms": 5_000,
    },
)
```

The human approves or rejects from the OpenBox dashboard. The SDK resumes or raises `ApprovalRejectedError` accordingly.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `False` | Enable HITL polling |
| `poll_interval_ms` | `int` | `5000` | How often to poll for a decision |
| `skip_tool_types` | `set[str]` | `set()` | Tools that never wait for HITL |

---

### Behavior Rules (AGE)

Behavior Rules detect patterns across sequences of tool calls within a session. They are configured in the dashboard and enforced by the OpenBox Activity Governance Engine (AGE).

Example use cases:
- Flag if an agent calls an external URL more than N times in one session
- Detect unusual tool call sequences (e.g. data exfiltration patterns)
- Enforce rate limits per tool type

The SDK automatically attaches HTTP span telemetry so that outbound HTTP calls are captured and sent with `ActivityCompleted` events.

---

### Tool classification

Classify tools into semantic types for richer execution trees and type-based policy matching:

```python
governed = create_openbox_graph_handler(
    graph=agent,
    api_url=os.environ["OPENBOX_URL"],
    api_key=os.environ["OPENBOX_API_KEY"],
    agent_name="MyAgent",
    tool_type_map={
        "search_web": "http",
        "export_data": "http",
        "query_db":    "database",
        "write_file":  "builtin",
    },
)
```

**Supported values:** `"http"`, `"database"`, `"builtin"`, `"a2a"`, `"custom"`

The SDK appends `__openbox` metadata to `activity_input` so Rego can match on tool type:

```rego
result := {"decision": "REQUIRE_APPROVAL", "reason": "HTTP tools need approval."} if {
    input.event_type == "ActivityStarted"
    not input.hook_trigger
    some item in input.activity_input
    item["__openbox"].tool_type == "http"
}
```

---

## Hook governance

The SDK uses built-in instrumentation to intercept low-level operations made by your tools. This runs automatically when `enable_telemetry=True` (the default).

### HTTP hooks

Intercepts outbound HTTP requests via `httpx`, `requests`, `urllib3`, and `urllib`. Each request is evaluated at two stages:

- **started** — before the request is sent (can block)
- **completed** — after the response is received (informational, captures status code and body)

Governance payloads include `http_method`, `http_url`, `request_body`, `response_body`, `http_status_code`, and `request_headers`/`response_headers`.

The SDK automatically ignores requests to the OpenBox Core API itself to prevent recursion.

### Database hooks

Intercepts database queries for all supported libraries:

| Library | Protocol |
|---|---|
| `psycopg2` | PostgreSQL |
| `asyncpg` | PostgreSQL (async) |
| `mysql-connector-python` | MySQL |
| `pymysql` | MySQL |
| `sqlite3` | SQLite |
| `pymongo` | MongoDB |
| `redis` | Redis |
| `sqlalchemy` | ORM (any backend) |

Governance payloads include `db_system`, `db_name`, `db_operation`, `db_statement`, and `server_address`/`server_port`.

If your SQLAlchemy engine is created before the handler, pass it explicitly:

```python
from sqlalchemy import create_engine

engine = create_engine("postgresql://...")

governed = create_openbox_graph_handler(
    graph=agent,
    api_url=os.environ["OPENBOX_URL"],
    api_key=os.environ["OPENBOX_API_KEY"],
    agent_name="MyAgent",
    sqlalchemy_engine=engine,
)
```

### File I/O hooks

Intercepts `builtins.open()` and `os.fdopen()` to track file operations. Governance payloads include `file_path`, `file_mode`, `file_operation`, and byte counts.

System paths (`/dev/`, `/proc/`, `/sys/`, `__pycache__`, `.pyc`, `.so`) are automatically skipped.

### Custom function tracing

Use the `@traced` decorator to capture internal function calls as traced spans with governance evaluation:

```python
from openbox_langgraph import traced

@traced
def process_data(input_data):
    return transform(input_data)

@traced(name="custom-span-name", capture_args=True, capture_result=True)
async def fetch_data(url):
    return await http_get(url)
```

For manual span creation:

```python
from openbox_langgraph import create_span

with create_span("my-operation", {"input": data}) as span:
    result = do_something()
    span.set_attribute("output", result)
```

---

## Error handling

```python
from openbox_langgraph import (
    GovernanceBlockedError,
    GovernanceHaltError,
    GuardrailsValidationError,
    ApprovalRejectedError,
    ApprovalTimeoutError,
)

try:
    result = await governed.ainvoke({"messages": [...]}, config=...)
except GovernanceBlockedError as e:
    print(f"Action blocked by policy: {e}")
except GovernanceHaltError as e:
    print(f"Session halted: {e}")
except GuardrailsValidationError as e:
    print(f"Guardrail triggered: {e}")
except ApprovalRejectedError as e:
    print(f"Human rejected the action: {e}")
except ApprovalTimeoutError as e:
    print(f"HITL approval timed out: {e}")
```

| Exception | When raised |
|---|---|
| `GovernanceBlockedError` | Policy returned `BLOCK` |
| `GovernanceHaltError` | Policy returned `HALT` |
| `GuardrailsValidationError` | Guardrail fired on an LLM prompt or tool output |
| `ApprovalRejectedError` | Human rejected a `REQUIRE_APPROVAL` decision |
| `ApprovalTimeoutError` | HITL polling exceeded timeout (server-controlled) |

---

## Advanced usage

### Streaming

`astream_governed` yields the original event stream while governance runs in the background:

```python
async for event in governed.astream_governed(
    {"messages": [{"role": "user", "content": "..."}]},
    config={"configurable": {"thread_id": "session-001"}},
    stream_mode="values",
):
    pass
```

### Multi-turn sessions

Pass a consistent `thread_id` across turns:

```python
config = {"configurable": {"thread_id": "user-42-session-7"}}

await governed.ainvoke({"messages": [{"role": "user", "content": "Hello"}]}, config=config)
await governed.ainvoke({"messages": [{"role": "user", "content": "Export the data"}]}, config=config)
```

### Subagent detection

For multi-agent systems, provide a `resolve_subagent_name` hook to identify subagent tool calls:

```python
def detect_subagent(event):
    if event.name == "delegate_to_researcher":
        return "researcher"
    return None

governed = create_openbox_graph_handler(
    graph=agent,
    api_url=os.environ["OPENBOX_URL"],
    api_key=os.environ["OPENBOX_API_KEY"],
    agent_name="MyAgent",
    resolve_subagent_name=detect_subagent,
)
```

When a subagent is detected, the SDK tags the governance event with `subagent_name` for execution tree tracking in the dashboard.

### `fail_closed` mode

For high-sensitivity agents, block all tool calls if OpenBox Core is unreachable:

```python
governed = create_openbox_graph_handler(
    graph=agent,
    on_api_error="fail_closed",
    ...
)
```

---

## Debugging

Set `OPENBOX_DEBUG=1` to log all governance requests and responses:

```bash
OPENBOX_DEBUG=1 python agent.py
```

Output:

```
[OpenBox Debug] governance request: { "event_type": "ActivityStarted", "activity_type": "search_web", ... }
[OpenBox Debug] governance response: { "verdict": "allow", ... }
```

---

## Contributing

```bash
git clone https://github.com/OpenBox-AI/openbox-langgraph-sdk-python
cd openbox-langgraph-sdk-python
uv sync --all-extras
uv run pytest
uv run ruff check openbox_langgraph/
```

---

## License

MIT
