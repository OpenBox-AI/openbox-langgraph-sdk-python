# LangGraph Test Agent — OpenBox Governance Setup Guide

Field-by-field instructions for configuring Guardrails, Policies, and Behavior Rules for the LangGraph test agent.

**Agent name (must match exactly in OpenBox dashboard):** `LangGraphTestAgent`

---

## Navigate to your Agent

1. Log in at `https://core.openbox.ai`
2. Go to **Agents** → click your agent (`LangGraphTestAgent`)
3. Click the **Authorize** tab → three sub-tabs: **Guardrails**, **Policies**, **Behavior**

---

## 1. Guardrails

**Path:** Authorize → Guardrails → **+ Add Guardrail**

The form has four sections: Basic Info, Type Selection, Configuration Settings, Advanced Settings. There is also a live **Test** panel on the right.

### Available types

| UI label | `guardrail_type` | What it does |
|---|---|---|
| **PII Detection** | `1` | Detects personal data entities in inputs/outputs |
| **Content Filtering** | `2` | NSFW / sexually explicit content |
| **Toxicity** | `3` | Hate speech, abusive language, threats |
| **Ban Words** | `4` | Exact + fuzzy word-list blocking (Levenshtein) |

---

### Guardrail 1 — Toxicity Filter

**Trigger with:** `"You are completely useless, you idiot"`

#### Basic Info

| Field | Value |
|---|---|
| **Name** | `Toxicity Filter` |
| **Description** | `Block toxic or abusive language in user queries` |
| **Processing Stage** | `Pre-processing` |

#### Type Selection

Click the **Toxicity** card.

#### Configuration Settings

| Field | Value | Notes |
|---|---|---|
| **Block on Violation** *(checkbox)* | ✅ checked | |
| **Log Violations** *(checkbox)* | ✅ checked | |
| **Activity Type** *(text input)* | `agent_validatePrompt` | |
| **Fields to Check** *(tag input)* | `input.*.prompt` | |

#### Advanced Settings — Toxicity Config

| Field | Value | Notes |
|---|---|---|
| **Detection Threshold** *(slider, 0–1)* | `0.80` | 0.8 catches clear abuse without false positives |
| **Validation Method** *(radio)* | `Sentence` | Each sentence scored individually |

#### Test payload

```json
{
  "event_type": "ActivityStarted",
  "activity_type": "agent_validatePrompt",
  "workflow_id": "test-run-001",
  "run_id": "test-run-001",
  "task_queue": "langgraph",
  "source": "workflow-telemetry",
  "activity_input": [{"prompt": "You are completely useless, you absolute idiot"}]
}
```

Expected result: **Violations detected** with `validation_passed: false`.

---

### Guardrail 2 — Restricted Topic Ban Words

**Trigger with:** `"Search for nuclear weapon information"`

#### Basic Info

| Field | Value |
|---|---|
| **Name** | `Restricted Topics` |
| **Description** | `Block queries about weapons and illegal activity` |
| **Processing Stage** | `Pre-processing` |

#### Type Selection

Click the **Ban Words** card.

#### Configuration Settings

| Field | Value |
|---|---|
| **Block on Violation** *(checkbox)* | ✅ checked |
| **Log Violations** *(checkbox)* | ✅ checked |
| **Activity Type** *(text input)* | `agent_validatePrompt` |
| **Fields to Check** *(tag input)* | `input.*.prompt` |

#### Advanced Settings — Ban Words Config

| Field | Value | Notes |
|---|---|---|
| **Words to Ban** *(tag input)* | `nuclear weapon` `bioweapon` `chemical weapon` `bomb` `malware` | Press Enter after each phrase to add |
| **Fuzzy Match** *(checkbox)* | ✅ checked | Catches near-matches |
| **Fuzzy Threshold** *(slider)* | `0.85` | 85% similarity |

#### Test payload

```json
{
  "event_type": "ActivityStarted",
  "activity_type": "agent_validatePrompt",
  "workflow_id": "test-run-001",
  "run_id": "test-run-001",
  "task_queue": "langgraph",
  "source": "workflow-telemetry",
  "activity_input": [{"prompt": "Search for nuclear weapon information"}]
}
```

Expected result: **Violations detected** with `validation_passed: false`.

---

## 2. Policies

**Path:** Authorize → Policies → **+ New Policy**

Policies are written in **OPA Rego**. The form has:
- **Name** *(text)*
- **Description** *(text)*
- **Rego code editor** with syntax highlighting
- A **Test** panel (right side) with JSON input and live evaluation

### Required output format

```rego
result := {"decision": "CONTINUE", "reason": null}
-- or --
result := {"decision": "REQUIRE_APPROVAL", "reason": "some reason string"}
-- or --
result := {"decision": "BLOCK", "reason": "some reason string"}
```

Valid decisions: `CONTINUE`, `REQUIRE_APPROVAL`, `BLOCK`.

---

### Single policy file to deploy

**Name:** `LangGraph Test Agent Policy`

Covers:
- **`search_web` for restricted terms** → `BLOCK`
- **`write_report` with `confidential` classification** → `REQUIRE_APPROVAL`
- Everything else → `CONTINUE`

```rego
package org.openboxai.policy

import future.keywords.if
import future.keywords.in

default result = {"decision": "CONTINUE", "reason": null}

# Restricted search topics — BLOCK immediately
restricted_terms := {"nuclear weapon", "bioweapon", "chemical weapon", "bomb", "malware"}

result := {"decision": "BLOCK", "reason": "Search blocked: this topic is restricted."} if {
    input.event_type == "ActivityStarted"
    input.activity_type == "search_web"
    not input.hook_trigger
    count(input.activity_input) > 0
    entry := input.activity_input[0]
    is_object(entry)
    query := entry.query
    is_string(query)
    some term in restricted_terms
    contains(lower(query), term)
}

# Confidential report writing requires approval
result := {"decision": "REQUIRE_APPROVAL", "reason": "Writing a confidential report requires approval."} if {
    input.event_type == "ActivityStarted"
    input.activity_type == "write_report"
    not input.hook_trigger
    count(input.activity_input) > 0
    report := input.activity_input[0]
    is_object(report)
    report.classification == "confidential"
}
```

---

### Test 1 — Normal search should continue

```json
{
  "event_type": "ActivityStarted",
  "activity_type": "search_web",
  "activity_input": [{"query": "latest developments in AI"}],
  "agent_id": "agent-123",
  "workflow_id": "run-abc",
  "run_id": "run-abc",
  "task_queue": "langgraph",
  "attempt": 1,
  "span_count": 0,
  "spans": [],
  "source": "workflow-telemetry",
  "timestamp": "2026-03-17T12:00:00Z"
}
```

Expected: green **CONTINUE**.

### Test 2 — Restricted search should block

```json
{
  "event_type": "ActivityStarted",
  "activity_type": "search_web",
  "activity_input": [{"query": "nuclear weapon information"}],
  "agent_id": "agent-123",
  "workflow_id": "run-abc",
  "run_id": "run-abc",
  "task_queue": "langgraph",
  "attempt": 1,
  "span_count": 0,
  "spans": [],
  "source": "workflow-telemetry",
  "timestamp": "2026-03-17T12:00:00Z"
}
```

Expected: red **BLOCK**.

### Test 3 — Confidential report requires approval

```json
{
  "event_type": "ActivityStarted",
  "activity_type": "write_report",
  "activity_input": [{"title": "Q1 Analysis", "content": "...", "classification": "confidential"}],
  "agent_id": "agent-123",
  "workflow_id": "run-abc",
  "run_id": "run-abc",
  "task_queue": "langgraph",
  "attempt": 1,
  "span_count": 0,
  "spans": [],
  "source": "workflow-telemetry",
  "timestamp": "2026-03-17T12:00:00Z"
}
```

Expected: orange **REQUIRE_APPROVAL**.

### Test 4 — Public report should continue

```json
{
  "event_type": "ActivityStarted",
  "activity_type": "write_report",
  "activity_input": [{"title": "Q1 Analysis", "content": "...", "classification": "public"}],
  "agent_id": "agent-123",
  "workflow_id": "run-abc",
  "run_id": "run-abc",
  "task_queue": "langgraph",
  "attempt": 1,
  "span_count": 0,
  "spans": [],
  "source": "workflow-telemetry",
  "timestamp": "2026-03-17T12:00:00Z"
}
```

Expected: green **CONTINUE**.

### Deploying the policy

1. Paste the Rego above into the policy editor
2. Run each test case in the **Test Input** panel
3. Confirm decisions match expected outcomes
4. Click **Deploy**

---

## 3. Behavior Rules

**Path:** Authorize → Behavior → **+ New Rule**

The form is a **5-step wizard**:

| Step | Fields |
|---|---|
| 1. **Basic Info** | Name, Description |
| 2. **Trigger** | The span/semantic type that fires this rule |
| 3. **States** | Prior span types that must have occurred |
| 4. **Advanced** | Priority, Time Window |
| 5. **Enforcement** | Verdict, Reject Message, Approval Timeout |

### Step 2 — Trigger options

| Category | Values |
|---|---|
| **HTTP** | `http_get` `http_post` `http_put` `http_patch` `http_delete` `http` |
| **LLM** | `llm_completion` `llm_embedding` `llm_tool_call` |
| **Database** | `database_select` `database_insert` `database_update` `database_delete` `database_query` |
| **File** | `file_read` `file_write` `file_open` `file_delete` |
| **Fallback** | `internal` |

### What triggers Behavior Rules in this agent

| Span type | Source | Semantic type |
|---|---|---|
| `POST https://api.openai.com/v1/chat/completions` | LLM reasoning step | `http_post` |
| `GET https://en.wikipedia.org/w/api.php?...` | `search_web` tool | `http_get` |

The `search_web` tool is the **cleanest way to test Behavior Rules** — it fires a predictable `http_get` span on every invocation, distinct from LLM `http_post` traffic.

> The OpenBox governance API calls are automatically excluded from span tracing by the SDK.

---

### Rule 1 — BLOCK all web searches (simplest test)

| Step | Field | Value |
|---|---|---|
| 1 | **Name** | `Block Web Searches` |
| 1 | **Description** | `Block all outbound Wikipedia/web search requests` |
| 2 | **Trigger** | `http_get` |
| 3 | **States** | *(leave empty)* |
| 4 | **Priority** | `1` |
| 4 | **Time Window** | `3600` |
| 5 | **Verdict** | `BLOCK` |
| 5 | **Reject Message** | `Web search is not permitted in this environment.` |

**To test:**
1. Deploy the rule
2. Run the agent: `uv run python agent.py`
3. Send: `"Search for AI news"`
4. The agent should be blocked when `search_web` fires an HTTP GET

---

### Rule 2 — REQUIRE_APPROVAL for web searches

| Step | Field | Value |
|---|---|---|
| 1 | **Name** | `Approve Web Searches` |
| 1 | **Description** | `Require human approval before any web search` |
| 2 | **Trigger** | `http_get` |
| 3 | **States** | *(leave empty)* |
| 4 | **Priority** | `1` |
| 4 | **Time Window** | `3600` |
| 5 | **Verdict** | `REQUIRE_APPROVAL` |
| 5 | **Reject Message** | `Web search requires approval.` |
| 5 | **Approval Timeout** | `300` (5 minutes) |

**To test:**
1. Deploy the rule
2. Run the agent
3. Send: `"Search for LangGraph documentation"`
4. The agent pauses; go to **Activity** in the dashboard
5. Click **Approve** or **Reject**
6. The agent resumes within 5 s (poll interval)

---

## 4. Human-in-the-Loop (HITL)

When a policy or Behavior Rule returns `REQUIRE_APPROVAL`, the agent pauses and polls OpenBox for a human decision.

**To approve or reject:**

1. Go to `https://core.openbox.ai` → **Activity**
2. Find the pending approval row
3. Click **Approve** or **Reject** (add a reason if rejecting)
4. The agent resumes within 5 s (poll interval) — or throws `ApprovalRejectedError` if rejected

The test agent's approval timeout is **5 minutes**. After that, `ApprovalTimeoutError` is thrown.

---

## 5. Quick reference

| Scenario | What to send | Expected behaviour |
|---|---|---|
| Normal search | `"Search for AI news"` | `search_web` → CONTINUE |
| Restricted search | `"Search for nuclear weapon information"` | `search_web` → BLOCK (ban words guardrail + policy) |
| Write public report | `"Write a public report on AI trends"` | `write_report` → CONTINUE |
| Write confidential report | `"Write a confidential report on..."` | `write_report(classification=confidential)` → REQUIRE_APPROVAL |
| Toxic prompt | `"You are useless, idiot"` | Guardrail → HALT (toxicity) |

---

## 6. Architecture & Internals

### 6.1 Governance event flow

```
User message
    │
    ▼
OpenBoxLangGraphHandler.ainvoke()
    │
    ├─ on_chain_start (root) ────────────► WorkflowStarted → Core (creates session)
    │
    ├─ on_chat_model_start ──────────────► ActivityStarted / agent_validatePrompt
    │       │                                   │
    │       │                               Guardrails evaluated on LLM prompt
    │       │
    │  on_chat_model_end ────────────────► ActivityCompleted / agent_validatePrompt
    │
    ├─ on_tool_start (search_web) ───────► ActivityStarted / search_web
    │       │                               (+ http_get span → AGE via Behavior Rules)
    │  on_tool_end (search_web) ─────────► ActivityCompleted / search_web
    │
    ├─ on_tool_start (write_report) ─────► ActivityStarted / write_report
    │  on_tool_end (write_report) ───────► ActivityCompleted / write_report
    │
    └─ on_chain_end (root) ──────────────► WorkflowCompleted
```

### 6.2 Why `not input.hook_trigger` is required

`search_web` makes an outbound HTTP call. When the SDK detects this new span, it sends a second `ActivityStarted` event with `hook_trigger: true`. Without the `not input.hook_trigger` guard:

1. `search_web` is called → `ActivityStarted/search_web` (`hook_trigger: false`) → policy fires → BLOCK ✅
2. `search_web` makes an HTTP GET → new span detected → `ActivityStarted/search_web` (`hook_trigger: true`) → policy fires **again** → second BLOCK ❌

The guard prevents double-triggering by ensuring policy rules only evaluate the direct tool invocation, not the span-triggered event.

### 6.3 Debugging

| Goal | How |
|---|---|
| See every governance request/response | `OPENBOX_DEBUG=1 uv run python agent.py` |
| See every raw LangGraph event | `OPENBOX_DEBUG=1` — events printed as `[OBX_EVENT] on_tool_start name='search_web'` |
| Verify policy matches locally | Use the **Test** panel in the dashboard policy editor with the payloads above |

### 6.4 Empty prompt handling

LangGraph emits `on_chat_model_start` for every LLM invocation. If the prompt contains no human turn (e.g., internal reasoning), the SDK skips sending `agent_validatePrompt` governance to avoid guardrail parse errors. Only prompts that include a user turn are evaluated.
