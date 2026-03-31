# OpenBox LangGraph SDK — Code Standards

## Overview

This document establishes coding conventions, architectural patterns, and quality standards for openbox_langgraph development.

**Principles:** YAGNI (You Aren't Gonna Need It), KISS (Keep It Simple), DRY (Don't Repeat Yourself)

## File Organization

### Naming Conventions
- **Python modules:** kebab-case, descriptive (e.g., `db_governance_hooks.py`, `span_processor.py`)
- **Classes:** PascalCase (e.g., `GovernanceClient`, `WorkflowSpanProcessor`)
- **Functions/Methods:** snake_case (e.g., `evaluate_event()`, `_extract_governance_blocked()`)
- **Constants:** UPPER_SNAKE_CASE (e.g., `DEFAULT_GOVERNANCE_CONFIG`, `_API_KEY_PATTERN`)
- **Private members:** leading underscore (e.g., `_client`, `_dedup_run`)

### Module Structure
```
openbox_langgraph/
├── __init__.py              # Public API exports (45+ symbols)
├── config.py                # Configuration & global singleton
├── types.py                 # Core types & enums
├── errors.py                # Exception hierarchy
├── client.py                # GovernanceClient HTTP interface
├── langgraph_handler.py     # Main handler (entry point)
├── verdict_handler.py       # Verdict enforcement logic
├── hitl.py                  # Human-in-the-loop polling
├── tracing.py               # Tracing utilities
├── otel_setup.py            # Instrumentation initialization
├── span_processor.py        # Activity context tracking
├── hook_governance.py       # Unified hook evaluation
├── http_governance_hooks.py # HTTP interception
├── db_governance_hooks.py   # Database interception
└── file_governance_hooks.py # File I/O interception
```

### Size Limits
- **Module max:** 1,600 LOC (split if larger)
- **Class max:** 400 LOC (extract methods/inner classes)
- **Function max:** 50 LOC (refactor if larger, except event handlers)
- **Dataclass max:** 30 fields

**Rationale:** Optimal for code review, testing, and LLM context windows.

## Type Hints & Type Checking

### Strict mypy (non-negotiable)
```bash
mypy openbox_langgraph/ --strict
```

**Rules:**
1. All public functions require type hints
2. All return types required (no implicit None)
3. `from __future__ import annotations` at top of every module
4. Use `|` for unions, not `Union`
5. Use `list[T]`, `dict[K, V]`, not `List[T]`, `Dict[K, V]`
6. Any imports/casts must have a comment explaining why

**Exception:** `Any` only with docstring justification
```python
def evaluate_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
    """Send pre-built payload (hook calls pre-build, no schema validation needed)."""
```

### Type Alias Conventions
```python
# Top of module after imports
ServerEventType = Literal["WorkflowStarted", "ActivityStarted", ...]
HookPayload = dict[str, Any]
```

## Async/Await Patterns

### Rule: Async-First
All I/O is async. Sync wrappers exist only for middleware hooks and synchronous contexts.

**Async Pattern:**
```python
async def evaluate_event(self, event: LangChainGovernanceEvent) -> GovernanceVerdictResponse | None:
    """Send event to OpenBox Core (async)."""
    client = self._get_client()  # Lazy-init persistent httpx.AsyncClient
    response = await client.post(...)
    return parse_governance_response(response.json())
```

**Sync Wrapper (for middleware hooks):**
```python
def evaluate_event_sync(self, event: LangChainGovernanceEvent) -> GovernanceVerdictResponse | None:
    """Synchronous wrapper for sync middleware contexts (no asyncio.run)."""
    try:
        loop = asyncio.get_running_loop()
        # Already in async context — schedule as task, don't block
        return loop.create_task(self.evaluate_event(event))
    except RuntimeError:
        # No event loop — use asyncio.run (careful with cleanup)
        return asyncio.run(self.evaluate_event(event))
```

### ContextVar Usage
- Use ContextVar for thread-safe, async-task-safe data (e.g., `_workflow_context`)
- Propagate explicitly across asyncio boundaries (context.attach/detach)
- Never rely on task-local storage across Task spawns

**Example:**
```python
_workflow_context: ContextVar[WorkflowSpanBuffer | None] = ContextVar(
    "_workflow_context", default=None
)

def set_activity_context(buf: WorkflowSpanBuffer) -> None:
    """Set activity context for current async task."""
    _workflow_context.set(buf)

def get_activity_context() -> WorkflowSpanBuffer | None:
    """Get activity context for current async task."""
    return _workflow_context.get()
```

## Error Handling

### Exception Hierarchy (errors.py)
- **OpenBoxError** — base class for all SDK errors
- **OpenBoxAuthError** — invalid API key
- **OpenBoxNetworkError** — OpenBox Core unreachable
- **OpenBoxInsecureURLError** — HTTP on non-localhost
- **GovernanceBlockedError** — BLOCK/HALT verdict (hook-level and SDK-level signatures)
- **GovernanceHaltError** — workflow-level HALT
- **GuardrailsValidationError** — guardrails failed
- **ApprovalExpiredError** — HITL approval window expired
- **ApprovalRejectedError** — user explicitly rejected
- **ApprovalTimeoutError** — HITL polling timeout

### Try-Catch Pattern
```python
try:
    response = await client.post(url, headers=headers)
    if response.status_code in (401, 403):
        raise OpenBoxAuthError("Invalid API key...")
    if not response.is_success:
        raise OpenBoxNetworkError(f"HTTP {response.status_code}")
except (OpenBoxAuthError, OpenBoxNetworkError):
    raise  # Re-raise SDK errors
except Exception as e:  # noqa: BLE001
    raise OpenBoxNetworkError(f"Cannot reach Core: {e}") from e
```

### Unwrapping Exceptions
LLM SDKs (OpenAI, Anthropic) wrap httpx errors. Walk the chain:
```python
def _extract_governance_blocked(exc: Exception) -> GovernanceBlockedError | None:
    """Walk __cause__/__context__ to find wrapped GovernanceBlockedError."""
    cause = exc
    seen = set()
    while cause is not None:
        if id(cause) in seen:
            break
        seen.add(id(cause))
        if isinstance(cause, GovernanceBlockedError):
            return cause
        cause = getattr(cause, '__cause__', None) or getattr(cause, '__context__', None)
    return None
```

## Code Patterns

### Dataclass Patterns
```python
from dataclasses import dataclass, field

@dataclass
class GovernanceConfig:
    """Configuration dataclass (fields with defaults must come after required fields)."""

    # No required fields (all have defaults)
    on_api_error: str = "fail_open"
    api_timeout: float = 30.0
    skip_chain_types: set[str] = field(default_factory=set)
    tool_type_map: dict[str, str] = field(default_factory=dict)
```

### Lazy Imports
For compatibility (sandbox restrictions, circular imports):
```python
def _get_logger():
    """Lazy logger import."""
    import logging
    return logging.getLogger(__name__)

def validate_url_security(api_url: str) -> None:
    """Lazy import urllib to avoid sandbox issues."""
    from urllib.parse import urlparse
    parsed = urlparse(api_url)
    ...
```

### Singleton Pattern (Global Config)
```python
_global_config = _GlobalConfigState()

def initialize(api_url: str, api_key: str, governance_timeout: float = 30.0) -> None:
    """Initialize global config (call once at startup)."""
    _global_config.configure(api_url, api_key, governance_timeout)

def get_global_config() -> _GlobalConfigState:
    """Get global config (read-only after init)."""
    return _global_config
```

### Persistent Client Pattern
```python
class GovernanceClient:
    def __init__(self, *, api_url: str, api_key: str, timeout: float = 30.0) -> None:
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None  # Lazy-init

    def _get_client(self) -> httpx.AsyncClient:
        """Return or create persistent async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def close(self) -> None:
        """Close underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
```

### Deduplication Pattern
```python
def __init__(self) -> None:
    self._dedup_run: tuple[str, str] | None = None
    self._dedup_sent: set[tuple[str, str]] = set()

def _is_duplicate(self, workflow_id: str, run_id: str, activity_id: str, event_type: str) -> bool:
    """Check dedup. Resets when (workflow_id, run_id) changes."""
    current_run = (workflow_id, run_id)
    if self._dedup_run != current_run:
        self._dedup_run = current_run
        self._dedup_sent = set()
    key = (activity_id, event_type)
    if key in self._dedup_sent:
        return True
    self._dedup_sent.add(key)
    return False
```

## Testing Standards

### Test File Naming
- `test_governance_changes.py` — behavior change tests
- `test_telemetry_payload.py` — schema/payload validation
- `test_contextvars_propagation.py` — async/sync context behavior

### Test Markers
```python
# End-to-end tests
@pytest.mark.requires_openbox_url
async def test_full_governance_flow():
    ...

# Unit tests
async def test_client_dedup():
    ...
```

### Async Test Pattern
```python
import pytest
from pytest_asyncio import fixture

@fixture
async def client():
    """Fixture with async setup/teardown."""
    c = GovernanceClient(api_url="...", api_key="...")
    yield c
    await c.close()

@pytest.mark.asyncio
async def test_evaluate_event(client):
    """Test async method."""
    event = LangChainGovernanceEvent(...)
    response = await client.evaluate_event(event)
    assert response is not None
```

### Mocking Pattern
```python
from unittest.mock import AsyncMock, MagicMock, patch

async def test_network_error_fail_open():
    """Test fail_open behavior."""
    with patch.object(httpx.AsyncClient, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.ConnectError("...")
        client = GovernanceClient(api_url="...", api_key="...", on_api_error="fail_open")
        result = await client.evaluate_event(event)
        assert result is None  # fail_open returns None
```

## Logging & Debugging

### Logging Pattern
```python
import logging
_logger = logging.getLogger(__name__)

async def evaluate_event(self, event: LangChainGovernanceEvent) -> GovernanceVerdictResponse | None:
    """Send governance event."""
    if os.environ.get("OPENBOX_DEBUG") == "1":
        _logger.debug(f"Evaluating event: {event.event_type}")
    response = await self._client.post(...)
    if response.status_code == 200:
        _logger.info(f"Verdict: {response.json()['verdict']}")
    return parse_governance_response(response.json())
```

### Debug Environment Variable
```bash
export OPENBOX_DEBUG=1
python my_agent.py  # Verbose output
```

## Documentation Standards

### Module Docstring
```python
"""OpenBox LangGraph SDK — Core types mirroring sdk-langgraph/src/types.ts.

This module defines:
- Verdict enum (5-tier graduated response)
- Event types (LangChain, workflow, server)
- Response parsing and utilities
"""
```

### Function Docstring (Google Style)
```python
async def evaluate_event(
    self, event: LangChainGovernanceEvent
) -> GovernanceVerdictResponse | None:
    """Send a governance event to OpenBox Core and return the verdict.

    Returns `None` on network failure when `on_api_error` is `fail_open`.
    Silently drops duplicate (activity_id, event_type) pairs within the same run.

    Args:
        event: The governance event payload to evaluate.

    Returns:
        Verdict response from Core, or None if network error and fail_open.

    Raises:
        OpenBoxNetworkError: On network failure when `on_api_error` is `fail_closed`.
    """
```

### Inline Comments
Only for non-obvious logic:
```python
# Hook guard: Rego policies use `not input.hook_trigger` to prevent re-evaluation.
# We mark hook_trigger=True here so Rego can skip double-evaluation.
payload["hook_trigger"] = True
```

## Linting & Formatting

### Ruff Configuration (pyproject.toml)
```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "C4", "PIE", "RUF"]
```

### Pre-Commit Check
```bash
ruff check openbox_langgraph/ --fix
mypy openbox_langgraph/ --strict
```

### Import Ordering
```python
from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

import httpx
from langchain_core.callbacks import AsyncCallbackHandler

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.config import GovernanceConfig
from openbox_langgraph.errors import OpenBoxError
```

## API Design

### Public API Exports (__init__.py)
Only export what users need:
```python
__all__ = [
    # Errors
    "OpenBoxError",
    "GovernanceBlockedError",
    # Config
    "GovernanceConfig",
    "initialize",
    # Types
    "Verdict",
    # Main Handler
    "OpenBoxLangGraphHandler",
    "create_openbox_graph_handler",
    # Tracing
    "setup_opentelemetry_for_governance",
    "traced",
    "create_span",
]
```

### Function Signature Stability
Once public, maintain backward compatibility:
- New parameters must have defaults
- Rename via deprecation warning
- Don't remove or reorder positional args

**Example:**
```python
def create_openbox_graph_handler(
    graph: RunnableWithToolChoice,
    *,
    api_url: str,
    api_key: str,
    agent_name: str,
    config: dict[str, Any] | None = None,
    # New parameter with default (backward compatible)
    sqlalchemy_engine: Any = None,
) -> OpenBoxLangGraphHandler:
    ...
```

## Validation

### Input Validation
```python
def validate_api_key_format(api_key: str) -> bool:
    """Return True if API key matches expected format (obx_live_* or obx_test_*)."""
    return bool(_API_KEY_PATTERN.match(api_key))

def validate_url_security(api_url: str) -> None:
    """Raise OpenBoxInsecureURLError if URL uses HTTP on non-localhost."""
    from urllib.parse import urlparse
    parsed = urlparse(api_url)
    is_localhost = parsed.hostname in ("localhost", "127.0.0.1", "::1")
    if parsed.scheme == "http" and not is_localhost:
        raise OpenBoxInsecureURLError(f"Insecure HTTP URL: {api_url}")
```

## Performance Considerations

### Lazy Initialization
```python
# Client is created on first use, not on handler init
def _get_client(self) -> httpx.AsyncClient:
    if self._client is None or self._client.is_closed:
        self._client = httpx.AsyncClient(timeout=self._timeout)
    return self._client
```

### Memory Management
- Dedup resets per run (not unbounded)
- ContextVar cleanup on handler close
- Span processor does not accumulate spans (processor pattern)

### Timeout Handling
```python
# All API calls have timeout (default 30s)
client = httpx.AsyncClient(timeout=30.0)
response = await client.post(url, json=payload)  # Uses client timeout
```

## Security Best Practices

1. **API Key Handling:**
   - Validate format on init
   - Use Bearer token in Authorization header
   - Never log API keys
   - Use lazy imports to avoid early evaluation

2. **URL Validation:**
   - Reject insecure HTTP on non-localhost
   - Always verify HTTPS for production URLs

3. **Input Sanitization:**
   - Event payloads serialized with safe_serialize() (no code execution)
   - Activity input/output truncated if too large
   - No eval() or exec() anywhere

4. **Error Messages:**
   - Don't expose API keys in exceptions
   - Don't include full payloads in logs (truncate)
   - Safe exception chain unwrapping

## Release & Versioning

**Semantic Versioning:** MAJOR.MINOR.PATCH

- **MAJOR:** Breaking API changes
- **MINOR:** New features (backward compatible)
- **PATCH:** Bug fixes

**Current Version:** 0.1.0 (beta, API may change)

**Changelog:** Maintain in plans/reports/ and project-roadmap.md
