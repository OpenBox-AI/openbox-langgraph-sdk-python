"""Tests for signed OpenBox governance HTTP requests."""

from __future__ import annotations

import base64
import hashlib
import json
from unittest.mock import MagicMock

import httpx
import pytest

from openbox_langgraph import hook_governance
from openbox_langgraph.client import ApprovalPollParams, GovernanceClient
from openbox_langgraph.config import get_global_config, initialize
from openbox_langgraph.identity import (
    OPENBOX_AGENT_DID_HEADER,
    OPENBOX_BODY_SHA256_HEADER,
)

PRIVATE_KEY = base64.b64encode(bytes(range(32))).decode("ascii")
DID = "did:aip:550e8400-e29b-41d4-a716-446655440000"


@pytest.mark.asyncio
async def test_governance_client_signs_validate_request() -> None:
    """Validate endpoint receives AIP headers and an empty body hash."""
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    client = GovernanceClient(
        api_url="https://core.openbox.ai",
        api_key="obx_test_abc",
        agent_did=DID,
        agent_private_key=PRIVATE_KEY,
    )
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    await client.validate_api_key()
    await client.close()

    assert seen[0].headers[OPENBOX_AGENT_DID_HEADER] == DID
    assert seen[0].headers[OPENBOX_BODY_SHA256_HEADER] == hashlib.sha256(b"").hexdigest()


@pytest.mark.asyncio
async def test_governance_client_signs_exact_evaluate_body() -> None:
    """Evaluate requests sign the same serialized bytes sent over HTTP."""
    seen_bodies: list[bytes] = []
    seen_headers: list[httpx.Headers] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        seen_bodies.append(body)
        seen_headers.append(request.headers)
        return httpx.Response(200, json={"verdict": "allow"})

    client = GovernanceClient(
        api_url="https://core.openbox.ai",
        api_key="obx_test_abc",
        agent_did=DID,
        agent_private_key=PRIVATE_KEY,
    )
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    await client.evaluate_raw({"z": 1, "message": "hello"})
    await client.close()

    assert json.loads(seen_bodies[0]) == {"z": 1, "message": "hello"}
    assert seen_headers[0][OPENBOX_BODY_SHA256_HEADER] == hashlib.sha256(seen_bodies[0]).hexdigest()


@pytest.mark.asyncio
async def test_governance_client_signs_approval_body() -> None:
    """HITL approval polling sends signed exact body bytes."""
    seen_bodies: list[bytes] = []
    seen_headers: list[httpx.Headers] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        seen_bodies.append(body)
        seen_headers.append(request.headers)
        return httpx.Response(200, json={"verdict": "allow"})

    client = GovernanceClient(
        api_url="https://core.openbox.ai",
        api_key="obx_test_abc",
        agent_did=DID,
        agent_private_key=PRIVATE_KEY,
    )
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    await client.poll_approval(
        ApprovalPollParams(workflow_id="workflow-1", run_id="run-1", activity_id="activity-1")
    )
    await client.close()

    assert json.loads(seen_bodies[0]) == {
        "workflow_id": "workflow-1",
        "run_id": "run-1",
        "activity_id": "activity-1",
    }
    assert seen_headers[0][OPENBOX_BODY_SHA256_HEADER] == hashlib.sha256(seen_bodies[0]).hexdigest()


def test_initialize_reads_agent_identity_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Global config resolves DID config from environment."""
    monkeypatch.setenv("OPENBOX_AGENT_DID", DID)
    monkeypatch.setenv("OPENBOX_AGENT_PRIVATE_KEY", PRIVATE_KEY)

    initialize(
        api_url="https://core.openbox.ai",
        api_key="obx_test_abc",
        validate=False,
    )

    config = get_global_config()
    assert config.agent_did == DID
    assert config.agent_private_key == PRIVATE_KEY


def test_hook_governance_signs_exact_body() -> None:
    """Hook-level governance uses the same signed request path as normal events."""
    seen_bodies: list[bytes] = []
    seen_headers: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        seen_bodies.append(body)
        seen_headers.append(request.headers)
        return httpx.Response(200, json={"verdict": "allow"})

    span = MagicMock()
    span_context = MagicMock()
    span_context.trace_id = 123
    span_context.span_id = 456
    span.get_span_context.return_value = span_context

    processor = MagicMock()
    processor.get_activity_context_by_trace.return_value = {
        "workflow_id": "workflow-1",
        "run_id": "run-1",
        "activity_id": "activity-1",
        "event_type": "ActivityStarted",
    }

    hook_governance.configure(
        "https://core.openbox.ai",
        "obx_test_abc",
        processor,
        agent_did=DID,
        agent_private_key=PRIVATE_KEY,
    )
    hook_governance._sync_client = httpx.Client(transport=httpx.MockTransport(handler))

    hook_governance.evaluate_sync(span, "https://example.com", {"hook_type": "http"})

    assert json.loads(seen_bodies[0])["hook_trigger"] is True
    assert seen_headers[0][OPENBOX_AGENT_DID_HEADER] == DID
    assert seen_headers[0][OPENBOX_BODY_SHA256_HEADER] == hashlib.sha256(seen_bodies[0]).hexdigest()
