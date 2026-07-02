"""evaluate_event signs the EXACT bytes it sends, with byte-correct AIP headers.

The signing path is unchanged by the base-SDK rewiring; this pins the lifecycle
evaluate route specifically (test_did_client_signing covers validate/approval).
"""

from __future__ import annotations

import base64
import hashlib

import httpx
import pytest

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.identity import OPENBOX_AGENT_DID_HEADER, OPENBOX_BODY_SHA256_HEADER
from openbox_langgraph.types import LangChainGovernanceEvent

_PRIVATE_KEY = base64.b64encode(bytes(range(32))).decode("ascii")
_DID = "did:aip:550e8400-e29b-41d4-a716-446655440000"


def _event() -> LangChainGovernanceEvent:
    return LangChainGovernanceEvent(
        source="workflow-telemetry",
        event_type="ToolStarted",
        workflow_id="wf-1",
        run_id="run-1",
        workflow_type="Agent",
        task_queue="langgraph",
        timestamp="2026-07-02T00:00:00Z",
        activity_id="act-1",
        activity_type="tool_call",
    )


@pytest.mark.asyncio
async def test_signed_evaluate_hashes_the_sent_body() -> None:
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = await request.aread()
        seen["headers"] = request.headers
        return httpx.Response(200, json={"verdict": "allow"})

    client = GovernanceClient(
        api_url="https://core.openbox.ai",
        api_key="obx_test_abc",
        agent_did=_DID,
        agent_private_key=_PRIVATE_KEY,
    )
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await client.evaluate_event(_event())
    await client.close()

    headers = seen["headers"]
    assert headers[OPENBOX_AGENT_DID_HEADER] == _DID  # type: ignore[index]
    assert headers[OPENBOX_BODY_SHA256_HEADER] == hashlib.sha256(seen["body"]).hexdigest()  # type: ignore[index,arg-type]


@pytest.mark.asyncio
async def test_unsigned_evaluate_emits_no_identity_headers() -> None:
    seen: dict[str, httpx.Headers] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        return httpx.Response(200, json={"verdict": "allow"})

    client = GovernanceClient(api_url="https://core.openbox.ai", api_key="obx_test_abc")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await client.evaluate_event(_event())
    await client.close()

    assert OPENBOX_AGENT_DID_HEADER not in seen["headers"]
