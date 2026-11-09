"""The on-behalf-of user reaches Core on the direct-httpx evaluate path too.

With neither a gate nor a core EvaluationClient, GovernanceClient posts to
/api/v1/governance/evaluate itself. That path must add the same
``on_behalf_of`` the base SDK adds, or information barriers would see only
the agent for these deployments.
"""

from __future__ import annotations

import json

import httpx
import pytest

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.types import LangChainGovernanceEvent

on_behalf_of = pytest.importorskip(
    "openbox_core.on_behalf_of", reason="openbox-sdk-python without on_behalf_of"
).on_behalf_of

AMY = {"user": "amy@firm.example", "source": "sdk"}


def _event() -> LangChainGovernanceEvent:
    return LangChainGovernanceEvent(
        source="workflow-telemetry",
        event_type="ToolStarted",
        workflow_id="wf-1",
        run_id="run-1",
        workflow_type="Agent",
        task_queue="langgraph",
        timestamp="2026-11-05T00:00:00Z",
        activity_id="act-1",
        activity_type="tool_call",
    )


def _recorder(sent: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"verdict": "allow"})

    return httpx.MockTransport(handler)


def _client() -> GovernanceClient:
    return GovernanceClient(api_url="https://core.test", api_key="obx_test_abc")


async def test_evaluate_raw_carries_the_user() -> None:
    sent: list[dict] = []
    client = _client()
    client._client = httpx.AsyncClient(transport=_recorder(sent))
    with on_behalf_of("amy@firm.example"):
        await client.evaluate_raw({"event_type": "ActivityStarted"})
    await client.evaluate_raw({"event_type": "ActivityStarted"})
    await client.close()
    assert sent[0]["on_behalf_of"] == AMY
    assert "on_behalf_of" not in sent[1]


async def test_evaluate_event_carries_the_user() -> None:
    sent: list[dict] = []
    client = _client()
    client._client = httpx.AsyncClient(transport=_recorder(sent))
    with on_behalf_of("amy@firm.example"):
        await client.evaluate_event(_event())
    await client.close()
    assert sent[0]["on_behalf_of"] == AMY


def test_evaluate_event_sync_carries_the_user() -> None:
    sent: list[dict] = []
    client = _client()
    client._sync_client = httpx.Client(transport=_recorder(sent))
    with on_behalf_of("amy@firm.example"):
        client.evaluate_event_sync(_event())
    assert sent[0]["on_behalf_of"] == AMY


async def test_a_value_already_on_the_payload_wins() -> None:
    sent: list[dict] = []
    client = _client()
    client._client = httpx.AsyncClient(transport=_recorder(sent))
    explicit = {"user": "colin@firm.example", "source": "caller"}
    with on_behalf_of("amy@firm.example"):
        await client.evaluate_raw({"event_type": "ActivityStarted", "on_behalf_of": explicit})
    await client.close()
    assert sent[0]["on_behalf_of"] == explicit
