"""GovernanceClient public surface stays stable and base-result fields pass through.

The internal rewiring onto the base SDK's result types must not change the
public method set or signatures, must expose the added base fields
(fallback_used/diagnostics/raw), and must never let a base ("openbox_core")
exception class escape past the wrapper.
"""

from __future__ import annotations

import inspect

import httpx
import pytest

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.errors import OpenBoxError, OpenBoxNetworkError
from openbox_langgraph.types import GovernanceVerdictResponse, LangChainGovernanceEvent, Verdict


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


def test_public_methods_present() -> None:
    for name in (
        "evaluate_event",
        "evaluate_event_sync",
        "evaluate_raw",
        "poll_approval",
        "validate_api_key",
        "halt_response",
        "close",
    ):
        assert callable(getattr(GovernanceClient, name))


def test_init_signature_unchanged() -> None:
    params = set(inspect.signature(GovernanceClient.__init__).parameters)
    expected = {"api_url", "api_key", "timeout", "on_api_error", "agent_did", "agent_private_key"}
    assert expected <= params


def test_verdict_response_exposes_base_fields() -> None:
    r = GovernanceVerdictResponse(verdict=Verdict.ALLOW)
    assert r.fallback_used is False
    assert r.diagnostics == []
    assert r.raw == {}


@pytest.mark.asyncio
async def test_base_result_fields_pass_through() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "verdict": "allow",
                "policy_id": "p1",
                "risk_score": 0.4,
                "diagnostics": [{"k": "v"}],
            },
        )

    client = GovernanceClient(api_url="https://core.openbox.ai", api_key="obx_test_abc")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    resp = await client.evaluate_event(_event())
    await client.close()
    assert resp is not None
    assert resp.policy_id == "p1"
    assert resp.risk_score == 0.4
    assert resp.raw  # non-empty for a real parsed body
    assert resp.diagnostics == [{"k": "v"}]


@pytest.mark.asyncio
async def test_core_errors_do_not_leak_under_fail_closed() -> None:
    # A body the base parser chokes on, under fail_closed, must surface as a
    # LangGraph OpenBoxError subclass — never a raw core/JSON exception.
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    client = GovernanceClient(
        api_url="https://core.openbox.ai", api_key="obx_test_abc", on_api_error="fail_closed"
    )
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(OpenBoxNetworkError) as exc:
        await client.evaluate_event(_event())
    await client.close()
    assert isinstance(exc.value, OpenBoxError)
