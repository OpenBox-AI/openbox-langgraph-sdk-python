"""Fail-open must NEVER swallow a real BLOCK verdict (trust-boundary regression).

`GovernanceClient` returns ``None`` to callers (the fail-open signal) ONLY for a
client-synthesized network fallback — verdict ALLOW + ``fallback_used`` + empty
``raw``. A REAL response body that carries ``fallback_used: true`` alongside a
blocking verdict must STILL enforce: keying fail-open on the wire flag alone
would flip BLOCK -> ALLOW at the trust boundary.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import httpx
import pytest

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.errors import OpenBoxNetworkError
from openbox_langgraph.types import LangChainGovernanceEvent, Verdict

_Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


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


def _client(handler: _Handler, on_api_error: str = "fail_open") -> GovernanceClient:
    c = GovernanceClient(
        api_url="https://core.openbox.ai", api_key="obx_test_abc", on_api_error=on_api_error
    )
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


@pytest.mark.asyncio
async def test_block_body_with_fallback_used_flag_still_blocks() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"verdict": "block", "fallback_used": True, "reason": "no"})

    client = _client(handler)
    resp = await client.evaluate_event(_event())
    await client.close()
    assert resp is not None, "a real BLOCK body must not fail-open to None"
    assert resp.verdict == Verdict.BLOCK
    assert resp.fallback_used is True  # flag preserved but does NOT downgrade the verdict


@pytest.mark.asyncio
async def test_real_allow_body_is_returned_not_collapsed() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"verdict": "allow"})

    client = _client(handler)
    resp = await client.evaluate_event(_event())
    await client.close()
    assert resp is not None
    assert resp.verdict == Verdict.ALLOW
    assert resp.fallback_used is False


@pytest.mark.asyncio
async def test_http_error_fail_open_returns_none() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={})

    client = _client(handler, on_api_error="fail_open")
    resp = await client.evaluate_event(_event())
    await client.close()
    assert resp is None  # client-synthesized fallback


@pytest.mark.asyncio
async def test_http_error_fail_closed_raises_langgraph_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={})

    client = _client(handler, on_api_error="fail_closed")
    with pytest.raises(OpenBoxNetworkError):
        await client.evaluate_event(_event())
    await client.close()


@pytest.mark.asyncio
async def test_transport_exception_fail_open_returns_none() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable")

    client = _client(handler, on_api_error="fail_open")
    resp = await client.evaluate_event(_event())
    await client.close()
    assert resp is None


@pytest.mark.asyncio
async def test_transport_exception_fail_closed_raises_langgraph_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable")

    client = _client(handler, on_api_error="fail_closed")
    with pytest.raises(OpenBoxNetworkError):
        await client.evaluate_event(_event())
    await client.close()
