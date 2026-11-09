"""Exposure-barrier fields reach Core on the direct-httpx evaluate path too.

The base SDK adds ``scope``, ``action_class``, ``resources`` and
``participants`` from the caller's context. GovernanceClient's own httpx path
must add the same fields, or the barrier check would see an action with no
labels (Unknown) for these deployments.
"""

from __future__ import annotations

import json

import httpx
import pytest

from openbox_langgraph.client import GovernanceClient

exposure = pytest.importorskip(
    "openbox_core.exposure", reason="openbox-sdk-python without exposure fields"
)


def _recorder(sent: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"verdict": "allow"})

    return httpx.MockTransport(handler)


async def test_evaluate_raw_carries_exposure_fields() -> None:
    sent: list[dict] = []
    client = GovernanceClient(api_url="https://core.test", api_key="obx_test_abc")
    client._client = httpx.AsyncClient(transport=_recorder(sent))
    doc = exposure.Resource("dms://doc/17", ["client:coke"], "source_system")
    with exposure.exposure_scope("matter:2026-0417"):
        with exposure.governed_action("retrieve", resources=[doc]):
            await client.evaluate_raw({"event_type": "ActivityStarted"})
    await client.evaluate_raw({"event_type": "ActivityStarted"})
    await client.close()
    assert sent[0]["scope"] == "matter:2026-0417"
    assert sent[0]["action_class"] == "retrieve"
    assert sent[0]["resources"] == [
        {"uri": "dms://doc/17", "labels": ["client:coke"], "label_source": "source_system"}
    ]
    assert "action_class" not in sent[1]
