"""Layer 1 hook-trigger golden capture: drives hook_governance.evaluate_sync directly.

Separate module from real_emitted_event_fixtures.py because this path goes
through hook_governance (module-level singleton config + sync httpx.Client)
rather than GovernanceClient.evaluate_event, and needs its own MagicMock
span/processor setup.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx

from openbox_langgraph import hook_governance

from .capture_harness import (
    TEST_API_KEY,
    TEST_API_URL,
    TEST_DID,
    TEST_PRIVATE_KEY,
    write_fixture_pair,
)


async def capture_hook_trigger_fixture() -> None:
    """Capture the hook-trigger wire body — must carry hook_trigger:true + non-empty spans."""
    seen_bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        seen_bodies.append(body)
        return httpx.Response(200, json={"verdict": "allow"})

    span = MagicMock()
    span_context = MagicMock()
    span_context.trace_id = 987654321
    span_context.span_id = 123456789
    span.get_span_context.return_value = span_context

    processor = MagicMock()
    processor.get_activity_context_by_trace.return_value = {
        "workflow_id": "golden-workflow-1",
        "run_id": "golden-run-1",
        "activity_id": "golden-activity-1",
        "event_type": "ActivityStarted",
    }

    hook_governance.configure(
        TEST_API_URL,
        TEST_API_KEY,
        processor,
        agent_did=TEST_DID,
        agent_private_key=TEST_PRIVATE_KEY,
    )
    hook_governance._sync_client = httpx.Client(transport=httpx.MockTransport(handler))

    hook_governance.evaluate_sync(
        span,
        "https://example.com/api/resource",
        {"hook_type": "http", "method": "GET", "url": "https://example.com/api/resource"},
    )

    body = json.loads(seen_bodies[0])
    if body.get("hook_trigger") is not True:
        msg = "hook body must carry hook_trigger: true"
        raise AssertionError(msg)
    if not body.get("spans"):
        msg = "hook body must carry non-empty spans"
        raise AssertionError(msg)
    write_fixture_pair("hook_trigger", body)
