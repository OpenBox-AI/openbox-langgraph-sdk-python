"""Reusable helpers for capturing real OpenBox LangGraph SDK wire payloads.

This module is the baseline oracle for wire-payload parity: it drives the
REAL `GovernanceClient` / `hook_governance` / `OpenBoxLangGraphHandler` code
paths against a mock HTTP transport and records exactly what goes on the
wire today. Nothing here simulates governance behavior — only the transport
is faked, matching the existing test suite's convention (see
tests/test_did_client_signing.py).

Future parity tests (post-refactor) can import these helpers to replay the
same captures and diff against the fixtures in tests/golden/*.json.
"""

from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.types import LangChainGovernanceEvent

from .id_normalization import normalize_json

# Test-only signing material — NOT a real OpenBox agent credential.
# Mirrors the constant used across tests/test_did_client_signing.py.
TEST_PRIVATE_KEY = base64.b64encode(bytes(range(32))).decode("ascii")
TEST_DID = "did:aip:550e8400-e29b-41d4-a716-446655440000"
TEST_API_URL = "https://core.openbox.ai"
TEST_API_KEY = "obx_test_abc"

GOLDEN_DIR = Path(__file__).parent


@dataclass
class RecordedRequest:
    """A single HTTP request captured by the mock transport."""

    method: str
    url: str
    headers: httpx.Headers
    body: bytes

    @property
    def json_body(self) -> dict[str, Any]:
        """Parse the captured body as JSON."""
        result: dict[str, Any] = json.loads(self.body)
        return result


@dataclass
class RequestRecorder:
    """Collects every request that passes through the mock transport."""

    requests: list[RecordedRequest] = field(default_factory=list)

    async def async_handler(self, request: httpx.Request) -> httpx.Response:
        """MockTransport handler for httpx.AsyncClient — records then allows."""
        body = await request.aread()
        self.requests.append(
            RecordedRequest(
                method=request.method, url=str(request.url), headers=request.headers, body=body
            )
        )
        return httpx.Response(200, json={"verdict": "allow"})

    def sync_handler(self, request: httpx.Request) -> httpx.Response:
        """MockTransport handler for httpx.Client — records then allows."""
        body = request.read()
        self.requests.append(
            RecordedRequest(
                method=request.method, url=str(request.url), headers=request.headers, body=body
            )
        )
        return httpx.Response(200, json={"verdict": "allow"})


def build_recording_client(
    *, signed: bool, recorder: RequestRecorder | None = None
) -> tuple[GovernanceClient, RequestRecorder]:
    """Build a `GovernanceClient` wired to a recording mock transport.

    Args:
        signed: If True, configure the client with the test AIP identity so
            all five X-OpenBox-Agent-* / X-OpenBox-Body-SHA256 headers are sent.
        recorder: Optional existing recorder to reuse (e.g. across multiple
            calls on the same client instance).
    """
    recorder = recorder or RequestRecorder()
    client = GovernanceClient(
        api_url=TEST_API_URL,
        api_key=TEST_API_KEY,
        agent_did=TEST_DID if signed else None,
        agent_private_key=TEST_PRIVATE_KEY if signed else None,
    )
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(recorder.async_handler))
    return client, recorder


# Filename marker for fixtures that could NOT be produced by driving the real
# handler (grep-verified: no construction site exists, or the config that
# would send it also suppresses it) — kept as a hand-authored pin of the
# client's serialization only, clearly distinguished from real-emitted bodies.
HANDBUILT_PIN_LABEL = "handbuilt_serialization_pin"


def write_fixture_pair(name: str, raw: dict[str, Any], *, handbuilt_pin: bool = False) -> None:
    """Write `{name}.raw.json` and `{name}.normalized.json` under tests/golden/.

    When `handbuilt_pin` is True, both filenames additionally carry the
    `.handbuilt_serialization_pin` marker (e.g.
    `chain_started.handbuilt_serialization_pin.raw.json`) so it's obvious at
    a glance — including to a future parity gate — that this fixture pins a
    hand-authored event's serialization, not a real handler emission.
    """
    stem = f"{name}.{HANDBUILT_PIN_LABEL}" if handbuilt_pin else name
    raw_path = GOLDEN_DIR / f"{stem}.raw.json"
    normalized_path = GOLDEN_DIR / f"{stem}.normalized.json"
    raw_path.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n")
    normalized_path.write_text(
        json.dumps(normalize_json(raw), sort_keys=True, indent=2) + "\n"
    )


def write_ordering_fixture(name: str, ordered: list[dict[str, Any]]) -> None:
    """Write a normalized ordered-capture fixture (list of {event_type, activity_id}).

    activity_id values are generated per-run (thread/uuid-derived) — normalized
    the same way as wire-body fixtures so regenerating produces a byte-identical
    file. event_type values are stable literals and pass through unchanged.
    """
    path = GOLDEN_DIR / f"{name}.json"
    path.write_text(json.dumps(normalize_json(ordered), sort_keys=True, indent=2) + "\n")


def new_run_ids() -> tuple[str, str]:
    """Return a fresh (workflow_id, run_id) pair, mirroring the handler's own scheme."""
    turn = uuid.uuid4().hex
    return f"thread-{turn[:8]}", f"thread-run-{turn[8:16]}"


def base_event_kwargs(*, workflow_id: str, run_id: str) -> dict[str, Any]:
    """Shared required fields for constructing a `LangChainGovernanceEvent` in fixtures."""
    from openbox_langgraph.types import rfc3339_now

    return {
        "source": "workflow-telemetry",
        "workflow_id": workflow_id,
        "run_id": run_id,
        "workflow_type": "GoldenBaselineAgent",
        "task_queue": "langgraph",
        "timestamp": rfc3339_now(),
    }


async def capture_single_event(
    event: LangChainGovernanceEvent, *, signed: bool = False
) -> RecordedRequest:
    """Send one event through evaluate_event via a recording mock transport.

    Returns the single captured request (raises if zero or multiple were sent —
    duplicate suppression could otherwise hide a wiring mistake in a fixture).
    """
    client, recorder = build_recording_client(signed=signed)
    try:
        await client.evaluate_event(event)
    finally:
        await client.close()
    if len(recorder.requests) != 1:
        msg = f"expected exactly 1 captured request, got {len(recorder.requests)}"
        raise AssertionError(msg)
    return recorder.requests[0]
