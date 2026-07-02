"""Baseline oracle for wire-payload parity: asserts the golden fixtures under
tests/golden/ exist, are non-empty, and pin the invariants a future refactor
must preserve (signed/unsigned runs carry the expected identity header sets,
real-emitted bodies have the shape the real handler actually constructs).

Almost all Layer 1 wire-body fixtures were captured from a REAL
`OpenBoxLangGraphHandler.ainvoke()` run (fake chat model, injected recording
client — see tests/golden/real_emitted_event_fixtures.py), so this file pins
the handler's own event construction, not just the client's serialization of
a hand-authored stand-in. The two `.handbuilt_serialization_pin` fixtures are
the verified exceptions (see tests/golden/layer1_handbuilt_pins.py for why).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openbox_langgraph.identity import (
    OPENBOX_AGENT_DID_HEADER,
    OPENBOX_AGENT_NONCE_HEADER,
    OPENBOX_AGENT_SIGNATURE_HEADER,
    OPENBOX_AGENT_TIMESTAMP_HEADER,
    OPENBOX_BODY_SHA256_HEADER,
)

GOLDEN_DIR = Path(__file__).parent / "golden"

_ALL_FIVE_IDENTITY_HEADERS = {
    OPENBOX_AGENT_DID_HEADER.lower(),
    OPENBOX_AGENT_TIMESTAMP_HEADER.lower(),
    OPENBOX_AGENT_NONCE_HEADER.lower(),
    OPENBOX_BODY_SHA256_HEADER.lower(),
    OPENBOX_AGENT_SIGNATURE_HEADER.lower(),
}

# Layer 1 wire-body events — each has a `.raw.json` + `.normalized.json` pair.
# Real-emitted (captured from an actual handler.ainvoke() run — see
# real_emitted_event_fixtures.py) unless noted otherwise.
_WIRE_BODY_EVENTS = [
    "signal_received",
    "workflow_started",
    "llm_started",
    "llm_completed",
    "tool_started",
    "tool_completed",
    "subagent_tool_started",
    "subagent_tool_completed",
    "chain_completed",
    "workflow_completed_error_close",
    # Hand-built pins: no real construction site exists for these (verified —
    # see layer1_handbuilt_pins.py docstring for why each is infeasible).
    "chain_started.handbuilt_serialization_pin",
    "workflow_failed.handbuilt_serialization_pin",
]

# Layer 3 full-graph ordering captures — single JSON list files.
_ORDERING_FIXTURES = [
    "ordering_single_llm_turn",
    "ordering_two_llm_calls_turn",
    "ordering_tool_call_turn",
]


def _load(name: str) -> Any:
    """Load and parse a golden fixture file.

    Returns `Any` because fixtures are heterogeneous: Layer 1/2 fixtures
    parse to a dict, Layer 3 ordering fixtures parse to a list.
    """
    path = GOLDEN_DIR / name
    if not path.exists():
        pytest.fail(f"golden fixture missing: {path}")
    content = path.read_text()
    if not content.strip():
        pytest.fail(f"golden fixture is empty: {path}")
    parsed: Any = json.loads(content)
    return parsed


@pytest.mark.parametrize("event_name", _WIRE_BODY_EVENTS)
def test_wire_body_fixture_pair_exists_and_nonempty(event_name: str) -> None:
    """Every Layer 1 event has both a raw and normalized fixture, non-empty."""
    raw = _load(f"{event_name}.raw.json")
    normalized = _load(f"{event_name}.normalized.json")
    assert raw, f"{event_name}.raw.json parsed to an empty value"
    assert normalized, f"{event_name}.normalized.json parsed to an empty value"


@pytest.mark.parametrize("fixture_name", _ORDERING_FIXTURES)
def test_ordering_fixture_exists_and_nonempty(fixture_name: str) -> None:
    """Every Layer 3 ordering capture is a non-empty list of captured events."""
    ordered = _load(f"{fixture_name}.json")
    assert isinstance(ordered, list)
    assert ordered, f"{fixture_name}.json captured zero events"
    for entry in ordered:
        assert "event_type" in entry
        assert "activity_id" in entry


def test_llm_completed_is_plain_activity_completed_with_no_spans() -> None:
    """LLMCompleted maps to a plain ActivityCompleted — no hook_trigger span data."""
    raw = _load("llm_completed.raw.json")
    assert raw["event_type"] == "ActivityCompleted"
    assert raw["status"] == "completed"
    assert "spans" not in raw, "LLMCompleted must not carry a spans field"


@pytest.mark.parametrize(
    ("fixture_name", "expected_keys"),
    [
        ("signal_received", {"event_type", "workflow_id", "run_id", "activity_id", "signal_name"}),
        (
            "workflow_started",
            {"event_type", "workflow_id", "run_id", "activity_id", "activity_input"},
        ),
        ("llm_started", {"event_type", "workflow_id", "run_id", "activity_id", "activity_input"}),
        ("llm_completed", {"event_type", "workflow_id", "run_id", "activity_id", "status"}),
        (
            "tool_started",
            {"event_type", "workflow_id", "run_id", "activity_id", "tool_name", "activity_input"},
        ),
        (
            "tool_completed",
            {"event_type", "workflow_id", "run_id", "activity_id", "tool_name", "status"},
        ),
        (
            "subagent_tool_started",
            {"event_type", "activity_id", "tool_name", "subagent_name", "activity_input"},
        ),
        (
            "subagent_tool_completed",
            {"event_type", "activity_id", "tool_name", "subagent_name", "status"},
        ),
        ("chain_completed", {"event_type", "workflow_id", "run_id", "activity_id", "status"}),
        (
            "workflow_completed_error_close",
            {"event_type", "workflow_id", "run_id", "activity_id", "status", "error"},
        ),
    ],
)
def test_real_emitted_body_has_expected_structural_keys(
    fixture_name: str, expected_keys: set[str]
) -> None:
    """Every real-emitted wire body has the fields the handler is expected to set.

    This does NOT assert exact values (those are volatile/representative) —
    it asserts the STRUCTURE a future refactor must preserve for each event
    type the real handler produces today.
    """
    raw = _load(f"{fixture_name}.raw.json")
    missing = expected_keys - raw.keys()
    assert not missing, f"{fixture_name}.raw.json is missing expected keys: {missing}"


def test_subagent_bodies_carry_subagent_name_and_a2a_tool_type() -> None:
    """Subagent-labelled Tool events set subagent_name + tool_type='a2a' on the wire."""
    started = _load("subagent_tool_started.raw.json")
    completed = _load("subagent_tool_completed.raw.json")
    assert started["subagent_name"] == "writer"
    assert started["tool_type"] == "a2a"
    assert completed["subagent_name"] == "writer"


def test_error_close_workflow_completed_has_failed_status() -> None:
    """The error-close path sends a literal WorkflowCompleted with status='failed'."""
    raw = _load("workflow_completed_error_close.raw.json")
    assert raw["event_type"] == "WorkflowCompleted"
    assert raw["status"] == "failed"
    assert raw.get("error"), "error-close body must carry a non-empty error field"


def test_handbuilt_pin_fixtures_are_labelled_and_distinct_from_real_emission() -> None:
    """The two verified-infeasible event types are pinned under the pin-labelled filename.

    chain_started's SDK-internal label is "ChainStarted", but to_server_event_type
    maps it to the wire label "WorkflowStarted" — this fixture pins the wire
    body, so it asserts the wire-mapped value (matching layer1_handbuilt_pins.py).
    """
    chain_started = _load("chain_started.handbuilt_serialization_pin.raw.json")
    workflow_failed = _load("workflow_failed.handbuilt_serialization_pin.raw.json")
    assert chain_started["event_type"] == "WorkflowStarted"
    assert chain_started["activity_type"] == "should_continue"
    assert workflow_failed["event_type"] == "WorkflowFailed"


def test_signed_run_has_all_five_identity_headers() -> None:
    """A GovernanceClient configured with agent_did/agent_private_key signs every request."""
    signed = _load("headers_signed.raw.json")
    present = {h.lower() for h in signed["present_headers"]}
    missing = _ALL_FIVE_IDENTITY_HEADERS - present
    assert not missing, f"signed run is missing identity headers: {missing}"


def test_unsigned_run_has_no_identity_headers() -> None:
    """A GovernanceClient with no agent identity never sends AIP signing headers."""
    unsigned = _load("headers_unsigned.raw.json")
    present = {h.lower() for h in unsigned["present_headers"]}
    leaked = _ALL_FIVE_IDENTITY_HEADERS & present
    assert not leaked, f"unsigned run leaked identity headers: {leaked}"


def test_two_llm_calls_in_one_turn_produce_independent_activity_rows() -> None:
    """Guards the skip-based LLMStarted de-dup: no keyed store collapses two LLM calls."""
    ordered = _load("ordering_two_llm_calls_turn.json")
    llm_completed_ids = [e["activity_id"] for e in ordered if e["event_type"] == "LLMCompleted"]
    assert len(llm_completed_ids) == 2, (
        "expected two independent LLMCompleted rows for a two-LLM-call turn, "
        f"got {llm_completed_ids}"
    )
    assert len(set(llm_completed_ids)) == 2, "the two LLMCompleted activity_ids must differ"
