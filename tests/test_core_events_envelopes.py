"""`core_events.to_envelope` wire-parity contract.

Pins the invariant the golden-parity gate (`test_lifecycle_golden_parity.py`)
depends on structurally: `to_envelope(event).to_payload_dict()` must equal
the legacy wire body (`event.to_dict()` + the inline `event_type`/
`task_queue`/`source` normalization `evaluate_event` applies) MINUS exactly
the three allowed compatibility deltas — never more, never less.
"""

from __future__ import annotations

from typing import Any

import pytest
from openbox_core.contracts.events import EventType

from openbox_langgraph.core_events import to_envelope
from openbox_langgraph.types import LangChainGovernanceEvent, to_server_event_type

_ALLOWED_DELTA_KEYS = frozenset({"hook_trigger", "spans", "span_count"})


def _legacy_wire_body(event: LangChainGovernanceEvent) -> dict[str, Any]:
    """Reproduce the exact pre-migration wire body `evaluate_event` sent.

    Mirrors the 3-line transform inlined in `GovernanceClient.evaluate_event`
    (and `graph_capture_harness._wire_body`) verbatim.
    """
    body = event.to_dict()
    body["event_type"] = to_server_event_type(event.event_type)
    body["task_queue"] = event.task_queue or "langgraph"
    body["source"] = "workflow-telemetry"
    return body


def _assert_only_allowed_delta(event: LangChainGovernanceEvent) -> dict[str, Any]:
    """Assert `to_envelope(event)`'s wire body differs from the legacy body
    by AT MOST the three allowed compatibility keys, with identical values on
    every shared key. Returns the new body for further per-event assertions.
    """
    legacy = _legacy_wire_body(event)
    new = to_envelope(event).to_payload_dict()

    diff_keys = set(legacy) ^ set(new)
    unexpected = diff_keys - _ALLOWED_DELTA_KEYS
    assert not unexpected, (
        f"unexpected wire delta beyond the 3 allowed compat keys: {unexpected} "
        f"(legacy={legacy}, new={new})"
    )
    for key in set(legacy) & set(new):
        assert legacy[key] == new[key], (
            f"value mismatch on {key!r}: {legacy[key]!r} != {new[key]!r}"
        )
    return new


def _base_kwargs(**overrides: Any) -> dict[str, Any]:
    base = {
        "source": "workflow-telemetry",
        "workflow_id": "wf-1",
        "run_id": "run-1",
        "workflow_type": "LangGraphRun",
        "task_queue": "langgraph",
        "timestamp": "2026-01-01T00:00:00.000Z",
    }
    base.update(overrides)
    return base


class TestLifecycleEventParity:
    """Every real-emittable lifecycle event type maps with only the 3 allowed deltas."""

    def test_workflow_started(self) -> None:
        event = LangChainGovernanceEvent(
            event_type="WorkflowStarted",
            activity_id="run-1-wf",
            activity_type="LangGraphRun",
            activity_input=[{"messages": [{"content": "hi"}]}],
            **_base_kwargs(),
        )
        new = _assert_only_allowed_delta(event)
        assert new["event_type"] == "WorkflowStarted"
        assert "hook_trigger" not in new
        assert "spans" not in new
        assert "span_count" not in new

    def test_signal_received(self) -> None:
        event = LangChainGovernanceEvent(
            event_type="SignalReceived",
            activity_id="run-1-sig",
            activity_type="user_prompt",
            signal_name="user_prompt",
            signal_args=["hi"],
            **_base_kwargs(),
        )
        new = _assert_only_allowed_delta(event)
        assert new["event_type"] == "SignalReceived"
        assert new["signal_name"] == "user_prompt"
        assert new["signal_args"] == ["hi"]

    def test_tool_started_with_subagent_and_tool_type(self) -> None:
        event = LangChainGovernanceEvent(
            event_type="ToolStarted",
            activity_id="run-abc",
            activity_type="echo_tool",
            activity_input=[
                {"text": "hi"},
                {"__openbox": {"subagent_name": "writer", "tool_type": "a2a"}},
            ],
            tool_name="echo_tool",
            tool_type="a2a",
            tool_input={"text": "hi"},
            subagent_name="writer",
            langgraph_node="tools",
            langgraph_step=2,
            **_base_kwargs(),
        )
        new = _assert_only_allowed_delta(event)
        assert new["event_type"] == "ActivityStarted"
        assert new["subagent_name"] == "writer"
        assert new["tool_type"] == "a2a"
        assert new["langgraph_step"] == 2

    def test_tool_completed(self) -> None:
        event = LangChainGovernanceEvent(
            event_type="ToolCompleted",
            activity_id="run-abc-c",
            activity_type="echo_tool",
            activity_output={"content": "echo: hi", "id": None, "artifact": None},
            tool_name="echo_tool",
            status="completed",
            duration_ms=1.5,
            **_base_kwargs(),
        )
        new = _assert_only_allowed_delta(event)
        assert new["event_type"] == "ActivityCompleted"
        # Nested None survives (matches golden id:null/artifact:null) even
        # though top-level None fields are never forwarded.
        assert new["activity_output"]["id"] is None
        assert new["activity_output"]["artifact"] is None

    def test_llm_started(self) -> None:
        event = LangChainGovernanceEvent(
            event_type="LLMStarted",
            activity_id="run-1-pre",
            activity_type="llm_call",
            activity_input=[{"prompt": "hi"}],
            prompt="hi",
            **_base_kwargs(),
        )
        new = _assert_only_allowed_delta(event)
        assert new["event_type"] == "ActivityStarted"

    def test_llm_completed_uses_activity_completed_factory_no_hook_no_spans(self) -> None:
        """LLMCompleted MUST produce a plain ActivityCompleted envelope — never
        a hook() envelope — so it can never fail the gate's
        ACTIVITY_COMPLETED_WITH_SPANS / HOOK_TRIGGER_FALSE strict checks."""
        event = LangChainGovernanceEvent(
            event_type="LLMCompleted",
            activity_id="run-1-pre-c",
            activity_type="llm_call",
            activity_output={"content": "hello"},
            status="completed",
            duration_ms=12.5,
            llm_model="gpt-4",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            has_tool_calls=False,
            completion="hello",
            langgraph_node="agent",
            langgraph_step=1,
            **_base_kwargs(),
        )
        envelope = to_envelope(event)
        assert envelope.event_type is EventType.ACTIVITY_COMPLETED
        assert envelope.hook_trigger is False
        assert envelope.spans == ()
        new = _assert_only_allowed_delta(event)
        assert new["llm_model"] == "gpt-4"
        assert new["input_tokens"] == 10

    def test_chain_completed_root_close(self) -> None:
        event = LangChainGovernanceEvent(
            event_type="ChainCompleted",
            activity_id="root-run-id",
            activity_type="LangGraph",
            workflow_output={"messages": []},
            activity_output={"result": "x"},
            status="completed",
            duration_ms=5.0,
            **_base_kwargs(),
        )
        new = _assert_only_allowed_delta(event)
        assert new["event_type"] == "WorkflowCompleted"
        assert new["workflow_output"] == {"messages": []}

    def test_workflow_completed_error_close_string_error(self) -> None:
        event = LangChainGovernanceEvent(
            event_type="WorkflowCompleted",
            activity_id="run-1-wf",
            activity_type="LangGraphRun",
            status="failed",
            error="boom",
            **_base_kwargs(),
        )
        new = _assert_only_allowed_delta(event)
        assert new["event_type"] == "WorkflowCompleted"
        assert new["status"] == "failed"
        assert new["error"] == "boom"

    def test_workflow_failed_dict_error(self) -> None:
        """WorkflowFailed has no real construction site in the handler today
        (verified — see tests/golden/layer1_handbuilt_pins.py) but the wire
        mapping must still be pinned for the hand-built-pin serialization."""
        event = LangChainGovernanceEvent(
            event_type="WorkflowFailed",
            activity_id="run-1-wf",
            activity_type="GoldenBaselineAgent",
            status="failed",
            error={"message": "unrecoverable workflow error"},
            **_base_kwargs(workflow_type="GoldenBaselineAgent"),
        )
        new = _assert_only_allowed_delta(event)
        assert new["event_type"] == "WorkflowFailed"
        assert new["error"] == {"message": "unrecoverable workflow error"}


class TestExtraPayloadNoneFiltering:
    """Top-level None fields never leak into `extra` as literal wire `null`s."""

    def test_unset_optional_fields_are_absent_not_null(self) -> None:
        event = LangChainGovernanceEvent(
            event_type="ToolStarted",
            activity_id="run-1",
            activity_type="my_tool",
            **_base_kwargs(),
        )
        body = to_envelope(event).to_payload_dict()
        # Every optional field left at its dataclass default (None) must be
        # absent from the wire body — never present as a literal null.
        for optional_field in ("tool_name", "tool_type", "subagent_name", "prompt"):
            assert optional_field not in body

    def test_nested_none_inside_populated_field_survives(self) -> None:
        """A populated dict field carrying its OWN nested None (e.g.
        activity_output={"id": None}) must reach the wire with that null —
        the top-level-only None filter must not recurse into it."""
        event = LangChainGovernanceEvent(
            event_type="ToolCompleted",
            activity_id="run-1-c",
            activity_type="my_tool",
            activity_output={"id": None, "content": "ok"},
            **_base_kwargs(),
        )
        body = to_envelope(event).to_payload_dict()
        assert body["activity_output"] == {"id": None, "content": "ok"}


class TestEnvelopeOwnedSlots:
    """activity_id/activity_type/timestamp ride the envelope's dedicated slots,
    never duplicated inside payload."""

    def test_activity_id_and_type_not_duplicated_in_payload(self) -> None:
        event = LangChainGovernanceEvent(
            event_type="ToolStarted",
            activity_id="run-1",
            activity_type="my_tool",
            **_base_kwargs(),
        )
        envelope = to_envelope(event)
        assert envelope.activity_id == "run-1"
        assert envelope.activity_type == "my_tool"
        assert "activity_id" not in envelope.payload
        assert "activity_type" not in envelope.payload

    def test_timestamp_rides_dedicated_slot(self) -> None:
        event = LangChainGovernanceEvent(
            event_type="WorkflowStarted",
            activity_id="run-1-wf",
            activity_type="LangGraphRun",
            **_base_kwargs(timestamp="2026-06-15T12:00:00.000Z"),
        )
        envelope = to_envelope(event)
        assert envelope.timestamp == "2026-06-15T12:00:00.000Z"
        assert "timestamp" not in envelope.payload


def test_to_server_event_type_never_produces_an_unmapped_wire_label() -> None:
    """`to_envelope`'s defensive ValueError branch (unreachable today) relies
    on `to_server_event_type` always returning one of the 6 known wire
    labels — pin that contract directly: unknown input falls through to its
    own documented default ("ActivityCompleted"), never an arbitrary string.
    """
    assert to_server_event_type("TotallyUnknownLabel") == "ActivityCompleted"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
