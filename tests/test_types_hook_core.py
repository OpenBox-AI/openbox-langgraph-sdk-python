"""Core type parsing and hook-governance payload tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from openbox_langgraph import hook_governance
from openbox_langgraph.errors import GovernanceBlockedError
from openbox_langgraph.types import (
    GovernanceVerdictResponse,
    LangChainGovernanceEvent,
    LangGraphStreamEvent,
    Verdict,
    highest_priority_verdict,
    lang_graph_event_to_server_type,
    parse_approval_response,
    parse_governance_response,
    rfc3339_now,
    safe_serialize,
    to_server_event_type,
    verdict_from_string,
    verdict_priority,
    verdict_requires_approval,
    verdict_should_stop,
)


class _Context:
    span_id = 0xABCD
    trace_id = 0x1234


class _Parent:
    span_id = 0x9999


class _Span:
    parent = _Parent()

    def get_span_context(self) -> _Context:
        return _Context()


class _SpanProcessor:
    def __init__(self) -> None:
        self.aborts: list[tuple[str, str, str]] = []
        self.halts: list[tuple[str, str, str]] = []

    def get_activity_context_by_trace(self, _trace_id: int) -> dict[str, Any]:
        return {
            "workflow_id": "workflow-1",
            "run_id": "run-1",
            "activity_id": "activity-1",
            "activity_type": "tool",
            "workflow_type": "AgentWorkflow",
            "task_queue": "langgraph",
            "source": "workflow-telemetry",
            "non_serializable": object(),
        }

    def get_activity_abort(self, workflow_id: str, activity_id: str) -> str | None:
        for wf, activity, reason in self.aborts:
            if wf == workflow_id and activity == activity_id:
                return reason
        return None

    def set_activity_abort(self, workflow_id: str, activity_id: str, reason: str) -> None:
        self.aborts.append((workflow_id, activity_id, reason))

    def set_halt_requested(self, workflow_id: str, activity_id: str, reason: str) -> None:
        self.halts.append((workflow_id, activity_id, reason))


def test_verdict_helpers_and_event_type_mapping() -> None:
    assert verdict_from_string(None) == Verdict.ALLOW
    assert verdict_from_string("continue") == Verdict.ALLOW
    assert verdict_from_string("stop") == Verdict.HALT
    assert verdict_from_string("request-approval") == Verdict.REQUIRE_APPROVAL
    assert verdict_priority(Verdict.BLOCK) == 3
    assert highest_priority_verdict([Verdict.ALLOW, Verdict.HALT, Verdict.BLOCK]) == Verdict.HALT
    assert verdict_should_stop(Verdict.BLOCK)
    assert verdict_requires_approval(Verdict.REQUIRE_APPROVAL)
    assert lang_graph_event_to_server_type("on_tool_start") == "ActivityStarted"
    assert lang_graph_event_to_server_type("unknown") is None
    assert to_server_event_type("AgentFinish") == "ActivityCompleted"
    assert to_server_event_type("unknown") == "ActivityCompleted"


def test_stream_event_and_governance_event_serialization() -> None:
    stream_event = LangGraphStreamEvent.from_dict(
        {
            "event": "on_tool_start",
            "name": "search",
            "run_id": "run-1",
            "metadata": None,
            "data": None,
            "tags": None,
            "parent_ids": None,
        }
    )
    governance_event = LangChainGovernanceEvent(
        source="workflow-telemetry",
        event_type="ToolStarted",
        workflow_id="workflow-1",
        run_id="run-1",
        workflow_type="AgentWorkflow",
        task_queue="langgraph",
        timestamp="2026-01-01T00:00:00.000Z",
        activity_id="activity-1",
    )

    assert stream_event.metadata == {}
    assert stream_event.data == {}
    assert stream_event.tags == []
    assert governance_event.to_dict()["activity_id"] == "activity-1"
    assert "activity_type" not in governance_event.to_dict()


def test_response_parsers_and_safe_serialize() -> None:
    response = parse_governance_response(
        {
            "action": "request_approval",
            "reason": "needs review",
            "risk_score": 0.7,
            "guardrails_result": {
                "input_type": "activity_output",
                "redacted_input": {"text": "***"},
                "validation_passed": False,
                "reasons": [{"type": "pii", "field": "text", "reason": "pii"}],
            },
        }
    )
    approval = parse_approval_response({"action": "stop", "expired": True})

    assert isinstance(response, GovernanceVerdictResponse)
    assert response.verdict == Verdict.REQUIRE_APPROVAL
    assert response.action == "require-approval"
    assert response.guardrails_result is not None
    assert response.guardrails_result.reasons[0].reason == "pii"
    assert approval.verdict == Verdict.HALT
    assert approval.expired
    assert rfc3339_now().endswith("Z")
    assert safe_serialize({"items": (1, object())})["items"][0] == 1


def test_hook_governance_configuration_and_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _SpanProcessor()
    hook_governance.configure(
        "https://core.openbox.test/",
        "obx_test_key",
        processor,  # type: ignore[arg-type]
        api_timeout=5,
        on_api_error=hook_governance.FAIL_CLOSED,
    )

    payload = hook_governance._build_payload(
        _Span(),
        {"hook_type": "http_request", "stage": "started"},
    )
    span_id, trace_id, parent_span_id = hook_governance.extract_span_context(_Span())

    assert hook_governance.is_configured()
    assert hook_governance.get_span_processor() is processor
    assert payload is not None
    assert payload["workflow_id"] == "workflow-1"
    assert payload["hook_trigger"]
    assert payload["spans"][0]["activity_id"] == "activity-1"
    assert isinstance(payload["non_serializable"], str)
    assert span_id == "000000000000abcd"
    assert trace_id == "00000000000000000000000000001234"
    assert parent_span_id == "0000000000009999"

    monkeypatch.setattr(hook_governance, "_span_processor", None)
    assert hook_governance._build_payload(_Span(), {}) is None


def test_hook_governance_handles_verdicts_and_fail_closed() -> None:
    processor = _SpanProcessor()
    hook_governance.configure(
        "https://core.openbox.test",
        "obx_test_key",
        processor,  # type: ignore[arg-type]
        on_api_error=hook_governance.FAIL_CLOSED,
    )

    with pytest.raises(GovernanceBlockedError):
        hook_governance._handle_verdict(
            {"verdict": "halt", "reason": "stop now"},
            "https://api.example.test",
            _Span(),
        )

    assert processor.aborts == [("workflow-1", "activity-1", "stop now")]
    assert processor.halts == [("workflow-1", "activity-1", "stop now")]

    response = MagicMock(status_code=503)
    with pytest.raises(GovernanceBlockedError):
        hook_governance._send_and_handle(response, "https://api.example.test")


def test_hook_governance_evaluate_sync_short_circuits_aborted_activity() -> None:
    processor = _SpanProcessor()
    processor.set_activity_abort("workflow-1", "activity-1", "approval required")
    hook_governance.configure(
        "https://core.openbox.test",
        "obx_test_key",
        processor,  # type: ignore[arg-type]
    )

    with pytest.raises(GovernanceBlockedError):
        hook_governance.evaluate_sync(_Span(), "https://api.example.test", {})
