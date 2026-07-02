"""Core type parsing / verdict / serialization tests.

Hook-governance payload tests were removed with the legacy in-repo hook
modules — hook payload shape + evaluation is now owned entirely by the base
``openbox_core`` instrumentation.
"""

from __future__ import annotations

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
