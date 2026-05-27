"""Core verdict, span processor, and HITL tests."""

from __future__ import annotations

from typing import Any

import pytest

from openbox_langgraph.errors import (
    ApprovalExpiredError,
    ApprovalRejectedError,
    GovernanceBlockedError,
    GovernanceHaltError,
    GuardrailsValidationError,
)
from openbox_langgraph.hitl import HITLPollParams, poll_until_decision
from openbox_langgraph.span_processor import WorkflowSpanProcessor
from openbox_langgraph.types import (
    ApprovalResponse,
    GovernanceVerdictResponse,
    GuardrailsReason,
    GuardrailsResult,
    HITLConfig,
    Verdict,
    WorkflowSpanBuffer,
)
from openbox_langgraph.verdict_handler import (
    VerdictEnforcementResult,
    enforce_verdict,
    is_hitl_applicable,
    lang_graph_event_to_context,
)


class _FallbackProcessor:
    def __init__(self) -> None:
        self.ended: list[Any] = []
        self.shutdown_called = False
        self.flush_timeout: int | None = None

    def on_end(self, span: Any) -> None:
        self.ended.append(span)

    def shutdown(self) -> None:
        self.shutdown_called = True

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        self.flush_timeout = timeout_millis
        return True


class _Span:
    def __init__(self, trace_id: int = 123, url: str | None = None) -> None:
        self.attributes = {"http.url": url} if url else {}
        self.context = type("Context", (), {"trace_id": trace_id})()

    def get_span_context(self) -> Any:
        return self.context


class _ApprovalClient:
    def __init__(self, responses: list[ApprovalResponse | None]) -> None:
        self.responses = responses
        self.calls = 0

    async def poll_approval(self, _params: Any) -> ApprovalResponse | None:
        self.calls += 1
        return self.responses.pop(0)


def test_lang_graph_event_to_context_and_hitl_applicability() -> None:
    assert lang_graph_event_to_context("on_chain_start", is_root=True) == "graph_root_start"
    assert lang_graph_event_to_context("on_chain_start") == "graph_node_start"
    assert lang_graph_event_to_context("on_tool_end") == "tool_end"
    assert lang_graph_event_to_context("unknown") == "other"
    assert is_hitl_applicable("tool_start")
    assert not is_hitl_applicable("chain_end")


def test_enforce_verdict_observation_context_ignores_block() -> None:
    result = enforce_verdict(
        GovernanceVerdictResponse(verdict=Verdict.BLOCK, reason="blocked"),
        "chain_end",
    )

    assert isinstance(result, VerdictEnforcementResult)
    assert not result.blocked


def test_enforce_verdict_raises_halt_and_block() -> None:
    with pytest.raises(GovernanceHaltError):
        enforce_verdict(GovernanceVerdictResponse(verdict=Verdict.HALT), "tool_start")

    with pytest.raises(GovernanceBlockedError):
        enforce_verdict(GovernanceVerdictResponse(verdict=Verdict.BLOCK), "tool_start")


def test_enforce_verdict_guardrail_reason_cleaning() -> None:
    response = GovernanceVerdictResponse(
        verdict=Verdict.ALLOW,
        guardrails_result=GuardrailsResult(
            input_type="activity_input",
            redacted_input={},
            validation_passed=False,
            reasons=[
                GuardrailsReason(
                    type="pii",
                    field="prompt",
                    reason="Sensitive text\n\nThought: internal scratchpad",
                )
            ],
        ),
    )

    with pytest.raises(GuardrailsValidationError) as exc_info:
        enforce_verdict(response, "tool_start")

    assert exc_info.value.reasons == ["Sensitive text"]


def test_enforce_verdict_require_approval_and_constrain_warning() -> None:
    approval = enforce_verdict(
        GovernanceVerdictResponse(verdict=Verdict.REQUIRE_APPROVAL),
        "tool_start",
    )

    assert approval.requires_hitl

    with pytest.raises(GovernanceBlockedError):
        enforce_verdict(
            GovernanceVerdictResponse(verdict=Verdict.REQUIRE_APPROVAL),
            "chain_start",
        )

    with pytest.warns(UserWarning):
        result = enforce_verdict(
            GovernanceVerdictResponse(verdict=Verdict.CONSTRAIN, reason="slow down"),
            "tool_start",
        )
    assert not result.requires_hitl


def test_span_processor_registers_and_cleans_workflow_state() -> None:
    processor = WorkflowSpanProcessor()
    buffer = WorkflowSpanBuffer(workflow_id="workflow-1", run_id="run-1")

    processor.register_workflow("workflow-1", buffer)
    processor.register_trace(123, "workflow-1", "activity-1")
    processor.set_verdict("workflow-1", Verdict.BLOCK, "blocked", run_id="run-1")
    processor.set_activity_context("workflow-1", "activity-1", {"activity_id": "activity-1"})
    processor.set_activity_abort("workflow-1", "activity-1", "approval required")
    processor.set_halt_requested("workflow-1", "activity-1", "halted")

    assert processor.get_buffer("workflow-1") is buffer
    assert processor.get_verdict("workflow-1") == {
        "verdict": Verdict.BLOCK,
        "reason": "blocked",
        "run_id": "run-1",
    }
    assert processor.get_activity_context_by_trace(123) == {"activity_id": "activity-1"}
    assert processor.get_activity_abort("workflow-1", "activity-1") == "approval required"
    assert processor.get_halt_requested("workflow-1", "activity-1") == "halted"

    processor.unregister_workflow("workflow-1")

    assert processor.get_buffer("workflow-1") is None
    assert processor.get_verdict("workflow-1") is None
    assert processor.get_activity_abort("workflow-1", "activity-1") is None
    assert processor.get_halt_requested("workflow-1", "activity-1") is None


def test_span_processor_context_fallbacks_and_forwarding() -> None:
    fallback = _FallbackProcessor()
    processor = WorkflowSpanProcessor(
        fallback_processor=fallback,
        ignored_url_prefixes=["https://core.openbox.test"],
    )

    processor.set_activity_context("workflow-1", "activity-1", {"activity_id": "activity-1"})
    assert processor.get_activity_context_by_trace(999) == {"activity_id": "activity-1"}

    processor.set_activity_context("workflow-1", "activity-2", {"activity_id": "activity-2"})
    assert processor.get_activity_context_by_trace(999) is None

    processor.set_sync_mode(True)
    assert processor.get_activity_context_by_trace(999) == {"activity_id": "activity-2"}

    ignored_span = _Span(url="https://core.openbox.test/api")
    normal_span = _Span(url="https://api.example.test")
    processor.on_end(ignored_span)
    processor.on_end(normal_span)
    processor.shutdown()

    assert fallback.ended == [ignored_span, normal_span]
    assert processor.force_flush(1234)
    assert fallback.flush_timeout == 1234
    assert fallback.shutdown_called


async def test_poll_until_decision_allows_after_transient_missing_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("openbox_langgraph.hitl.asyncio.sleep", _sleep)
    client = _ApprovalClient([None, ApprovalResponse(verdict=Verdict.ALLOW)])

    await poll_until_decision(
        client,  # type: ignore[arg-type]
        HITLPollParams("workflow-1", "run-1", "activity-1", "tool"),
        HITLConfig(poll_interval_ms=1),
    )

    assert client.calls == 2


async def test_poll_until_decision_raises_for_expired_and_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("openbox_langgraph.hitl.asyncio.sleep", _sleep)

    with pytest.raises(ApprovalExpiredError):
        await poll_until_decision(
            _ApprovalClient([ApprovalResponse(verdict=Verdict.REQUIRE_APPROVAL, expired=True)]),
            HITLPollParams("workflow-1", "run-1", "activity-1", "tool"),
            HITLConfig(poll_interval_ms=1),
        )

    with pytest.raises(ApprovalRejectedError):
        await poll_until_decision(
            _ApprovalClient([ApprovalResponse(verdict=Verdict.HALT, reason="no")]),
            HITLPollParams("workflow-1", "run-1", "activity-1", "tool"),
            HITLConfig(poll_interval_ms=1),
        )
