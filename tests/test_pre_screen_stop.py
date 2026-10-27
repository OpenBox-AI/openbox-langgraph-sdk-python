"""A pre-screen stop reaches the caller as the verdict Core gave, never masked
or swallowed by the best-effort closing event that follows it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openbox_langgraph.errors import GovernanceHaltError, OpenBoxNetworkError
from openbox_langgraph.langgraph_handler import OpenBoxLangGraphHandler
from openbox_langgraph.types import GovernanceVerdictResponse


class _HaltingClient:
    """HALTs the pre-screen; rejects the closing event as Core does once the
    halt has already closed the session (HTTP 400 under fail-closed)."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def evaluate_event(self, event):
        self.sent.append(event.event_type)
        if event.event_type == "LLMStarted":
            return GovernanceVerdictResponse.from_dict(
                {"verdict": "halt", "reason": "banned words"}
            )
        if event.event_type == "WorkflowCompleted":
            raise OpenBoxNetworkError("Governance API error: HTTP 400")
        return None


def _handler(*, send_chain_end_event: bool) -> tuple[OpenBoxLangGraphHandler, _HaltingClient]:
    handler = object.__new__(OpenBoxLangGraphHandler)
    handler._config = SimpleNamespace(
        agent_name="support-agent", task_queue="langgraph", session_id=None,
        send_chain_start_event=True, send_llm_start_event=True,
        send_chain_end_event=send_chain_end_event,
    )
    client = _HaltingClient()
    handler._client = client
    return handler, client


PROMPT = {"messages": [{"role": "user", "content": "refund me or I file a chargeback"}]}


async def test_halt_survives_rejected_closing_event():
    handler, client = _handler(send_chain_end_event=True)

    with pytest.raises(GovernanceHaltError, match="banned words"):
        await handler._pre_screen_input(PROMPT, "wf-1", "run-1")

    assert client.sent[-1] == "WorkflowCompleted"


async def test_halt_raised_when_closing_event_disabled():
    handler, client = _handler(send_chain_end_event=False)

    with pytest.raises(GovernanceHaltError, match="banned words"):
        await handler._pre_screen_input(PROMPT, "wf-1", "run-1")

    assert "WorkflowCompleted" not in client.sent
