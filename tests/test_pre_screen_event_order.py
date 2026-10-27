"""WorkflowStarted must reach Core before any other event of the run: Core
creates the session from it and holds SignalReceived / ActivityStarted with
SessionNotReady until it exists. Sent one at a time, a signal sent first
would block WorkflowStarted for good."""

from __future__ import annotations

from types import SimpleNamespace

from openbox_langgraph.langgraph_handler import OpenBoxLangGraphHandler


class _RecordingClient:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def evaluate_event(self, event):
        self.sent.append(event.event_type)
        return None


async def test_workflow_started_goes_before_the_user_prompt_signal():
    handler = object.__new__(OpenBoxLangGraphHandler)
    handler._config = SimpleNamespace(
        agent_name="support-agent", task_queue="langgraph", session_id=None,
        send_chain_start_event=True, send_llm_start_event=True, send_chain_end_event=True,
    )
    client = _RecordingClient()
    handler._client = client

    await handler._pre_screen_input(
        {"messages": [{"role": "user", "content": "check order A-1003"}]}, "wf-1", "run-1"
    )

    assert client.sent == ["WorkflowStarted", "SignalReceived", "LLMStarted"]
