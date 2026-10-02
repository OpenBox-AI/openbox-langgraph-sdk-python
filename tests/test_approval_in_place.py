"""Real graph/callback regressions for repeated approval without graph replay."""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Annotated, TypedDict

import httpx
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from openbox_core.client import EvaluationClient
from openbox_core.contracts.results import ApprovalResult, EvaluationResult, Verdict

from openbox_langgraph.config import initialize
from openbox_langgraph.errors import ApprovalExpiredError, ApprovalRejectedError
from openbox_langgraph.langgraph_handler import OpenBoxLangGraphHandler


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@pytest.fixture
async def scenario(monkeypatch):
    handlers = []

    def build(sync_tool: bool):
        loop = asyncio.get_running_loop()
        effects = []
        starts = []
        polls = []
        decisions = {}
        waiting = [asyncio.Event(), asyncio.Event()]

        def body(index: int) -> str:
            effects.append(f"body-{index}")
            return str(index)

        async def abody(index: int) -> str:
            return body(index)

        protected_tool = tool("protected_tool", description="Record a side effect")(
            body if sync_tool else abody
        )

        async def agent(state):
            completed = sum(msg.type == "tool" for msg in state["messages"])
            effects.append(f"plan-{completed + 1}")
            if completed == 2:
                return {"messages": [AIMessage(content="done")]}
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "protected_tool",
                                "args": {"index": completed + 1},
                                "id": f"call-{completed + 1}",
                            }
                        ],
                    )
                ]
            }

        graph = StateGraph(State)
        graph.add_node("agent", agent)
        graph.add_node("tools", ToolNode([protected_tool], handle_tool_errors=False))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent", lambda s: "tools" if s["messages"][-1].tool_calls else END
        )
        graph.add_edge("tools", "agent")

        def evaluate(client, payload):
            if (
                payload.get("event_type") == "ActivityStarted"
                and payload.get("activity_type") == "protected_tool"
            ):
                starts.append((payload["workflow_id"], payload["run_id"], payload["activity_id"]))
                return EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL)
            return EvaluationResult(verdict=Verdict.ALLOW)

        async def aevaluate(client, payload):
            return evaluate(client, payload)

        def poll(client, workflow_id, run_id, activity_id):
            key = (workflow_id, run_id, activity_id)
            polls.append(key)
            index = starts.index(key)
            loop.call_soon_threadsafe(waiting[index].set)
            return decisions.get(index, ApprovalResult())

        async def apoll(client, workflow_id, run_id, activity_id):
            return poll(client, workflow_id, run_id, activity_id)

        monkeypatch.setattr(EvaluationClient, "evaluate", evaluate)
        monkeypatch.setattr(EvaluationClient, "aevaluate", aevaluate)
        monkeypatch.setattr(EvaluationClient, "poll_approval", poll)
        monkeypatch.setattr(EvaluationClient, "apoll_approval", apoll)

        def unexpected_network(*args, **kwargs):
            raise AssertionError("Unexpected network request")

        monkeypatch.setattr(httpx.Client, "send", unexpected_network)
        monkeypatch.setattr(httpx.AsyncClient, "send", unexpected_network)
        initialize("https://core.example.test", "obx_test_approval", validate=False)
        handler = OpenBoxLangGraphHandler(graph=graph.compile())
        handler._config.hitl.poll_interval_ms = 5
        handlers.append(handler)
        return handler, effects, starts, polls, decisions, waiting

    yield build
    for handler in handlers:
        await handler._core_runtime.aclose()


async def run(handler, entrypoint):
    inputs = {"messages": [HumanMessage(content="do both actions")]}
    config = {"configurable": {"thread_id": "approval-in-place"}}
    if entrypoint == "ainvoke":
        return await handler.ainvoke(inputs, config=config)
    async for _ in getattr(handler, entrypoint)(inputs, config=config):
        pass


@pytest.mark.parametrize("sync_tool", [False, True])
@pytest.mark.parametrize("entrypoint", ["ainvoke", "astream_governed", "astream_events"])
async def test_two_approvals_resume_each_activity_once(scenario, sync_tool, entrypoint):
    handler, effects, starts, polls, decisions, waiting = scenario(sync_tool)
    task = asyncio.create_task(run(handler, entrypoint))
    try:
        await asyncio.wait_for(waiting[0].wait(), 2)
        await asyncio.sleep(0.025)
        assert effects == ["plan-1"]
        assert not task.done()
        decisions[0] = ApprovalResult(verdict=Verdict.ALLOW)

        await asyncio.wait_for(waiting[1].wait(), 2)
        await asyncio.sleep(0.025)
        assert effects == ["plan-1", "body-1", "plan-2"]
        assert not task.done()
        decisions[1] = ApprovalResult(verdict=Verdict.ALLOW)
        await asyncio.wait_for(task, 2)

        assert effects == ["plan-1", "body-1", "plan-2", "body-2", "plan-3"]
        assert len(starts) == 2
        assert len(set(starts)) == 2
        assert set(polls) == set(starts)
        assert handler._activity_approvals._turns == {}
        assert Counter(effects)["plan-1"] == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("sync_tool", [False, True])
@pytest.mark.parametrize(
    "decision,error",
    [
        (ApprovalResult(verdict=Verdict.BLOCK, reason="rejected"), ApprovalRejectedError),
        (ApprovalResult(expired=True), ApprovalExpiredError),
    ],
)
async def test_rejected_or_expired_approval_never_runs_body(scenario, sync_tool, decision, error):
    handler, effects, _, _, decisions, waiting = scenario(sync_tool)
    task = asyncio.create_task(run(handler, "ainvoke"))
    try:
        await asyncio.wait_for(waiting[0].wait(), 2)
        decisions[0] = decision
        with pytest.raises(error):
            await asyncio.wait_for(task, 2)
        assert effects == ["plan-1"]
        assert handler._activity_approvals._turns == {}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("sync_tool", [False, True])
async def test_cancellation_prevents_late_approval_running_body(scenario, sync_tool):
    handler, effects, _, _, decisions, waiting = scenario(sync_tool)
    task = asyncio.create_task(run(handler, "ainvoke"))
    await asyncio.wait_for(waiting[0].wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    decisions[0] = ApprovalResult(verdict=Verdict.ALLOW)
    await asyncio.sleep(0.05)
    assert effects == ["plan-1"]
    assert handler._activity_approvals._turns == {}
