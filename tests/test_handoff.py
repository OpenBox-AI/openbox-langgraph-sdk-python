"""Task-only handoffs between two governed agents (PROD-1109).

The parent sends only the task. The child records the handoff with Core,
runs on the task alone in the parent's scope, and asks Core whether its
result may return before it leaves. Any refusal reaches the parent as the
uniform refusal, and the parent keeps working.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.errors import GovernanceBlockedError
from openbox_langgraph.handoff import (
    HANDOFF_KEY,
    HandoffEnvelope,
    HandoffResult,
    handoff_scope,
    handoff_tool,
)
from openbox_langgraph.langgraph_handler import (
    OpenBoxLangGraphHandler,
    OpenBoxLangGraphHandlerOptions,
)
from openbox_langgraph.retriever_labels import UNIFORM_REFUSAL

exposure = pytest.importorskip(
    "openbox_core.exposure", reason="openbox-sdk-python without exposure fields"
)

PARENT = {"from_agent_did": "did:openbox:parent", "from_agent_id": "agent-parent"}


# ── Envelope ─────────────────────────────────────────────────────


def test_envelope_round_trip_and_validation() -> None:
    env = HandoffEnvelope("summarise", "mas-1", "did:p", "agent-p", "matter:0417")
    assert HandoffEnvelope.from_dict(env.to_dict()) == env
    with pytest.raises(ValueError, match="not an OpenBox handoff"):
        HandoffEnvelope.from_dict({"task": "x"})
    with pytest.raises(ValueError, match="from_agent_did"):
        partial = {"multi_agent_session_id": "m", "from_agent_id": "a"}
        HandoffEnvelope.from_dict({"task": "x", HANDOFF_KEY: partial})
    assert handoff_scope(None, "mas-1") == "mas:mas-1"
    assert handoff_scope("matter:1", "mas-1") == "matter:1"
    assert HandoffResult.from_dict({"refused": True, "reason": "anything"}).refused
    assert HandoffResult.from_dict("garbage").refused
    assert HandoffResult(refused=True).to_dict() == {"refused": True, "reason": UNIFORM_REFUSAL}


# ── Parent: the tool sends the task and nothing else ─────────────


def _tool(send: Callable[[dict[str, Any]], Any]) -> Any:
    return handoff_tool(
        name="ask_research",
        description="Hand a research task to the research agent.",
        target_agent_id="agent-child",
        send=send,
        multi_agent_session_id="mas-1",
        **PARENT,
    )


def test_the_tool_takes_only_a_task() -> None:
    assert set(_tool(lambda env: {}).args) == {"task"}


async def test_the_parent_sends_the_task_and_its_scope_only() -> None:
    sent: list[dict[str, Any]] = []

    async def send(env: dict[str, Any]) -> dict[str, Any]:
        sent.append(env)
        return {"result": "done"}

    with exposure.exposure_scope("matter:0417"):
        out = await _tool(send).ainvoke({"task": "find the 2024 pricing memo"})
    assert out == "done"
    assert sent == [
        {
            "task": "find the 2024 pricing memo",
            HANDOFF_KEY: {
                "multi_agent_session_id": "mas-1",
                "scope": "matter:0417",
                **PARENT,
            },
        }
    ]


def test_a_refused_handoff_reads_as_any_block_and_sync_send_works() -> None:
    refused = _tool(lambda env: {"refused": True, "reason": "x"})
    assert refused.invoke({"task": "t"}) == UNIFORM_REFUSAL
    assert _tool(lambda env: {"result": "ok"}).invoke({"task": "t"}) == "ok"


def test_an_async_send_needs_ainvoke() -> None:
    async def send(env: dict[str, Any]) -> dict[str, Any]:
        return {"result": "ok"}

    with pytest.raises(TypeError, match="ainvoke"):
        _tool(send).invoke({"task": "t"})


# ── Child: record, run on the task alone, check the return ───────


def _core(sent: list[dict], block: Callable[[dict], bool] = lambda b: False) -> GovernanceClient:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        verdict = "block" if block(body) else "allow"
        return httpx.Response(200, json={"verdict": verdict, "reason": "Action blocked by policy"})

    client = GovernanceClient(api_url="https://core.test", api_key="obx_test_child")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client._sync_client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def _child(sent: list[dict], seen: list[list[Any]], **core: Any) -> OpenBoxLangGraphHandler:
    def answer(state: MessagesState) -> dict[str, Any]:
        seen.append(list(state["messages"]))
        return {"messages": [AIMessage(content=f"answer: {state['messages'][-1].content}")]}

    g = StateGraph(MessagesState)
    g.add_node("answer", answer)
    g.add_edge(START, "answer")
    g.add_edge("answer", END)
    return OpenBoxLangGraphHandler(
        g.compile(), OpenBoxLangGraphHandlerOptions(client=_core(sent, **core))
    )


def _envelope(scope: str | None = "matter:0417") -> dict[str, Any]:
    return HandoffEnvelope("find the memo", "mas-1", scope=scope, **PARENT).to_dict()


def _is(body: dict, kind: str) -> bool:
    if kind == "Handoff":
        return body.get("event_type") == "Handoff"
    return body.get("action_class") == kind


async def test_the_child_records_runs_on_the_task_alone_and_checks_the_return() -> None:
    sent: list[dict] = []
    seen: list[list[Any]] = []
    reply = await _child(sent, seen).areceive_handoff(_envelope())

    assert reply == {"result": "answer: find the memo"}
    # Recorded first, with who handed it over.
    assert _is(sent[0], "Handoff")
    assert sent[0]["from_agent_did"] == "did:openbox:parent"
    assert sent[0]["multi_agent_session_id"] == "mas-1"
    # The graph saw the task and nothing else.
    assert len(seen) == 1 and [m.content for m in seen[0]] == ["find the memo"]
    # The return is checked last, naming the parent's session.
    assert _is(sent[-1], "handoff_return")
    assert sent[-1]["participants"] == [
        {"kind": "agent_session", "id": "agent-parent/matter:0417"}
    ]
    # Every event of the child is in the parent's scope.
    assert {b.get("scope") for b in sent} == {"matter:0417"}
    assert "scope" not in exposure.current_exposure_fields()


async def test_a_refused_handoff_never_runs_the_graph() -> None:
    sent: list[dict] = []
    seen: list[list[Any]] = []
    child = _child(sent, seen, block=lambda b: _is(b, "Handoff"))
    reply = await child.areceive_handoff(_envelope())
    assert reply == {"refused": True, "reason": UNIFORM_REFUSAL}
    assert seen == [] and len(sent) == 1


async def test_a_refused_return_never_sends_the_result() -> None:
    sent: list[dict] = []
    seen: list[list[Any]] = []
    reply = await _child(sent, seen, block=lambda b: _is(b, "handoff_return")).areceive_handoff(
        _envelope()
    )
    assert reply == {"refused": True, "reason": UNIFORM_REFUSAL}
    assert "answer" not in json.dumps(reply)


async def test_without_a_scope_both_sides_use_the_multi_agent_session() -> None:
    sent: list[dict] = []
    await _child(sent, []).areceive_handoff(_envelope(scope=None))
    assert {b.get("scope") for b in sent} == {"mas:mas-1"}
    assert sent[-1]["participants"] == [{"kind": "agent_session", "id": "agent-parent/mas:mas-1"}]


async def test_a_block_inside_the_child_comes_back_as_the_uniform_refusal() -> None:
    sent: list[dict] = []

    def blocked(state: MessagesState) -> dict[str, Any]:
        raise GovernanceBlockedError("block", "Action blocked by policy", "x")

    g = StateGraph(MessagesState)
    g.add_node("answer", blocked)
    g.add_edge(START, "answer")
    g.add_edge("answer", END)
    child = OpenBoxLangGraphHandler(g.compile(), OpenBoxLangGraphHandlerOptions(client=_core(sent)))
    assert (await child.areceive_handoff(_envelope()))["refused"] is True


async def test_not_an_envelope_is_rejected() -> None:
    with pytest.raises(ValueError):
        await _child([], []).areceive_handoff({"task": "x"})


# ── Round trip ───────────────────────────────────────────────────


@pytest.mark.parametrize("block_return", [False, True], ids=["allowed", "return-refused"])
async def test_parent_to_child_and_back(block_return: bool) -> None:
    sent: list[dict] = []
    child = _child(sent, [], block=lambda b: block_return and _is(b, "handoff_return"))
    tool = _tool(child.areceive_handoff)
    with exposure.exposure_scope("matter:0417"):
        out = await tool.ainvoke({"task": "find the memo"})
    assert out == (UNIFORM_REFUSAL if block_return else "answer: find the memo")
