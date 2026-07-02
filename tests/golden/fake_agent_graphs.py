"""Minimal compiled LangGraph graphs driven by a fake chat model, for golden capture.

Separated from graph_capture_harness.py (the recording/capture machinery) to
keep each module focused and under the project's file-size guideline.

Uses `FakeMessagesListChatModel` so no network LLM call is made — this
mirrors how the rest of the suite avoids live LLM calls while still
exercising the REAL handler/event-mapping/dedup code paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class _AgentState(TypedDict):
    """Minimal LangGraph state: a running message list."""

    messages: Annotated[list[BaseMessage], add_messages]


@tool
def echo_tool(text: str) -> str:
    """Echo the given text back (deterministic no-op tool for golden capture)."""
    return f"echo: {text}"


def build_single_llm_graph(responses: Sequence[AIMessage]) -> Any:
    """Compile a one-node graph: human input -> fake LLM reply -> END.

    `responses` is consumed in order across every `ainvoke` on the fake model
    instance — build a fresh graph per turn if you need a fresh response cursor.
    """
    model_responses: list[BaseMessage] = list(responses)
    model = FakeMessagesListChatModel(responses=model_responses)

    async def call_model(state: _AgentState) -> dict[str, Any]:
        result = await model.ainvoke(state["messages"])
        return {"messages": [result]}

    graph = StateGraph(_AgentState)
    graph.add_node("agent", call_model)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()


def build_tool_call_graph() -> Any:
    """Compile a two-node agent/tools graph that makes exactly one tool round-trip.

    Turn shape: LLM call (emits a tool_calls request) -> echo_tool -> LLM call
    (final answer, no more tool calls) -> END. Used for the subagent-labelled
    ToolStarted/ToolCompleted capture via `resolve_subagent_name`.
    """
    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "echo_tool", "args": {"text": "hi"}, "id": "call-1"}],
            ),
            AIMessage(content="done"),
        ]
    )

    async def call_model(state: _AgentState) -> dict[str, Any]:
        result = await model.ainvoke(state["messages"])
        return {"messages": [result]}

    def should_continue(state: _AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(_AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode([echo_tool]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def build_two_llm_call_graph() -> Any:
    """One node, two sequential model.ainvoke() calls in a single turn.

    Exercises the handler's skip-based LLMStarted dedup (langgraph_handler.py
    skips re-sending LLMStarted from _process_event because the guardrails
    callback owns it) across TWO distinct LLM invocations in the same node —
    there is no keyed de-dup store, so both calls must produce independent
    LLMStarted/LLMCompleted activity rows.
    """
    model = FakeMessagesListChatModel(
        responses=[AIMessage(content="first"), AIMessage(content="second")]
    )

    async def call_model_twice(state: _AgentState) -> dict[str, Any]:
        r1 = await model.ainvoke(state["messages"])
        r2 = await model.ainvoke([*state["messages"], r1])
        return {"messages": [r1, r2]}

    graph = StateGraph(_AgentState)
    graph.add_node("agent", call_model_twice)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()


def build_error_graph() -> Any:
    """Compile a one-node graph identical to build_single_llm_graph.

    Used with a `verdict_override` (see graph_capture_harness.py) that forces
    a HALT verdict on the LLMStarted pre-screen event — the graph itself
    never runs (enforcement fails before the stream starts), so the model
    response list is irrelevant and left empty.
    """
    return build_single_llm_graph([AIMessage(content="unreachable — pre-screen halts first")])
