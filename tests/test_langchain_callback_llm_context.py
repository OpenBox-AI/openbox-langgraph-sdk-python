"""Phase 5 — LLM lifecycle ownership migrated to the shared pure-LangChain-Core
callback.

Drives a REAL `create_react_agent`-shape graph with a REAL `OpenBoxRuntime`
(fake Core transport, real gate/adapter) so the callback fires through
`astream_events`/`ainvoke` exactly as production does: `_governed_config`
installs both `OpenBoxLangChainCore{Async,Sync}CallbackHandler` with
`send_llm_start_event`/`send_llm_end_event=True`, `pre_screen_response`
(mapped to `EvaluationResult`, M18), `pre_screen_activity_id`, and
registry-backed `register_trace`/`unregister_trace` (M21).

Covers (phase-05 step 6): trace registered before the fake provider call;
LLMCompleted reuses the SAME activity_id as LLMStarted; the first call
resolves via the `event_run_id` alias with NO orphan `-c` row (H11);
pre-screen reuse (exactly one ActivityStarted for call 1); redaction mutates
the pre-call message; the consumer skips a callback-owned LLMCompleted;
the fallback close still fires when ownership is absent; the error path
sends a failed completion and unregisters the trace, guarded by
`llm_completed_sent`; a completion REQUIRE_APPROVAL polls-and-continues
WITHOUT a graph replay (C4).
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatResult
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

pytest.importorskip("openbox_core")

from openbox_core.conformance.fake_core import FakeCore
from openbox_core.conformance.instrumentation import installed_conformance_runtime
from openbox_core.context import ContextStore

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.core_adapter import LangGraphFrameworkAdapter
from openbox_langgraph.langgraph_handler import (
    OpenBoxLangGraphHandler,
    OpenBoxLangGraphHandlerOptions,
)
from openbox_langgraph.tool_activity_binding import bind_tools_activity_scope


class _AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def echo_tool(text: str) -> str:
    """Echo text back."""
    return f"echo: {text}"


def _build_single_llm_graph(responses: list[AIMessage]) -> Any:
    """A single LLM call, no tool call — the simplest chat-model turn."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    model = FakeMessagesListChatModel(responses=responses)

    async def call_model(state: _AgentState) -> dict[str, Any]:
        result = await model.ainvoke(state["messages"])
        return {"messages": [result]}

    graph = StateGraph(_AgentState)
    graph.add_node("agent", call_model)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()


def _build_two_llm_call_graph() -> Any:
    """One tool call in between two LLM calls — exercises pre-screen reuse
    for call 1 and a real (non-pre-screen) evaluate for call 2."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

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


def _build_handler_with_bridge(graph: Any, runtime: Any, *, use_gate_client: bool = True) -> Any:
    """C1-armed handler wired the SAME way `_governed_config` wires it in
    production: injected `client` skips `__init__`'s own runtime build,
    `_core_runtime` is pointed at the conformance runtime, and the bridge +
    ToolNode binding are wired explicitly.

    `use_gate_client=True` (default) routes `_pre_screen_input`'s OWN
    `evaluate_event` calls through the SAME `runtime.gate` the callback
    uses — so a single `FakeCore` verdict queue drives BOTH the pre-screen
    call and the callback's calls, letting a test assert pre-screen REUSE
    (one ActivityStarted, not two) against one shared queue.
    """
    from openbox_langchain import ActivityBridge

    if use_gate_client:
        client = GovernanceClient(
            api_url="https://core.openbox.ai", api_key="obx_test_abc", gate=runtime.gate
        )
    else:
        client = GovernanceClient(api_url="https://core.openbox.ai", api_key="obx_test_abc")
    handler = OpenBoxLangGraphHandler(
        graph=graph, options=OpenBoxLangGraphHandlerOptions(client=client)
    )
    handler._core_runtime = runtime  # type: ignore[attr-defined]
    handler._activity_bridge = ActivityBridge()  # type: ignore[attr-defined]
    bind_tools_activity_scope(
        graph,
        core_runtime=runtime,
        config=handler._config,  # type: ignore[attr-defined]
        resolve_tool_type=lambda name: None,
        bridge=handler._activity_bridge,  # type: ignore[attr-defined]
    )
    return handler


def _llm_payloads(fake_core: FakeCore, event_type: str) -> list[dict[str, Any]]:
    """Lifecycle (non-hook) payloads of `event_type` sent by the callback.

    No `spans` field on lifecycle envelopes (see
    `test_langchain_callback_tool_ordering.py`'s identical rationale) — the
    flat `event_type` wire field is the discriminator instead.
    """
    return [
        p for p in fake_core.payloads if p.get("event_type") == event_type and not p.get("spans")
    ]


async def test_trace_registered_before_provider_call_and_same_id_close() -> None:
    """Trace is registered (resolvable via the base ContextStore's exact
    trace-id tier) before the fake provider "HTTP" call completes, and
    LLMCompleted closes on the SAME activity_id as LLMStarted (H11/M21) —
    the CORE-VISIBLE same-id change this phase ships, no `-c` suffix."""
    fake_core = FakeCore()  # ALLOW
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    graph = _build_single_llm_graph([AIMessage(content="hello from fake model")])

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler_with_bridge(graph, runtime)
        await handler.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "llm-ctx-same-id"}},
        )

    started = _llm_payloads(fake_core, "ActivityStarted")
    completed = _llm_payloads(fake_core, "ActivityCompleted")
    assert len(started) == 1
    assert len(completed) == 1
    assert started[0]["activity_id"] == completed[0]["activity_id"]
    assert not started[0]["activity_id"].endswith("-c")


async def test_provider_http_spans_attach_to_callback_owned_llm_activity() -> None:
    """A normal chat-model provider HTTP call needs no app tracing glue.

    The LangChain callback owns the LLM activity and creates the parent span
    before the model body runs; base HTTPX instrumentation then emits both hook
    stages under that same LLM activity.
    """
    fake_core = FakeCore()
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)

    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class _HttpModel(FakeMessagesListChatModel):
        async def _agenerate(self, messages: Any, *args: Any, **kwargs: Any) -> ChatResult:
            async def transport(request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, json={"id": "chatcmpl-test"})

            async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
                await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
                )
            return super()._generate(messages, *args, **kwargs)

    model = _HttpModel(responses=[AIMessage(content="ok")])

    async def call_model(state: _AgentState) -> dict[str, Any]:
        result = await model.ainvoke(state["messages"])
        return {"messages": [result]}

    graph = StateGraph(_AgentState)
    graph.add_node("agent", call_model)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    compiled = graph.compile()

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler_with_bridge(compiled, runtime, use_gate_client=True)
        await handler.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "llm-ctx-provider-http"}},
        )

    llm_started = _llm_payloads(fake_core, "ActivityStarted")
    assert len(llm_started) == 1
    llm_activity_id = llm_started[0]["activity_id"]

    hook_spans = [
        (payload, payload["spans"][0])
        for payload in fake_core.payloads
        if payload.get("spans")
        and payload["spans"][0].get("http_url") == "https://api.openai.com/v1/chat/completions"
    ]
    assert len(hook_spans) == 2
    assert [span["stage"] for _, span in hook_spans] == ["started", "completed"]
    assert {payload["activity_id"] for payload, _ in hook_spans} == {llm_activity_id}
    assert hook_spans[0][1]["span_id"] == hook_spans[1][1]["span_id"]


async def test_first_call_resolves_via_event_run_id_alias_no_orphan_c_row() -> None:
    """H11 — the FIRST LLM call's bridge record is keyed by the pre-screen
    activity_id (`"{run_id}-pre"`), which diverges from the callback's own
    `event_run_id`. Completion must resolve via the alias to that SAME
    pre-screen id — never emit a second, orphan `-c`-suffixed row."""
    fake_core = FakeCore()
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    graph = _build_single_llm_graph([AIMessage(content="hello from fake model")])

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler_with_bridge(graph, runtime)
        result = await handler.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "llm-ctx-alias"}},
        )

    started = _llm_payloads(fake_core, "ActivityStarted")
    completed = _llm_payloads(fake_core, "ActivityCompleted")
    assert len(started) == 1
    assert len(completed) == 1
    assert started[0]["activity_id"].endswith("-pre")
    assert completed[0]["activity_id"] == started[0]["activity_id"]
    assert not any(p["activity_id"].endswith("-c") for p in completed)
    assert result["messages"][-1].content == "hello from fake model"


async def test_pre_screen_reused_exactly_one_activity_started_for_call_one() -> None:
    """The upstream `_pre_screen_input` pre-screen verdict for the FIRST
    human-turn prompt is REUSED by the callback (M18: mapped
    `GovernanceVerdictResponse -> EvaluationResult`) — exactly ONE
    ActivityStarted reaches Core for call 1, not two (pre-screen send +
    a second callback send)."""
    fake_core = FakeCore()
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    graph = _build_single_llm_graph([AIMessage(content="hello from fake model")])

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler_with_bridge(graph, runtime, use_gate_client=True)
        await handler.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "llm-ctx-prescreen-reuse"}},
        )

    started = _llm_payloads(fake_core, "ActivityStarted")
    assert len(started) == 1, "pre-screen verdict must be REUSED, not a second independent evaluate"


async def test_redaction_mutates_pre_call_message() -> None:
    """Guardrails redaction on the pre-screened FIRST call must mutate the
    human message BEFORE the model ever sees it — in-place, pre-call."""
    # Two leading plain ALLOWs are `_pre_screen_input`'s OWN SignalReceived +
    # WorkflowStarted sends (which precede its LLMStarted pre-screen send) —
    # the SAME shared FakeCore queue backs all three, FIFO, so the
    # guardrails-carrying verdict must be queued THIRD to reach the LLMStarted
    # pre-screen rather than being consumed by an earlier send.
    fake_core = FakeCore(
        {"verdict": "allow"},
        {"verdict": "allow"},
        {
            "verdict": "allow",
            "guardrails_result": {
                "input_type": "activity_input",
                "redacted_input": [{"prompt": "[REDACTED]"}],
                "validation_passed": True,
            },
        },
    )
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)

    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    captured_prompts: list[str] = []

    class _CapturingModel(FakeMessagesListChatModel):
        # `_generate` runs INSIDE `agenerate`, AFTER the callback manager has
        # already fired `on_chat_model_start` (and therefore after the
        # callback's in-place redaction mutation) — the correct point to
        # observe what the model actually receives, unlike overriding
        # `ainvoke` (which reads messages BEFORE the callback manager runs).
        def _generate(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
            for msg in messages:
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    captured_prompts.append(content)
            return super()._generate(messages, *args, **kwargs)

    model = _CapturingModel(responses=[AIMessage(content="ok")])

    async def call_model(state: _AgentState) -> dict[str, Any]:
        result = await model.ainvoke(state["messages"])
        return {"messages": [result]}

    graph = StateGraph(_AgentState)
    graph.add_node("agent", call_model)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    compiled = graph.compile()

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler_with_bridge(compiled, runtime, use_gate_client=True)
        await handler.ainvoke(
            {"messages": [HumanMessage(content="my ssn is 123-45-6789")]},
            config={"configurable": {"thread_id": "llm-ctx-redaction"}},
        )

    assert "[REDACTED]" in captured_prompts
    assert not any("123-45-6789" in p for p in captured_prompts)


async def test_consumer_skips_callback_owned_llm_completed() -> None:
    """The consumer's OWN LLMCompleted close (the pre-phase-5 `-c` fallback)
    must be skipped entirely when the callback owns this call's completion —
    never a double ActivityCompleted for the same LLM call."""
    fake_core = FakeCore()
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    graph = _build_single_llm_graph([AIMessage(content="hello from fake model")])

    class _RecordingClient(GovernanceClient):
        def __init__(self, gate: Any) -> None:
            super().__init__(api_url="https://core.openbox.ai", api_key="obx_test_abc", gate=gate)
            self.sent_llm_completed: list[Any] = []

        async def evaluate_event(self, event: Any) -> Any:
            if event.event_type == "LLMCompleted":
                self.sent_llm_completed.append(event)
            return await super().evaluate_event(event)

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        from openbox_langchain import ActivityBridge

        client = _RecordingClient(runtime.gate)
        handler = OpenBoxLangGraphHandler(
            graph=graph, options=OpenBoxLangGraphHandlerOptions(client=client)
        )
        handler._core_runtime = runtime  # type: ignore[attr-defined]
        handler._activity_bridge = ActivityBridge()  # type: ignore[attr-defined]
        bind_tools_activity_scope(
            graph,
            core_runtime=runtime,
            config=handler._config,  # type: ignore[attr-defined]
            resolve_tool_type=lambda name: None,
            bridge=handler._activity_bridge,  # type: ignore[attr-defined]
        )
        await handler.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "llm-ctx-consumer-skip"}},
        )

    assert client.sent_llm_completed == [], "consumer must skip a callback-owned LLMCompleted"
    assert len(_llm_payloads(fake_core, "ActivityCompleted")) == 1


async def test_fallback_close_fires_when_ownership_absent() -> None:
    """No bridge armed (injected-client style — the pre-existing fallback
    scenario the spec names) -> the consumer's own `-c`-suffixed close still
    fires exactly as it did before this phase."""
    graph = _build_single_llm_graph([AIMessage(content="hello from fake model")])

    from openbox_langgraph.types import GovernanceVerdictResponse, LangChainGovernanceEvent, Verdict

    class _RecordingClient(GovernanceClient):
        def __init__(self) -> None:
            super().__init__(api_url="https://core.openbox.ai", api_key="obx_test_abc")
            self.events: list[LangChainGovernanceEvent] = []

        async def evaluate_event(self, event: LangChainGovernanceEvent) -> Any:
            self.events.append(event)
            return GovernanceVerdictResponse(verdict=Verdict.ALLOW)

    client = _RecordingClient()
    handler = OpenBoxLangGraphHandler(graph, OpenBoxLangGraphHandlerOptions(client=client))
    assert handler._core_runtime is None  # type: ignore[attr-defined]
    assert handler._activity_bridge is None  # type: ignore[attr-defined]

    await handler.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "llm-ctx-fallback"}},
    )

    completed = [e for e in client.events if e.event_type == "LLMCompleted"]
    assert len(completed) == 1
    assert completed[0].activity_id is not None and completed[0].activity_id.endswith("-c")


async def test_error_path_sends_failed_completion_and_unregisters_trace() -> None:
    """A raising chat model still gets a failed LLMCompleted from the
    callback's `on_llm_error`, and the trace binding is unregistered — same
    guard (`llm_completed_sent`) prevents `agenerate`'s
    `gather(return_exceptions=True)` capture from double-sending."""
    fake_core = FakeCore()
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)

    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class _RaisingModel(FakeMessagesListChatModel):
        # `_generate` runs INSIDE `ainvoke`'s callback-wrapped `agenerate` —
        # AFTER `on_chat_model_start` has already fired (LLMStarted sent, a
        # bridge record exists) — so raising here mirrors a real provider
        # HTTP failure mid-call and exercises `on_llm_error`, not a
        # graph-level try/except that never reaches the callback manager.
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("provider exploded")

    model = _RaisingModel(responses=[AIMessage(content="unused")])

    async def call_model(state: _AgentState) -> dict[str, Any]:
        try:
            result = await model.ainvoke(state["messages"])
        except RuntimeError:
            # Turn continues with an error message (mirrors a
            # handle_tool_errors-equivalent recovery at the node level).
            return {"messages": [AIMessage(content="error handled")]}
        return {"messages": [result]}

    graph = StateGraph(_AgentState)
    graph.add_node("agent", call_model)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    compiled = graph.compile()

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler_with_bridge(compiled, runtime)
        result = await handler.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "llm-ctx-error"}},
        )

    assert result["messages"][-1].content == "error handled"
    started = _llm_payloads(fake_core, "ActivityStarted")
    completed = _llm_payloads(fake_core, "ActivityCompleted")
    assert len(started) == 1
    # Guarded by llm_completed_sent (Phase 2): on_llm_end/on_llm_error is
    # invoked exactly once for a single failure, so exactly one failed
    # ActivityCompleted closes the SAME activity_id LLMStarted opened — no
    # double row from `agenerate`'s `gather(return_exceptions=True)` capture.
    assert len(completed) == 1
    assert completed[0]["activity_id"] == started[0]["activity_id"]
    assert completed[0].get("error") is not None


async def test_completion_approval_polls_and_continues_without_graph_replay() -> None:
    """C4 — a REQUIRE_APPROVAL verdict on LLMCompleted must poll-and-continue
    IN-LINE inside `_process_event` (never raise `GovernanceBlockedError` out
    to `ainvoke`'s outer catch/retry) — no graph replay, no re-run of the
    tool side effects that already happened in the SAME turn."""
    # Two leading ALLOWs are `_pre_screen_input`'s OWN SignalReceived +
    # WorkflowStarted sends (same shared FakeCore queue, FIFO — see
    # test_redaction_mutates_pre_call_message's identical note); the THIRD
    # is the LLMStarted pre-screen itself (reused by the callback for call 1,
    # not re-queued); the FOURTH is the callback's LLMCompleted.
    fake_core = FakeCore(
        {"verdict": "allow"},
        {"verdict": "allow"},
        {"verdict": "allow"},
        {"verdict": "require_approval", "approval_id": "app-llm-1"},
    )
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    graph = _build_single_llm_graph([AIMessage(content="hello from fake model")])

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler_with_bridge(graph, runtime, use_gate_client=True)
        with patch(
            "openbox_langgraph.langgraph_handler.poll_until_decision",
            new=AsyncMock(return_value=None),
        ) as mock_poll:
            result = await handler.ainvoke(
                {"messages": [HumanMessage(content="hi")]},
                config={"configurable": {"thread_id": "llm-ctx-completion-approval"}},
            )

    mock_poll.assert_awaited_once()
    polled_params = mock_poll.await_args.args[1]
    completed = _llm_payloads(fake_core, "ActivityCompleted")
    assert completed
    assert polled_params.activity_id == completed[0]["activity_id"]
    # The turn completed normally past the poll — proof there was no
    # graph-replay retry (a replay would re-run `call_model` a second time,
    # which this single-LLM-call graph has no branching to distinguish, but
    # a second gate call for a 2nd LLMStarted WOULD show up if it had replayed).
    assert result["messages"][-1].content == "hello from fake model"
    assert len(_llm_payloads(fake_core, "ActivityStarted")) == 1
