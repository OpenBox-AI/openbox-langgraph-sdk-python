"""Trace-context registration: the base-SDK `TraceContextRegistry` (over the
core runtime's private `ContextStore`) resolves the SAME activity identity for
the trace id the handler generated for a given tool/LLM call.

Registration happens at every tool/LLM start/completion boundary in
`langgraph_handler.py`, trace-only (no ContextVar bind — see
`trace_context_registry.py`'s module docstring for why a bind here can never
reach a spawned LangGraph tool task). It is only populated when the handler
owns a real core runtime; injected-client handlers are lifecycle-only and
register nothing (and create no bridging OTel span).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any
from uuid import uuid4

import pytest

pytest.importorskip("openbox_core")

from langchain_core.messages import HumanMessage
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.core_runtime import create_core_runtime
from openbox_langgraph.langgraph_handler import (
    OpenBoxLangGraphHandler,
    OpenBoxLangGraphHandlerOptions,
    _GuardrailsCallbackHandler,
    _RootRunTracker,
    _RunBufferManager,
)
from openbox_langgraph.types import GovernanceVerdictResponse, LangGraphStreamEvent, Verdict

# NOTE: the `_unconfigured_global_state` (autouse) and `_ensure_recording_tracer_provider`
# fixtures this module depends on live in `tests/conftest.py` — shared across
# every context-binding test module. See that file's docstring for why both
# are required (global-config OTel-pollution hang + degraded trace_id=0
# without a real TracerProvider).


class _AllowEverythingClient(GovernanceClient):
    """Always-ALLOW stub — isolates the test to context-binding, not verdict logic."""

    def __init__(self) -> None:
        super().__init__(api_url="https://core.openbox.ai", api_key="obx_test_abc")

    async def evaluate_event(self, event: Any) -> GovernanceVerdictResponse | None:
        return GovernanceVerdictResponse(verdict=Verdict.ALLOW)


@pytest.fixture
def make_handler() -> Iterator[Callable[..., OpenBoxLangGraphHandler]]:
    """Factory for handlers wired with a real core runtime, without touching
    global config or making network calls.

    An injected `client` keeps `__init__` from building its own core runtime;
    `_core_runtime` is then assigned directly — the same attribute `__init__`'s
    non-injected branch would build, but without resolving global config. Base
    instrumentation is the only hook runtime, so each runtime installs it; the
    fixture closes every runtime at teardown to uninstall (no global leak).
    """
    created: list[OpenBoxLangGraphHandler] = []

    def _factory(*, multi_agent_session_id: str | None = None) -> OpenBoxLangGraphHandler:
        handler = OpenBoxLangGraphHandler(
            graph=None,
            options=OpenBoxLangGraphHandlerOptions(
                client=_AllowEverythingClient(),
                multi_agent_session_id=multi_agent_session_id,
            ),
        )
        handler._core_runtime = create_core_runtime(  # type: ignore[attr-defined]
            handler._config,  # type: ignore[attr-defined]
            api_url="https://core.openbox.ai",
            api_key="obx_test_abc",
            governance_timeout=30.0,
        )
        created.append(handler)
        return handler

    yield _factory

    for handler in created:
        if handler._core_runtime is not None:  # type: ignore[attr-defined]
            handler._core_runtime.close()  # type: ignore[attr-defined]


def _tool_start_event(run_id: str) -> LangGraphStreamEvent:
    return LangGraphStreamEvent(
        event="on_tool_start",
        name="search_web",
        run_id=run_id,
        metadata={"langgraph_node": "agent", "langgraph_step": 1},
        data={"input": {"query": "weather"}},
        parent_ids=["parent-run-1"],
    )


def _tool_end_event(run_id: str) -> LangGraphStreamEvent:
    return LangGraphStreamEvent(
        event="on_tool_end",
        name="search_web",
        run_id=run_id,
        metadata={"langgraph_node": "agent", "langgraph_step": 1},
        data={"output": "sunny"},
        parent_ids=["parent-run-1"],
    )


@pytest.mark.asyncio
async def test_tool_start_registers_context_in_base_registry(make_handler) -> None:
    """After ToolStarted processing, the base ContextStore resolves the
    workflow_id/activity_id/activity_type for the trace id the handler generated,
    via its EXACT trace-id tier (the only registration path — no fallback)."""
    handler = make_handler()
    workflow_id, run_id = "wf-bind-1", "run-bind-1"
    event_run_id = "tool-run-1"
    root_tracker, buffer = _RootRunTracker(), _RunBufferManager()

    await handler._process_event(
        _tool_start_event(event_run_id), "thread-1", workflow_id, run_id, root_tracker, buffer
    )

    buf = buffer.get(event_run_id)
    assert buf is not None and buf.otel_span is not None
    trace_id = buf.otel_span.get_span_context().trace_id
    assert trace_id  # a real (non-zero) trace id was generated

    # EXACT trace lookup on the base store — `register_activity`'s dual-write
    # into `ContextStore.register_trace` is what this resolves.
    core_ctx = handler._core_runtime.context_store.context_for_trace(trace_id)  # type: ignore[union-attr]
    assert core_ctx is not None
    assert core_ctx.workflow_id == workflow_id
    assert core_ctx.run_id == run_id
    assert core_ctx.activity_id == event_run_id
    assert core_ctx.activity_type == "search_web"


@pytest.mark.asyncio
async def test_tool_completion_updates_then_clears_base_store(make_handler) -> None:
    """ToolCompleted: the base store's trace binding is updated to the completed
    identity (same activity_id as started — the lifecycle shares one id), THEN
    unregistered (span lifetime ends here) — see langgraph_handler.py's
    on_tool_end comment for why update-then-unregister happens in that order."""
    handler = make_handler()
    workflow_id, run_id = "wf-bind-2", "run-bind-2"
    event_run_id = "tool-run-2"
    root_tracker, buffer = _RootRunTracker(), _RunBufferManager()

    await handler._process_event(
        _tool_start_event(event_run_id), "thread-1", workflow_id, run_id, root_tracker, buffer
    )
    buf = buffer.get(event_run_id)
    assert buf is not None and buf.otel_span is not None
    trace_id = buf.otel_span.get_span_context().trace_id

    await handler._process_event(
        _tool_end_event(event_run_id), "thread-1", workflow_id, run_id, root_tracker, buffer
    )

    # Base store: the trace is unregistered — no longer resolvable by exact match.
    assert handler._core_runtime.context_store.context_for_trace(trace_id) is None  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_llm_start_registers_distinct_trace_from_tool(make_handler) -> None:
    """LLM calls create their OWN OTel span/trace_id (distinct from any parent
    tool's) — the registration for on_chat_model_start must register THAT trace,
    not silently reuse or collide with a concurrently active tool's trace."""
    handler = make_handler()
    workflow_id, run_id = "wf-bind-3", "run-bind-3"
    root_tracker, buffer = _RootRunTracker(), _RunBufferManager()

    llm_event = LangGraphStreamEvent(
        event="on_chat_model_start",
        name="gpt-4",
        run_id="llm-run-1",
        metadata={"langgraph_node": "agent", "langgraph_step": 2},
        data={"input": {"messages": [[{"type": "human", "content": "hi"}]]}},
        parent_ids=["parent-run-1"],
    )
    await handler._process_event(
        llm_event, "thread-1", workflow_id, run_id, root_tracker, buffer
    )

    buf = buffer.get("llm-run-1")
    assert buf is not None and buf.otel_span is not None
    trace_id = buf.otel_span.get_span_context().trace_id

    core_ctx = handler._core_runtime.context_store.context_for_trace(trace_id)  # type: ignore[union-attr]
    assert core_ctx is not None
    assert core_ctx.activity_type == "llm_call"
    assert core_ctx.activity_id == "llm-run-1"
    assert core_ctx.metadata.get("subagent_name") is None  # no subagent on this call


@pytest.mark.asyncio
async def test_pre_llm_callback_registers_active_trace_before_stream_event(make_handler) -> None:
    """The guardrails callback fires before the real chat-model HTTP request.
    It must register that active OTel trace immediately, because the later
    LangGraph stream event can arrive too late for started/preflight hooks."""
    handler = make_handler()
    workflow_id, run_id = "wf-callback-bind", "run-callback-bind"
    callback_run_id = uuid4()
    llm_activity_map: dict[str, str] = {}
    llm_trace_map: dict[str, int] = {}
    cb = _GuardrailsCallbackHandler(
        client=handler._client,  # type: ignore[attr-defined]
        config=handler._config,  # type: ignore[attr-defined]
        workflow_id=workflow_id,
        run_id=run_id,
        thread_id="thread-callback-bind",
        llm_activity_map=llm_activity_map,
        llm_trace_map=llm_trace_map,
        core_runtime=handler._core_runtime,  # type: ignore[attr-defined]
    )

    parent_span = otel_trace.get_tracer("test-pre-llm-callback").start_span("upstream")
    parent_token = otel_context.attach(otel_trace.set_span_in_context(parent_span))
    parent_trace_id = parent_span.get_span_context().trace_id
    try:
        await cb.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[HumanMessage(content="hello")]],
            run_id=callback_run_id,
            metadata={"langgraph_node": "agent", "langgraph_step": 7},
        )

        event_run_id = str(callback_run_id)
        assert llm_activity_map[event_run_id] == event_run_id
        trace_id = llm_trace_map[event_run_id]
        assert trace_id
        assert trace_id != parent_trace_id
        assert otel_trace.get_current_span().get_span_context().trace_id == trace_id

        core_ctx = handler._core_runtime.context_store.context_for_trace(trace_id)  # type: ignore[union-attr]
        assert core_ctx is not None
        assert core_ctx.workflow_id == workflow_id
        assert core_ctx.run_id == run_id
        assert core_ctx.activity_id == event_run_id
        assert core_ctx.activity_type == "llm_call"
        assert core_ctx.activity_input == [{"prompt": "hello"}]
        assert core_ctx.metadata.get("node") == "agent"
        assert core_ctx.metadata.get("step") == 7

        root_tracker, buffer = _RootRunTracker(), _RunBufferManager()
        llm_event = LangGraphStreamEvent(
            event="on_chat_model_start",
            name="ChatOpenAI",
            run_id=event_run_id,
            metadata={"langgraph_node": "agent", "langgraph_step": 7},
            data={"input": {"messages": [[{"type": "human", "content": "hello"}]]}},
            parent_ids=[],
        )
        await handler._process_event(
            llm_event,
            "thread-callback-bind",
            workflow_id,
            run_id,
            root_tracker,
            buffer,
            llm_activity_map=llm_activity_map,
            llm_trace_map=llm_trace_map,
        )
        buf = buffer.get(event_run_id)
        assert buf is not None
        assert buf.llm_started is True
        assert buf.otel_span is None, "stream fallback must not create a competing LLM span"
        assert handler._core_runtime.context_store.context_for_trace(trace_id) == core_ctx  # type: ignore[union-attr]

        await cb.on_llm_end(response=None, run_id=callback_run_id)
        assert handler._core_runtime.context_store.context_for_trace(trace_id) is None  # type: ignore[union-attr]
        assert otel_trace.get_current_span().get_span_context().trace_id == parent_trace_id
    finally:
        if str(callback_run_id) in llm_trace_map:
            await cb.on_llm_error(RuntimeError("cleanup"), run_id=callback_run_id)
        otel_context.detach(parent_token)
        parent_span.end()


@pytest.mark.asyncio
async def test_multi_agent_session_id_threads_into_registered_context(make_handler) -> None:
    """`OpenBoxLangGraphHandlerOptions.multi_agent_session_id` flows through
    `merge_config` -> `GovernanceConfig.multi_agent_session_id` ->
    `build_activity_context` -> `ActivityContext.multi_agent_session_id`,
    kept SEPARATE from `session_id` end to end (never merged into it)."""
    handler = make_handler(multi_agent_session_id="mas-42")
    assert handler._config.multi_agent_session_id == "mas-42"  # type: ignore[attr-defined]
    assert handler._config.session_id is None  # type: ignore[attr-defined]

    workflow_id, run_id = "wf-bind-5", "run-bind-5"
    event_run_id = "tool-run-5"
    root_tracker, buffer = _RootRunTracker(), _RunBufferManager()

    await handler._process_event(
        _tool_start_event(event_run_id), "thread-1", workflow_id, run_id, root_tracker, buffer
    )

    buf = buffer.get(event_run_id)
    assert buf is not None and buf.otel_span is not None
    trace_id = buf.otel_span.get_span_context().trace_id

    core_ctx = handler._core_runtime.context_store.context_for_trace(trace_id)  # type: ignore[union-attr]
    assert core_ctx is not None
    assert core_ctx.multi_agent_session_id == "mas-42"
    assert core_ctx.session_id is None


@pytest.mark.asyncio
async def test_injected_client_is_lifecycle_only_no_span_no_registration() -> None:
    """Injected-client handlers never build a core runtime — they are
    lifecycle-only: no bridging OTel span is created and nothing is registered
    (see `activity_context_binding.should_dual_write`)."""
    handler = OpenBoxLangGraphHandler(
        graph=None, options=OpenBoxLangGraphHandlerOptions(client=_AllowEverythingClient())
    )
    assert handler._core_runtime is None  # type: ignore[attr-defined]
    assert handler._span_processor is None  # type: ignore[attr-defined]

    root_tracker, buffer = _RootRunTracker(), _RunBufferManager()
    await handler._process_event(
        _tool_start_event("tool-run-3"), "thread-1", "wf-bind-4", "run-bind-4",
        root_tracker, buffer,
    )

    buf = buffer.get("tool-run-3")
    # Lifecycle-only path: no bridging OTel span was created for the tool call.
    assert buf is not None and buf.otel_span is None
