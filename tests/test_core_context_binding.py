"""Dual-write consistency: the legacy `WorkflowSpanProcessor` and the base-SDK
`TraceContextRegistry` (over the core runtime's private `ContextStore`) must
resolve the SAME activity identity for the SAME trace id.

Registration happens at every tool/LLM start/completion boundary in
`langgraph_handler.py`, trace-only (no ContextVar bind — see
`trace_context_registry.py`'s module docstring for why a bind here can never
reach a spawned LangGraph tool task). The legacy processor stays the
authoritative source for legacy hooks; the base store is purely additive and
only populated when the handler owns a real core runtime.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("openbox_core")

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.core_runtime import create_core_runtime, get_trace_registry
from openbox_langgraph.langgraph_handler import (
    OpenBoxLangGraphHandler,
    OpenBoxLangGraphHandlerOptions,
    _RootRunTracker,
    _RunBufferManager,
)
from openbox_langgraph.span_processor import WorkflowSpanProcessor
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


def _build_handler_with_core_runtime(
    *, multi_agent_session_id: str | None = None
) -> OpenBoxLangGraphHandler:
    """A handler wired with BOTH a real core runtime AND a legacy span processor,
    without touching global OTel state or making network calls.

    Mirrors `test_hook_approval_retry_baseline.py`'s documented technique: an
    injected `client` keeps `__init__` from calling `initialize()`-gated
    global OTel setup (`setup_opentelemetry_for_governance`), which would
    otherwise register a span processor with the process-wide TracerProvider
    every test run and leak across the pytest session. `_core_runtime` and
    `_span_processor` are then assigned directly — the same attributes
    `__init__`'s non-injected branch would have built, just without the
    network/global-state side effects that branch also carries.
    """
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
    handler._span_processor = WorkflowSpanProcessor()  # type: ignore[attr-defined]
    return handler


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
async def test_tool_start_dual_writes_same_context_to_both_stores() -> None:
    """After ToolStarted processing, the legacy processor and the base
    TraceContextRegistry resolve the SAME workflow_id/activity_id/activity_type
    for the trace id the handler generated for this tool call."""
    handler = _build_handler_with_core_runtime()
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

    legacy_ctx = handler._span_processor.get_activity_context_by_trace(trace_id)  # type: ignore[union-attr]
    assert legacy_ctx is not None
    assert legacy_ctx["workflow_id"] == workflow_id
    assert legacy_ctx["activity_id"] == event_run_id
    assert legacy_ctx["activity_type"] == "search_web"

    registry = get_trace_registry(handler._core_runtime)  # type: ignore[arg-type]
    core_ctx = registry.resolve(trace_id)
    assert core_ctx is not None
    assert core_ctx.workflow_id == workflow_id
    assert core_ctx.run_id == run_id
    assert core_ctx.activity_id == event_run_id
    assert core_ctx.activity_type == "search_web"
    # Also resolvable through the base SDK's OWN exact-trace lookup (proves
    # `ContextStore.register_trace` — not just this registry's private
    # bookkeeping — actually received the write).
    assert handler._core_runtime.context_store.context_for_trace(trace_id) == core_ctx  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_tool_completion_updates_then_clears_both_stores() -> None:
    """ToolCompleted: the legacy processor clears its activity_context entry;
    the base registry updates the trace to the `-c` completed identity, THEN
    unregisters it (span lifetime ends here) — see langgraph_handler.py's
    on_tool_end comment for why update-then-unregister happens in that order."""
    handler = _build_handler_with_core_runtime()
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

    # Legacy: activity_context cleared for this workflow_id:activity_id key.
    assert (
        handler._span_processor.get_activity_context_by_trace(trace_id) is None  # type: ignore[union-attr]
        or handler._span_processor._activity_context.get(  # type: ignore[union-attr]
            f"{workflow_id}:{event_run_id}"
        )
        is None
    )

    # Base store: the trace is unregistered — no longer resolvable by exact match.
    assert handler._core_runtime.context_store.context_for_trace(trace_id) is None  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_llm_start_dual_writes_distinct_trace_from_tool() -> None:
    """LLM calls create their OWN OTel span/trace_id (distinct from any parent
    tool's) — the dual-write for on_chat_model_start must register THAT trace,
    not silently reuse or collide with a concurrently active tool's trace."""
    handler = _build_handler_with_core_runtime()
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
async def test_multi_agent_session_id_threads_into_dual_written_context() -> None:
    """`OpenBoxLangGraphHandlerOptions.multi_agent_session_id` flows through
    `merge_config` -> `GovernanceConfig.multi_agent_session_id` ->
    `build_activity_context` -> `ActivityContext.multi_agent_session_id`,
    kept SEPARATE from `session_id` end to end (never merged into it) — the
    legacy `LangChainGovernanceEvent` has no field for it, so this dual-write
    is the only place it's currently observable."""
    handler = _build_handler_with_core_runtime(multi_agent_session_id="mas-42")
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
async def test_dual_write_skipped_when_core_runtime_is_none() -> None:
    """Injected-client handlers never build a core runtime — the dual-write
    must be a complete no-op for them, legacy-only, exactly as documented in
    `activity_context_binding.should_dual_write`."""
    handler = OpenBoxLangGraphHandler(
        graph=None, options=OpenBoxLangGraphHandlerOptions(client=_AllowEverythingClient())
    )
    handler._span_processor = WorkflowSpanProcessor()  # type: ignore[attr-defined]
    assert handler._core_runtime is None  # type: ignore[attr-defined]

    root_tracker, buffer = _RootRunTracker(), _RunBufferManager()
    await handler._process_event(
        _tool_start_event("tool-run-3"), "thread-1", "wf-bind-4", "run-bind-4",
        root_tracker, buffer,
    )

    buf = buffer.get("tool-run-3")
    assert buf is not None and buf.otel_span is not None
    trace_id = buf.otel_span.get_span_context().trace_id
    # Legacy processor still got the registration (unaffected, additive-only).
    assert handler._span_processor.get_activity_context_by_trace(trace_id) is not None  # type: ignore[union-attr]
