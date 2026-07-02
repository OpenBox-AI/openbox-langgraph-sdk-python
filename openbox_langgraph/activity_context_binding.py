"""Dual-write ActivityContext registration + per-turn cleanup for the opt-in
base-SDK core runtime.

`langgraph_handler.py` already registers trace/activity correlation with the
legacy `WorkflowSpanProcessor` at every tool/LLM start and clears it at
completion — that dual-write's counterpart lands here so the (opt-in, later
phase) core hook runtime can resolve the SAME activity via
`TraceContextRegistry` without the legacy processor's behavior changing by one
line. This module is INERT unless a handler has an actual core runtime — see
`should_dual_write`.

Trace-only, never a ContextVar bind: LangGraph tool/LLM execution runs inside
a spawned `asyncio.Task` (or, for sync tools, a `run_in_executor` worker
thread) that already snapshot its ContextVar chain before this module's
registration call can run — a bind here can never reach that code. See
`core_runtime.py`'s module docstring and `tests/test_contextvars_propagation.py`
for the empirical proof.
"""

from __future__ import annotations

from typing import Any

from openbox_core.contracts.context import ActivityContext
from openbox_core.runtime import OpenBoxRuntime

from openbox_langgraph.config import GovernanceConfig
from openbox_langgraph.core_runtime import get_trace_registry

__all__ = [
    "build_activity_context",
    "register_activity",
    "should_dual_write",
    "unregister_activity",
]


def should_dual_write(core_runtime: OpenBoxRuntime | None) -> bool:
    """True only when the handler owns a real core runtime.

    Handlers built with an injected `client` (e.g. a test double, or a
    subclass overriding `evaluate_event`) have `core_runtime is None` by
    construction (`langgraph_handler.py.__init__`) — this dual-write is
    legacy-only for them, exactly like the gate-routing it mirrors.
    """
    return core_runtime is not None


def build_activity_context(
    *,
    config: GovernanceConfig,
    workflow_id: str,
    run_id: str,
    activity_id: str,
    activity_type: str,
    activity_input: Any = None,
    langgraph_node: str | None = None,
    langgraph_step: int | None = None,
    tool_type: str | None = None,
    tool_name: str | None = None,
    subagent_name: str | None = None,
    parent_ids: list[str] | None = None,
) -> ActivityContext:
    """Map one LangGraph activity boundary onto the base SDK's `ActivityContext`.

    `workflow_id`/`run_id` are the PER-TURN ids `langgraph_handler.py` mints in
    `ainvoke`/`astream_governed`/`astream`/`astream_events` (NOT LangGraph's own
    per-node `run_id`, which becomes `activity_id` instead) — matching the
    legacy governance event's own workflow_id/run_id fields exactly, so a hook
    resolving this context reports the same turn a dashboard operator sees
    from the legacy path.
    """
    metadata: dict[str, Any] = {}
    if langgraph_node is not None:
        metadata["node"] = langgraph_node
    if langgraph_step is not None:
        metadata["step"] = langgraph_step
    if tool_type is not None:
        metadata["tool_type"] = tool_type
    if tool_name is not None:
        metadata["tool_name"] = tool_name
    if subagent_name is not None:
        metadata["subagent_name"] = subagent_name
    if parent_ids:
        metadata["parent_ids"] = list(parent_ids)

    return ActivityContext(
        workflow_id=workflow_id,
        run_id=run_id,
        workflow_type=config.agent_name or "LangGraphRun",
        task_queue=config.task_queue or "langgraph",
        activity_id=activity_id,
        activity_type=activity_type,
        activity_input=activity_input,
        agent_name=config.agent_name,
        session_id=config.session_id,
        multi_agent_session_id=config.multi_agent_session_id,
        metadata=metadata,
    )


def register_activity(
    core_runtime: OpenBoxRuntime | None,
    trace_id: int,
    ctx: ActivityContext,
) -> None:
    """Trace-only dual-write into the runtime's private `TraceContextRegistry`.

    No-op when `core_runtime` is `None` (injected-client handlers) — see
    `should_dual_write`. Skips a zero/falsy `trace_id` the same way the legacy
    `if trace_id:` guard at the call site already does, so a degraded OTel
    span (no active provider) never registers a bogus all-zero correlation.
    """
    if core_runtime is None or not trace_id:
        return
    get_trace_registry(core_runtime).register(trace_id, ctx)


def unregister_activity(core_runtime: OpenBoxRuntime | None, trace_id: int | None) -> None:
    """Counterpart cleanup for `register_activity` — activity completion."""
    if core_runtime is None or not trace_id:
        return
    get_trace_registry(core_runtime).unregister(trace_id)
