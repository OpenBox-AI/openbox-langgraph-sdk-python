"""Bind the base-SDK ``ActivityContext`` around ACTUAL tool execution.

Why not a callback: LangChain dispatches callbacks in an ISOLATED context
(``CallbackManager.on_tool_start`` runs handlers so their ContextVar mutations
never reach the tool body — verified empirically), and ``BaseTool.arun`` runs
the tool body inside a freshly ``copy_context()``-ed child context. So a
callback cannot put anything on the ContextVar tier the tool body reads.

How this works instead — WRAP each tool so its body runs inside
``openbox_core.context.activity_scope(ctx, store=runtime.context_store)``, the
same way Temporal's ``core_activity_scope`` binds around activity execution.
``resolve_context`` checks that ContextVar tier FIRST, so every HTTP/file/db
hook fired inside the tool resolves to THIS tool deterministically.

Per-turn ids reach the wrapper through the tool's OWN ``RunnableConfig`` — the
handler injects ``{workflow_id, run_id}`` into ``config["metadata"]`` per turn,
and ``BaseTool.arun`` binds that config on ``var_child_runnable_config`` inside
the tool body (readable via ``ensure_config``). This is per-invocation, so it is
concurrency-safe: two turns driven together on one handler never see each
other's ids (a single module ContextVar would — verified it does not here).

This module owns NO hook payloads and NO span construction — it only supplies
the right context; hook instrumentation stays entirely in ``openbox_core``.
Trace registration in ``langgraph_handler`` remains a best-effort BACKUP.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

from langchain_core.runnables.config import ensure_config
from openbox_core.context import activity_scope

from openbox_langgraph.activity_context_binding import build_activity_context
from openbox_langgraph.config import GovernanceConfig
from openbox_langgraph.types import safe_serialize

if TYPE_CHECKING:
    from openbox_core.runtime import OpenBoxRuntime

__all__ = ["TURN_METADATA_KEY", "turn_metadata", "wrap_graph_tools"]

_logger = logging.getLogger(__name__)

# Namespaced key under RunnableConfig["metadata"] carrying {workflow_id, run_id}
# for the current turn. Namespaced so it never collides with user metadata.
TURN_METADATA_KEY = "__openbox_activity"

_WRAP_MARK = "_openbox_activity_scoped"


def turn_metadata(workflow_id: str, run_id: str) -> dict[str, dict[str, str]]:
    """The metadata entry the handler merges into a turn's RunnableConfig.

    Carried down to each tool by LangGraph and read back by the wrapper via
    ``ensure_config`` — a per-invocation channel (concurrency-safe), not a
    shared ContextVar.
    """
    return {TURN_METADATA_KEY: {"workflow_id": workflow_id, "run_id": run_id}}


def wrap_graph_tools(
    graph: Any,
    *,
    core_runtime: OpenBoxRuntime,
    config: GovernanceConfig,
    resolve_tool_type: Callable[[str], str | None],
) -> None:
    """Wrap every tool in ``graph`` so its body runs inside ``activity_scope``.

    Idempotent per tool. Best-effort: a graph shape this can't introspect, or a
    tool exposing neither ``func`` nor ``coroutine`` (a custom ``BaseTool``
    overriding ``_run``/``_arun``), is left to the trace-registration backup —
    logged, never raised, so handler construction never fails on an exotic graph.

    NOTE: wrapping binds each tool to THIS runtime's store (captured in the
    closure). A compiled graph should therefore be wrapped by ONE handler; if a
    second handler (different store) reuses the same graph object, the idempotency
    mark leaves the tools bound to the first handler's store, and the second
    handler's hooks fall back to trace registration for those tools (no
    misattribution — the second store simply resolves nothing on its ContextVar
    tier and uses the backup).
    """
    store = core_runtime.context_store
    try:
        tools = list(_iter_graph_tools(graph))
    except Exception as exc:  # never break construction on an unexpected graph shape
        _logger.debug("tool activity-scope wrapping skipped: cannot introspect graph (%s)", exc)
        return
    for tool in tools:
        _wrap_tool(tool, store, config, resolve_tool_type)


def _iter_graph_tools(graph: Any) -> Any:
    """Yield the tool objects inside a compiled LangGraph graph's ToolNodes."""
    from langgraph.prebuilt import ToolNode

    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict):
        return
    seen: set[int] = set()
    for node in nodes.values():
        tool_node = _find_tool_node(node, ToolNode, seen, depth=0)
        if tool_node is None:
            continue
        yield from getattr(tool_node, "tools_by_name", {}).values()


def _find_tool_node(obj: Any, tool_node_cls: type, seen: set[int], depth: int) -> Any:
    """Walk the small set of wrapper attrs LangGraph nests a ToolNode under."""
    if obj is None or depth > 4 or id(obj) in seen:
        return None
    seen.add(id(obj))
    if isinstance(obj, tool_node_cls):
        return obj
    for attr in ("bound", "node", "runnable", "steps", "func", "_func"):
        child = getattr(obj, attr, None)
        # `steps` is a list on RunnableSequence — descend into each element.
        candidates = child if isinstance(child, (list, tuple)) else (child,)
        for candidate in candidates:
            found = _find_tool_node(candidate, tool_node_cls, seen, depth + 1)
            if found is not None:
                return found
    return None


def _current_turn() -> dict[str, str] | None:
    """This tool run's {workflow_id, run_id} from its own RunnableConfig, or None.

    ``ensure_config`` reads ``var_child_runnable_config`` — bound by
    ``BaseTool.arun`` to THIS tool run's config inside the tool body (and copied
    into the executor thread for sync tools), so it is the correct turn even
    under concurrent turns on one handler.
    """
    try:
        metadata = (ensure_config().get("metadata") or {}).get(TURN_METADATA_KEY)
    except Exception:  # ensure_config never raises today, but stay non-fatal
        return None
    return metadata if isinstance(metadata, dict) else None


def _wrap_tool(
    tool: Any,
    store: Any,
    config: GovernanceConfig,
    resolve_tool_type: Callable[[str], str | None],
) -> None:
    if getattr(tool, _WRAP_MARK, False):
        return
    name = getattr(tool, "name", None) or "tool"

    def _build_ctx(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        turn = _current_turn()
        if turn is None:  # outside a governed turn (e.g. direct tool use) — don't bind
            return None
        activity_input = safe_serialize(kwargs) if kwargs else (safe_serialize(args) or None)
        return build_activity_context(
            config=config,
            workflow_id=turn.get("workflow_id", ""),
            run_id=turn.get("run_id", ""),
            # A tool run id is not exposed at the func boundary (LangChain
            # generates it above this layer), so mint a unique id — activity
            # TYPE (the tool name) is what hook resolution keys on.
            activity_id=uuid.uuid4().hex,
            activity_type=name,
            activity_input=activity_input,
            tool_type=resolve_tool_type(name),
            tool_name=name,
        )

    func = getattr(tool, "func", None)
    coroutine = getattr(tool, "coroutine", None)
    if func is None and coroutine is None:
        _logger.debug(
            "tool %r has no func/coroutine to wrap — hook context falls back to "
            "trace registration for this tool",
            name,
        )
        return

    if func is not None:
        @wraps(func)
        def sync_wrapped(*args: Any, **kwargs: Any) -> Any:
            ctx = _build_ctx(args, kwargs)
            if ctx is None:
                return func(*args, **kwargs)
            with activity_scope(ctx, store=store):
                return func(*args, **kwargs)

        tool.func = sync_wrapped

    if coroutine is not None:
        @wraps(coroutine)
        async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
            ctx = _build_ctx(args, kwargs)
            if ctx is None:
                return await coroutine(*args, **kwargs)
            with activity_scope(ctx, store=store):
                return await coroutine(*args, **kwargs)

        tool.coroutine = async_wrapped

    tool._openbox_activity_scoped = True
