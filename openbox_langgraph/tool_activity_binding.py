"""Bind the base-SDK ``ActivityContext`` around ACTUAL tool execution, at the
LangGraph ``ToolNode`` request boundary — the one seam where the tool-run id
can be made canonical.

Why the ToolNode seam, not ``tool.func`` / ``tool.coroutine``
-------------------------------------------------------------
The deep ``func``/``coroutine`` boundary is too late: LangChain has already
minted the tool-run id above it, so a wrapper there cannot make the id it binds
equal the ``on_tool_start`` run id the handler maps to ``ActivityStarted``.
``ToolNode`` exposes ``wrap_tool_call`` / ``awrap_tool_call`` seams that run
BEFORE the tool run id exists — and, crucially, before ``execute(request)``
mints it.

Canonical id by WRITE, not read
--------------------------------
``ToolCallRequest.runtime.config["run_id"]`` is empty at the wrapper seam (the
run id is generated inside ``execute()`` by ``BaseTool.arun``, after the
wrapper runs — verified empirically). So this binder does the inverse: it MINTS
a canonical id and WRITES it into ``request.runtime.config["run_id"]`` before
calling ``execute``. ``BaseTool`` then forwards that id into the run it starts,
so the ``on_tool_start`` event — hence ``ToolStarted.activity_id`` — carries the
SAME id this binder bound its ``ActivityContext`` with. One id for the tool
lifecycle and every hook span fired inside it.

Concurrency-safe by construction
---------------------------------
Each tool call gets its OWN ``ToolCallRequest`` (its own ``runtime.config``
dict) and its OWN ``activity_scope`` bound in its OWN execution context, so two
tool calls in one ``ToolNode`` — or two turns on one handler — never see each
other's context (verified: async bodies, sync executor threads, and concurrent
calls all resolve their own id). This replaces the earlier module-level
ContextVar, which cross-contaminated concurrent turns.

Scope
-----
This module owns NO hook payloads and NO span construction — it only supplies
the right context; hook instrumentation stays entirely in ``openbox_core``.
Trace registration in ``langgraph_handler`` remains a best-effort EXACT backup
for child work that ContextVars cannot reach (e.g. a raw thread spawned inside
a tool).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from openbox_core.context import activity_scope

from openbox_langgraph.activity_context_binding import build_activity_context
from openbox_langgraph.config import GovernanceConfig
from openbox_langgraph.errors import OpenBoxConfigError
from openbox_langgraph.types import safe_serialize

if TYPE_CHECKING:
    from openbox_core.contracts.context import ActivityContext
    from openbox_core.runtime import OpenBoxRuntime

__all__ = ["TURN_METADATA_KEY", "bind_tools_activity_scope", "turn_metadata"]

_logger = logging.getLogger(__name__)

# Namespaced key under RunnableConfig["metadata"] carrying {workflow_id, run_id}
# for the current turn. Namespaced so it never collides with user metadata.
TURN_METADATA_KEY = "__openbox_activity"

# Marks a ToolNode instance whose wrappers this binder has already composed —
# idempotent so building a second handler over the same graph never re-wraps.
_BIND_MARK = "_openbox_activity_bound"


def turn_metadata(workflow_id: str, run_id: str) -> dict[str, dict[str, str]]:
    """The metadata entry the handler merges into a turn's RunnableConfig.

    Carried down to each tool by LangGraph and read back by the wrapper from
    ``ToolCallRequest.runtime.config`` — a per-invocation channel
    (concurrency-safe), not a shared ContextVar.
    """
    return {TURN_METADATA_KEY: {"workflow_id": workflow_id, "run_id": run_id}}


def bind_tools_activity_scope(
    graph: Any,
    *,
    core_runtime: OpenBoxRuntime,
    config: GovernanceConfig,
    resolve_tool_type: Callable[[str], str | None],
) -> None:
    """Compose an OpenBox activity-scope wrapper onto every ``ToolNode`` in ``graph``.

    Idempotent per ToolNode. Best-effort: a graph shape this can't introspect is
    left to the trace-registration backup — logged, never raised, so handler
    construction never fails on an exotic graph.

    Binding closes over THIS runtime's store; a compiled graph should be wrapped
    by ONE handler (see ``langgraph_handler``'s construction note).
    """
    store = core_runtime.context_store
    try:
        tool_nodes = list(_iter_tool_nodes(graph))
    except Exception as exc:  # never break construction on an unexpected graph shape
        _logger.debug("tool activity-scope binding skipped: cannot introspect graph (%s)", exc)
        return
    for tool_node in tool_nodes:
        _bind_tool_node(tool_node, store, config, resolve_tool_type)


def _iter_tool_nodes(graph: Any) -> Any:
    """Yield the ``ToolNode`` instances nested inside a compiled graph's nodes."""
    from langgraph.prebuilt import ToolNode

    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict):
        return
    seen: set[int] = set()
    yielded: set[int] = set()
    for node in nodes.values():
        tool_node = _find_tool_node(node, ToolNode, seen, depth=0)
        if tool_node is not None and id(tool_node) not in yielded:
            yielded.add(id(tool_node))
            yield tool_node


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


def _turn_ids(request: Any) -> dict[str, str] | None:
    """This tool call's {workflow_id, run_id} from its own RunnableConfig, or None.

    ``ToolCallRequest.runtime.config`` is the tool call's own config (the SAME
    dict ``execute`` forwards to the tool), so the turn ids the handler injected
    per turn are read here without a shared ContextVar.
    """
    runtime = getattr(request, "runtime", None)
    config = getattr(runtime, "config", None) if runtime is not None else None
    if not isinstance(config, dict):
        return None
    turn = (config.get("metadata") or {}).get(TURN_METADATA_KEY)
    return turn if isinstance(turn, dict) else None


def _prepare_binding(
    request: Any,
    config: GovernanceConfig,
    resolve_tool_type: Callable[[str], str | None],
) -> tuple[ActivityContext | None, Callable[[], None] | None]:
    """Mint the canonical activity id and build the ``ActivityContext``, or
    ``(None, None)`` when this tool cannot be bound.

    Returns ``(ctx, install_run_id)``: ``install_run_id`` WRITES the canonical
    id into the tool call's ``config["run_id"]`` so the ``on_tool_start`` event
    — thus ``ToolStarted.activity_id`` — matches ``ctx.activity_id`` exactly.
    It is called once here and MUST be re-called before every delegate
    ``execute(request)`` invocation: ``BaseTool`` CONSUMES the id with
    ``config.pop("run_id")`` per run, and a user wrapper may call ``execute``
    more than once (retries) — without the re-install, the second run would
    mint a fresh LangChain id while the bound scope still carries the first.

    ``(None, None)`` means "no governed turn": the caller executes the tool
    UNBOUND (default) or, under ``strict_activity_context``, has already
    raised. Never mints an id it does not also write.
    """
    tool_call = getattr(request, "tool_call", None) or {}
    name = tool_call.get("name") or "tool"
    turn = _turn_ids(request)
    if turn is None:
        _handle_missing_binding(config, name, tool_call.get("id"))
        return None, None

    # runtime.config is a dict here (verified by _turn_ids returning non-None) —
    # the SAME dict the ToolNode ``execute`` closure forwards to the tool run.
    cfg = request.runtime.config
    canonical = uuid.uuid4()

    def install_run_id() -> None:
        cfg["run_id"] = canonical

    install_run_id()
    args = tool_call.get("args")
    ctx = build_activity_context(
        config=config,
        workflow_id=turn.get("workflow_id", ""),
        run_id=turn.get("run_id", ""),
        activity_id=str(canonical),
        activity_type=name,
        activity_input=safe_serialize(args) if args else None,
        tool_type=resolve_tool_type(name),
        tool_name=name,
        tool_call_id=tool_call.get("id"),
    )
    return ctx, install_run_id


def _handle_missing_binding(
    config: GovernanceConfig, tool_name: str, tool_call_id: str | None
) -> None:
    """Strict mode raises; default mode warns once and lets the tool run unbound.

    The message carries enough to fix a missing binding: tool name, tool_call_id,
    and that no governed-turn metadata reached this tool call.
    """
    detail = (
        f"tool={tool_name!r} tool_call_id={tool_call_id!r}: no governed-turn "
        f"metadata on the tool RunnableConfig — cannot prove the ActivityContext "
        f"(run outside an OpenBox-governed turn, or turn metadata did not propagate)"
    )
    if config.strict_activity_context:
        raise OpenBoxConfigError(
            f"strict_activity_context: refusing to run an unbound tool. {detail}"
        )
    _logger.warning(
        "openbox_langgraph: executing tool UNBOUND — its hook spans will not "
        "attach to any activity. %s",
        detail,
    )


def _bind_tool_node(
    tool_node: Any,
    store: Any,
    config: GovernanceConfig,
    resolve_tool_type: Callable[[str], str | None],
) -> None:
    """Compose OpenBox binding onto a ToolNode's ``_wrap_tool_call`` /
    ``_awrap_tool_call`` seams, preserving any user-provided wrapper.

    A user wrapper runs INSIDE the bound scope and receives a re-installing
    ``execute`` (see ``_prepare_binding`` — the canonical id must be re-written
    before every call because ``BaseTool`` pops it per run).

    Seam installation mirrors LangGraph's own dispatch: ``_awrap_tool_call`` is
    installed UNLESS the user provided a sync-only wrapper. LangGraph's
    ``_arun_one`` falls back to calling the SYNC wrapper (with a sync execute
    shim) when no async wrapper exists — installing ours unconditionally would
    intercept that fallback and silently skip the user's sync wrapper on the
    async path. Leaving the seam unset keeps async execution routing through
    ``openbox_sync`` (which composes the user wrapper), exactly as stock
    LangGraph does; the sync wrapper runs inline on the event-loop thread, so
    the ContextVar bind still reaches the tool body.
    """
    if getattr(tool_node, _BIND_MARK, False):
        return
    existing_sync: Callable[..., Any] | None = getattr(tool_node, "_wrap_tool_call", None)
    existing_async: Callable[..., Awaitable[Any]] | None = getattr(
        tool_node, "_awrap_tool_call", None
    )

    def openbox_sync(request: Any, execute: Callable[[Any], Any]) -> Any:
        ctx, install_run_id = _prepare_binding(request, config, resolve_tool_type)
        if ctx is None:
            if existing_sync is not None:
                return existing_sync(request, execute)
            return execute(request)
        assert install_run_id is not None  # paired with ctx by construction

        def reexecute(req: Any) -> Any:
            install_run_id()  # BaseTool popped it — re-install for THIS run
            return execute(req)

        with activity_scope(ctx, store=store):
            if existing_sync is not None:
                return existing_sync(request, reexecute)
            return reexecute(request)

    async def openbox_async(request: Any, execute: Callable[[Any], Awaitable[Any]]) -> Any:
        ctx, install_run_id = _prepare_binding(request, config, resolve_tool_type)
        if ctx is None:
            if existing_async is not None:
                return await existing_async(request, execute)
            return await execute(request)
        assert install_run_id is not None  # paired with ctx by construction

        async def reexecute(req: Any) -> Any:
            install_run_id()  # BaseTool popped it — re-install for THIS run
            return await execute(req)

        with activity_scope(ctx, store=store):
            if existing_async is not None:
                return await existing_async(request, reexecute)
            return await reexecute(request)

    tool_node._wrap_tool_call = openbox_sync
    if existing_sync is None or existing_async is not None:
        tool_node._awrap_tool_call = openbox_async
    tool_node._openbox_activity_bound = True
