"""OpenBox LangGraph SDK — OpenBoxLangGraphHandler.

Wraps any compiled LangGraph graph and processes the v2 event stream to apply
OpenBox governance at every node, tool, and LLM invocation.

For framework-specific integrations (e.g. DeepAgents) use the dedicated
`openbox-deepagent` package which extends this handler.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, cast

from openbox_langchain import (
    ActivityBridge,
    OpenBoxLangChainCoreAsyncCallbackHandler,
    OpenBoxLangChainCoreCallbackOptions,
    OpenBoxLangChainCoreSyncCallbackHandler,
)
from openbox_langchain.activity_bridge import EventType
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace

from openbox_langgraph.activity_context_binding import (
    build_activity_context,
    register_activity,
    should_dual_write,
    unregister_activity,
)
from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.config import get_global_config, merge_config
from openbox_langgraph.core_runtime import create_core_runtime, get_trace_registry
from openbox_langgraph.errors import (
    ApprovalExpiredError,
    ApprovalRejectedError,
    ApprovalTimeoutError,
    GovernanceBlockedError,
    GovernanceHaltError,
    GuardrailsValidationError,
)
from openbox_langgraph.hitl import HITLPollParams, poll_until_decision
from openbox_langgraph.tool_activity_binding import bind_tools_activity_scope, turn_metadata
from openbox_langgraph.types import (
    GovernanceVerdictResponse,
    LangChainGovernanceEvent,
    LangGraphStreamEvent,
    rfc3339_now,
    safe_serialize,
)
from openbox_langgraph.verdict_handler import (
    enforce_verdict,
    lang_graph_event_to_context,
)

_logger = logging.getLogger(__name__)

_otel_tracer = otel_trace.get_tracer("openbox-langgraph")


def _extract_governance_blocked(exc: Exception) -> GovernanceBlockedError | None:
    """Walk exception chain to find a wrapped GovernanceBlockedError.

    LLM SDKs (OpenAI, Anthropic) wrap httpx errors. When an OTel hook raises
    GovernanceBlockedError inside httpx, the LLM SDK wraps it as APIConnectionError.
    This function unwraps the chain via __cause__ / __context__ to recover it.
    """
    cause: BaseException | None = exc
    seen: set[int] = set()
    while cause is not None:
        if id(cause) in seen:
            break
        seen.add(id(cause))
        if isinstance(cause, GovernanceBlockedError):
            return cause
        cause = getattr(cause, '__cause__', None) or getattr(cause, '__context__', None)
    return None


def _approval_poll_activity_id(hook_err: GovernanceBlockedError, run_id: str) -> str:
    """Resolve the activity id `ainvoke`'s outer HITL poll should use (C5).

    Core matches a pending approval on `(workflow_id, run_id, activity_id)`
    exactly (`GovernanceClient.poll_approval` / `ApprovalPollParams` — no
    other identifying field travels in that request), so polling the WRONG
    activity_id never resolves and `poll_until_decision`'s unbounded `while
    True` loop hangs forever.

    A REQUIRE_APPROVAL raised by the pure-LangChain-Core tool callback
    (installed under the C1 condition) carries the tool's REAL activity_id in
    `.identifier` — set by `LangGraphFrameworkAdapter._raise_pending_approval`
    from `current_activity_context()`, which resolves correctly here because
    `run_inline=True` means the callback raises INSIDE the ToolNode's
    `activity_scope(ctx, store=store)` (see `tool_activity_binding.py`). Use
    it verbatim so the poll targets the SAME row the tool's ActivityStarted
    opened — mirroring the pre-existing tool_start/tool_end HITL poll in
    `_process_event`, which has always polled the tool's own activity_id
    rather than a synthetic hook id.

    Falls back to the legacy synthetic `f"{run_id}-hook"` id when no
    identifier is carried (the base-hook — HTTP/DB/file/function preflight —
    REQUIRE_APPROVAL path this `except` block already handled before this
    phase; those raises carry no tool activity_id and are unaffected).
    """
    return hook_err.identifier or f"{run_id}-hook"


# ═══════════════════════════════════════════════════════════════════
# Run buffer (tracks in-flight runs for duration/context)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class _RunBuffer:
    run_id: str
    run_type: str
    name: str
    thread_id: str
    start_time_ms: float = field(default_factory=lambda: time.monotonic() * 1000)
    start_time_ns: int = field(default_factory=time.time_ns)
    langgraph_node: str | None = None
    langgraph_step: int | None = None
    subagent_name: str | None = None
    llm_started: bool = False  # True only when LLMStarted was actually sent to Core
    otel_span: Any = None       # OTel span for context propagation across asyncio.Task
    otel_token: Any = None      # OTel context detach token
    fallback_llm_activity_id: str | None = None
    """Fallback-path only (no bridge/callback owns this LLM call): the
    activity_id the consumer's OWN LLMCompleted close must use as its base
    (before appending ``-c``). Set by ``_map_event``'s ``on_chat_model_start``
    to the pre-screen's ``"{run_id}-pre"`` id when THIS call is the one that
    consumed ``_pre_screen_input``'s verdict (call 1) — mirrors the retired
    ``_GuardrailsCallbackHandler``'s ``llm_activity_map`` for the ONE path
    that still needs it (injected-client / subagent-gated handlers, where no
    callback ever claims LLM ownership). ``None`` for every other call, which
    falls back to ``event_run_id`` (this field's absence == pre-phase-5
    behavior unchanged)."""


class _RunBufferManager:
    def __init__(self) -> None:
        self._buffers: dict[str, _RunBuffer] = {}

    def register(
        self,
        run_id: str,
        run_type: str,
        name: str,
        thread_id: str,
        langgraph_node: str | None = None,
        langgraph_step: int | None = None,
        subagent_name: str | None = None,
    ) -> None:
        self._buffers[run_id] = _RunBuffer(
            run_id=run_id,
            run_type=run_type,
            name=name,
            thread_id=thread_id,
            langgraph_node=langgraph_node,
            langgraph_step=langgraph_step,
            subagent_name=subagent_name,
        )

    def get(self, run_id: str) -> _RunBuffer | None:
        return self._buffers.get(run_id)

    def remove(self, run_id: str) -> None:
        self._buffers.pop(run_id, None)

    def duration_ms(self, run_id: str) -> float | None:
        buf = self._buffers.get(run_id)
        if buf is None:
            return None
        return time.monotonic() * 1000 - buf.start_time_ms


# ═══════════════════════════════════════════════════════════════════
# Root run tracker (identifies the outermost graph invocation)
# ═══════════════════════════════════════════════════════════════════

class _RootRunTracker:
    def __init__(self) -> None:
        self._root_run_id: str | None = None

    def is_root(self, run_id: str) -> bool:
        """Return True and register run_id as root if no root exists yet."""
        if self._root_run_id is None:
            self._root_run_id = run_id
            return True
        return self._root_run_id == run_id

    @property
    def root_run_id(self) -> str | None:
        return self._root_run_id

    def reset(self) -> None:
        self._root_run_id = None


class _PreScreenClaim:
    """One-shot claim tracker for the FALLBACK (no bridge/callback) LLM path.

    Fallback-path only: when a callback owns LLM lifecycle (C1, bridge armed)
    the callback's own `bridge.prepare_llm(..., event_run_id=...)` alias (H11)
    already resolves which call gets the pre-screen id — this class is never
    consulted. Without a callback, the consumer's own `on_chat_model_start`
    (`_map_event`) is the only code that ever sees each LLM call, so IT must
    decide which one (call 1, and only call 1) claims the pre-screen's
    `"{run_id}-pre"` activity_id — mirroring the retired
    `_GuardrailsCallbackHandler`'s per-instance `_pre_screen_response`
    consume-once field, now scoped to a turn via this object instead of a
    callback instance.
    """

    def __init__(self, activity_id: str | None) -> None:
        self._activity_id = activity_id
        self._claimed = False

    def claim(self) -> str | None:
        """Return the pre-screen activity_id exactly once; `None` after (or
        if there was never a pre-screen response to begin with)."""
        if self._claimed or self._activity_id is None:
            return None
        self._claimed = True
        return self._activity_id


# ═══════════════════════════════════════════════════════════════════
# Options
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OpenBoxLangGraphHandlerOptions:
    """Configuration options for `OpenBoxLangGraphHandler`."""

    client: GovernanceClient | None = None
    on_api_error: str = "fail_open"
    api_timeout: int = 30_000
    send_chain_start_event: bool = True
    send_chain_end_event: bool = True
    send_tool_start_event: bool = True
    send_tool_end_event: bool = True
    send_llm_start_event: bool = True
    send_llm_end_event: bool = True
    skip_chain_types: set[str] = field(default_factory=set)
    skip_tool_types: set[str] = field(default_factory=set)
    hitl: Any = None  # HITLConfig | dict | None
    session_id: str | None = None
    multi_agent_session_id: str | None = None
    """Optional multi-agent session correlation id, kept separate from `session_id`.

    Threaded onto the base SDK's `ActivityContext.multi_agent_session_id` when
    `use_core_instrumentation`-style dual-write context binding is active;
    unused on the legacy-only path (no wire field carries it today)."""
    agent_name: str | None = None
    task_queue: str = "langgraph"
    use_native_interrupt: bool = False
    use_core_instrumentation: bool = True
    """Route hook governance through the shared ``openbox_core`` base
    instrumentation (``LangGraphFrameworkAdapter`` + InstrumentationManager) —
    the only hook runtime. Default ``True``. Setting it ``False`` fails fast
    (``OpenBoxConfigError``): legacy in-repo OTel hooks have been removed, so
    there is no fallback. Has no effect when `client` is injected — that path
    is lifecycle-only and builds no core runtime, so no hooks are armed."""
    root_node_names: set[str] = field(default_factory=set)
    resolve_subagent_name: Callable[[LangGraphStreamEvent], str | None] | None = None
    """Optional hook for framework-specific subagent name detection.

    Called on every `on_chain_start` / `on_tool_start` event.
    Return the subagent name if this event is a subagent invocation, else None.
    DeepAgents integration sets this to detect `task` tool sub-graphs.
    """
    sqlalchemy_engine: Any = None
    """Optional SQLAlchemy Engine instance to instrument for DB governance.
    Required when the engine is created before the handler (e.g. SQLDatabase.from_uri()).
    """
    tool_type_map: dict[str, str] | None = None
    """Optional mapping of tool_name → tool_type for execution tree classification.

    Supported values: "http", "database", "builtin", "a2a", "custom".
    If a tool is not listed and subagent_name is set, defaults to "a2a".
    Otherwise defaults to "custom".

    Example::

        tool_type_map={"search_web": "http", "query_db": "database"}
    """


# ═══════════════════════════════════════════════════════════════════
# OpenBoxLangGraphHandler
# ═══════════════════════════════════════════════════════════════════

class OpenBoxLangGraphHandler:
    """Wraps a compiled LangGraph graph and applies OpenBox governance to its event stream.

    Usage:
        governed = await create_openbox_graph_handler(
            graph=my_compiled_graph,
            api_url=os.environ["OPENBOX_URL"],
            api_key=os.environ["OPENBOX_API_KEY"],
            agent_name="MyAgent",
        )
        result = await governed.ainvoke(
            {"messages": [{"role": "user", "content": "Hello"}]},
            config={"configurable": {"thread_id": "session-abc"}},
        )
    """

    def __init__(
        self,
        graph: Any,
        options: OpenBoxLangGraphHandlerOptions | None = None,
    ) -> None:
        opts = options or OpenBoxLangGraphHandlerOptions()
        self._graph = graph
        self._resolve_subagent_name = opts.resolve_subagent_name

        # Build GovernanceConfig from options
        self._config = merge_config({
            "on_api_error": opts.on_api_error,
            "api_timeout": opts.api_timeout,
            "send_chain_start_event": opts.send_chain_start_event,
            "send_chain_end_event": opts.send_chain_end_event,
            "send_tool_start_event": opts.send_tool_start_event,
            "send_tool_end_event": opts.send_tool_end_event,
            "send_llm_start_event": opts.send_llm_start_event,
            "send_llm_end_event": opts.send_llm_end_event,
            "skip_chain_types": opts.skip_chain_types,
            "skip_tool_types": opts.skip_tool_types,
            "hitl": opts.hitl,
            "session_id": opts.session_id,
            "multi_agent_session_id": opts.multi_agent_session_id,
            "agent_name": opts.agent_name,
            "task_queue": opts.task_queue,
            "use_native_interrupt": opts.use_native_interrupt,
            "use_core_instrumentation": opts.use_core_instrumentation,
            "root_node_names": opts.root_node_names,
            "tool_type_map": opts.tool_type_map or {},
        })

        # Legacy in-repo hook governance (and its WorkflowSpanProcessor body
        # buffer) has been removed — base openbox_core instrumentation is the
        # only hook runtime. The attribute is retained (always None) so any
        # external subclass touching it degrades gracefully rather than
        # AttributeError-ing.
        self._span_processor = None

        # Ownership channel for the pure-LangChain-Core callback (C1). Created
        # ONLY under the exact same condition the callback is installed and the
        # ToolNode wrapper is told to prepare records — see the `else` branch
        # below. `None` here means "no callback, no bridge, consumer governs
        # every tool event unconditionally", matching today's behavior exactly.
        self._activity_bridge: ActivityBridge | None = None

        if opts.client:
            # Injected client (e.g. a test double, or a subclass overriding
            # evaluate_event) is used exactly as given — LIFECYCLE-ONLY: no
            # core runtime is built and NO hook instrumentation is armed. This
            # is the F2 seam: an injected client's OWN evaluate_event override
            # still intercepts every lifecycle governance call the handler
            # makes; hook-level (HTTP/DB/file/function) governance is simply
            # not active on this path. No activity bridge either — the
            # pure-LangChain-Core callback needs a real core runtime's
            # gate/adapter, which this path deliberately does not build.
            self._client = opts.client
            self._core_runtime = None
        else:
            gc = get_global_config()
            # Own core runtime, own private ContextStore (create_core_runtime's
            # isolation guarantee) — built from the SAME resolved
            # api_url/api_key/timeout/on_api_error/agent_did/agent_private_key
            # the GovernanceClient below is constructed from, so the gate
            # evaluates against the identical Core endpoint/identity. Base
            # instrumentation (the only hook runtime) is installed inside
            # create_core_runtime; it raises OpenBoxConfigError when
            # use_core_instrumentation=False (no legacy fallback exists).
            self._core_runtime = create_core_runtime(
                self._config,
                api_url=gc.api_url,
                api_key=gc.api_key,
                governance_timeout=gc.governance_timeout,
                agent_did=gc.agent_did,
                agent_private_key=gc.agent_private_key,
                extra_ignored_urls={gc.api_url} if gc.api_url else None,
            )
            self._client = GovernanceClient(
                api_url=gc.api_url,
                api_key=gc.api_key,
                timeout=gc.governance_timeout,  # seconds
                on_api_error=self._config.on_api_error,
                agent_did=gc.agent_did,
                agent_private_key=gc.agent_private_key,
                gate=self._core_runtime.gate,
            )
            # C1 — install condition == prepare condition: the pure-LangChain-Core
            # callback (installed per-turn in `_governed_config`) is only ever
            # armed for a handler with its OWN core runtime AND no subagent-name
            # resolver. A subagent-gated handler (DeepAgents-style `task` tool
            # sub-graphs) stays consumer-governed exactly as before — arming a
            # bridge here without a matching callback installed would have the
            # wrapper `prepare_tool` records the consumer then treats as "sent"
            # they never are (the C1 blackout this phase fixes). Only build the
            # bridge under this SAME condition; otherwise leave it `None` from
            # `__init__`'s top so `bridge=None` reaches `bind_tools_activity_scope`
            # and the wrapper never prepares a record.
            if self._resolve_subagent_name is None:
                self._activity_bridge = ActivityBridge()

            # Bind the base-SDK ActivityContext around ACTUAL tool execution at
            # the ToolNode request seam: mint a canonical activity id, write it
            # to the tool's config["run_id"] so on_tool_start (thus
            # ToolStarted.activity_id) carries it, and run execute() inside
            # activity_scope on this runtime's private store. Hooks fired inside
            # a tool then resolve to that exact activity via the ContextVar tier
            # (the primary mechanism), instead of the trace-lookup fallback.
            # Best-effort + idempotent; injected-client (lifecycle-only)
            # handlers skip this — they own no store.
            if graph is not None:
                bind_tools_activity_scope(
                    graph,
                    core_runtime=self._core_runtime,
                    config=self._config,
                    resolve_tool_type=lambda name: self._resolve_tool_type(name, None),
                    bridge=self._activity_bridge,
                )

    # ─────────────────────────────────────────────────────────────
    # Pre-screen: enforce guardrails before stream starts
    # ─────────────────────────────────────────────────────────────

    async def _pre_screen_input(
        self,
        input: dict[str, Any],
        workflow_id: str,
        run_id: str,
        graph_input: dict[str, Any] | None = None,
    ) -> tuple[bool, GovernanceVerdictResponse | None]:
        """Send WorkflowStarted + LLMStarted governance events before the stream starts.

        Returns (workflow_started_sent, pre_screen_response):
        - workflow_started_sent: True if WorkflowStarted was sent (suppress duplicate
          from on_chain_start in _process_event).
        - pre_screen_response: the LLMStarted verdict response, passed to the callback
          handler so on_chat_model_start can reuse it for PII redaction without sending
          a second ActivityStarted event.

        Unlike the callback handler (which LangGraph's runner silently swallows),
        exceptions raised here propagate directly to the ainvoke/astream_governed
        caller — so GuardrailsValidationError, GovernanceHaltError, GovernanceBlockedError
        all reach the user's except block and halt the session correctly.
        """
        # ── 0. SignalReceived — fire before WorkflowStarted so the dashboard shows
        # the user prompt as the trigger that initiated the session.
        # Extract the last human message from the input as the signal payload.
        _sig_messages = input.get("messages") or []
        _user_prompt: str | None = None
        for _msg in reversed(_sig_messages):
            if isinstance(_msg, dict):
                if _msg.get("role") in ("user", "human"):
                    _user_prompt = _msg.get("content") or None
                    break
            elif hasattr(_msg, "type") and _msg.type in ("human", "generic"):
                _c = _msg.content
                _user_prompt = _c if isinstance(_c, str) else None
                break
        if _user_prompt:
            sig_event = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type="SignalReceived",
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                activity_id=f"{run_id}-sig",
                activity_type="user_prompt",
                signal_name="user_prompt",
                signal_args=[_user_prompt],
            )
            await self._client.evaluate_event(sig_event)

        # ── 1. WorkflowStarted — must precede any ActivityStarted so the dashboard
        # creates a session to attach the guardrail event to (mirrors both SDKs).
        # Gated on send_chain_start_event only, NOT on send_llm_start_event.
        if self._config.send_chain_start_event:
            wf_start = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type="WorkflowStarted",
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                activity_id=f"{run_id}-wf",
                activity_type=self._config.agent_name or "LangGraphRun",
                activity_input=[safe_serialize(input)],
            )
            await self._client.evaluate_event(wf_start)
            workflow_started_sent = True
        else:
            workflow_started_sent = False

        # ── 2. LLMStarted pre-screen — enforce guardrails on the user prompt
        if not self._config.send_llm_start_event:
            return workflow_started_sent, None

        messages = input.get("messages") or []
        prompt_parts: list[str] = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role in ("user", "human") and isinstance(content, str):
                    prompt_parts.append(content)
            elif hasattr(msg, "type") and msg.type in ("human", "generic"):
                c = msg.content
                if isinstance(c, str):
                    prompt_parts.append(c)
        if not prompt_parts:
            return workflow_started_sent, None
        prompt_text = "\n".join(prompt_parts)

        gov = LangChainGovernanceEvent(
            source="workflow-telemetry",
            event_type="LLMStarted",
            workflow_id=workflow_id,
            run_id=run_id,
            workflow_type=self._config.agent_name or "LangGraphRun",
            task_queue=self._config.task_queue,
            timestamp=rfc3339_now(),
            session_id=self._config.session_id,
            activity_id=f"{run_id}-pre",
            activity_type="llm_call",
            activity_input=[{"prompt": prompt_text}],
            prompt=prompt_text,
        )

        response = await self._client.evaluate_event(gov)
        if response is None:
            return workflow_started_sent, None

        # Enforce — exceptions propagate directly to the caller here.
        # If blocked/halted, close the WorkflowStarted session first so the
        # dashboard doesn't show an orphaned open session.
        enforcement_error: Exception | None = None
        try:
            result = enforce_verdict(response, "llm_start")
        except Exception as exc:
            enforcement_error = exc
            result = None  # type: ignore[assignment]

        if (
            enforcement_error is not None
            and workflow_started_sent
            and self._config.send_chain_end_event
        ):
            wf_end = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type="WorkflowCompleted",
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                activity_id=f"{run_id}-wf",
                activity_type=self._config.agent_name or "LangGraphRun",
                status="failed",
                error=str(enforcement_error),
            )
            await self._client.evaluate_event(wf_end)
            raise enforcement_error

        if result and result.requires_hitl:
            try:
                await poll_until_decision(
                    self._client,
                    HITLPollParams(
                        workflow_id=workflow_id,
                        run_id=run_id,
                        activity_id=f"{run_id}-pre",
                        activity_type="llm_call",
                    ),
                    self._config.hitl,
                )
            except (ApprovalRejectedError, ApprovalExpiredError, ApprovalTimeoutError) as e:
                raise GovernanceHaltError(str(e)) from e

        return workflow_started_sent, response

    def _cleanup_turn(self, workflow_id: str) -> None:
        """Drop this turn's core-runtime dual-write bindings + abort marks.

        No-op when the handler has no core runtime (`_core_runtime is None` —
        injected-client handlers, legacy-only). Every public entry point calls
        this from the OUTERMOST `finally` of its stream loop so it runs
        exactly once per turn regardless of success, mid-stream exception, or
        (for `ainvoke`) an approved hook-approval retry — see the `finally`
        placement in each entry point for why ordering after the retry matters.
        Stays SYNCHRONOUS (existing tests spy/patch it with a plain callable
        called without `await`) — the C6 orphan-close below uses the base
        SDK's SYNC gate for the same reason.

        Only sweeps THIS turn's `workflow_id` — never
        `self._core_runtime.context_store.clear()`, which would also drop
        another concurrent turn's bindings on the same handler (multiple
        `ainvoke`/`astream*` calls can be in flight together) and the
        runtime's `halt_requested` flag, neither of which is this turn's to
        clear.

        C6/M19 — when this turn armed an `ActivityBridge`, also sweep it:
        ``sweep_workflow`` drops every bridge record for this workflow and
        returns them so any ``tool_started_sent and not tool_completed_sent``
        row (a sibling `asyncio.gather` cancellation leaves no `on_tool_end`/
        `on_tool_error` to close it — `CancelledError` is a `BaseException`,
        not caught by the callback's own body) gets a synthetic failed close
        via the runtime's sync gate. Every record this bridge ever prepared
        may ALSO carry an abort-mark the callback set via
        `current_activity_context()` inside the ToolNode-seam `activity_scope`
        (a ContextVar bind that never goes through
        `register_activity`/`TraceContextRegistry`, so the trace-only `sweep`
        above never sees that key) — clear those directly on the runtime's
        store so they cannot leak for the handler's lifetime.

        The abort-mark clear runs UNCONDITIONALLY for every swept tool record
        (not gated on `record.abort_marked`): a REQUIRE_APPROVAL raised by the
        callback can propagate through ANY entry point (`ainvoke` polls and
        retries it; `astream_governed`/`astream`/`astream_events` have no
        catch/poll loop at all — pre-existing, HITL retry is `ainvoke`-only —
        and simply propagate it to the caller), so a per-entry-point catch
        site is not a reliable place to set the flag. `clear_activity_aborted`
        is an idempotent set-discard — a no-op for a record that was never
        actually aborted — so clearing unconditionally is always safe.
        `record.abort_marked` is still set (see `_mark_bridge_abort`) and
        checked here as a fast-path/diagnostic signal, not a gate.
        """
        if self._core_runtime is None:
            return
        get_trace_registry(self._core_runtime).sweep(workflow_id)
        if self._activity_bridge is None:
            return
        store = self._core_runtime.context_store
        for record in self._activity_bridge.sweep_workflow(workflow_id):
            store.clear_activity_aborted(workflow_id, record.activity_id)
            if record.tool_started_sent and not record.tool_completed_sent:
                _logger.info(
                    "[OpenBox] sweeping orphan callback-started tool activity "
                    "%s (started, never completed — sibling cancellation?)",
                    record.activity_id,
                )
                self._close_orphan_bridge_tool(workflow_id, record)

    def _mark_bridge_abort(self, workflow_id: str, activity_id: str) -> None:
        """Record (M19) that the base store's abort mark for `activity_id` was
        set via the ToolNode-seam ContextVar path, NOT `register_activity` —
        so `_cleanup_turn`'s `TraceContextRegistry.sweep` (which only clears
        keys it registered) will never see it, and the mark would otherwise
        leak on the runtime's `ContextStore` for the handler's lifetime.

        No-op when this turn has no bridge (consumer-governed path — the
        adapter's OWN abort-mark clearing there is exactly what `sweep`
        already covers, via `register_activity`'s trace-only dual-write).
        Safe to call with an activity_id the bridge never prepared (e.g. the
        legacy synthetic `f"{run_id}-hook"` id from a base-hook approval,
        C5's fallback branch) — `ActivityBridge.get` returns None and this is
        a no-op, exactly matching pre-phase-4 behavior for that path.
        """
        if self._activity_bridge is None:
            return
        record = self._activity_bridge.get(workflow_id, activity_id)
        if record is not None:
            record.abort_marked = True

    def _close_orphan_bridge_tool(self, workflow_id: str, record: Any) -> None:
        """Best-effort failed ActivityCompleted for a swept orphan tool row.

        Uses the runtime's SYNC gate directly (never the async
        `GovernanceClient`) so `_cleanup_turn` can stay fully synchronous.
        Strictly telemetry (closing an already-abandoned row) — failures are
        logged, never raised, so a governance API hiccup during turn cleanup
        never masks whatever exception (if any) is already propagating
        through the `finally` this is called from.
        """
        from openbox_core.contracts.events import activity_completed

        try:
            envelope = activity_completed(
                workflow_id=workflow_id,
                run_id=workflow_id,
                workflow_type=self._config.agent_name or "LangGraphRun",
                activity_id=record.activity_id,
                activity_type=record.tool_name or "tool",
                task_queue=self._config.task_queue,
                error="Activity abandoned (turn cleanup swept an unclosed row)",
            )
            self._core_runtime.gate.evaluate(envelope)  # type: ignore[union-attr]
        except Exception:
            _logger.warning(
                "[OpenBox] failed to close orphan bridge tool activity %s",
                record.activity_id,
                exc_info=True,
            )

    def _reset_after_approval(self, workflow_id: str) -> None:
        """Clear the abort mark(s) a hook set for this turn BEFORE an approved
        REQUIRE_APPROVAL retry re-invokes the graph, so the retry runs
        GOVERNED instead of short-circuiting on the stale abort flag the
        blocked first pass left behind.

        No-op when the handler has no core runtime (`_core_runtime is None`)
        OR the runtime's adapter is the base default `CoreAdapter` (only
        reachable when `use_core_instrumentation=False` — that adapter has no
        `reset_after_approval`, matching this turn never having armed base
        instrumentation in the first place, so there is nothing to reset).
        Call BEFORE re-invoking the graph, AFTER `poll_until_decision`
        resolves — matches `LangGraphFrameworkAdapter.reset_after_approval`'s
        own ordering contract.
        """
        if self._core_runtime is None:
            return
        reset = getattr(self._core_runtime.adapter, "reset_after_approval", None)
        if reset is not None:
            reset(workflow_id)

    def _governed_config(
        self,
        config: dict[str, Any] | None,
        *,
        workflow_id: str,
        run_id: str,
        thread_id: str,
        pre_screen_response: GovernanceVerdictResponse | None,
    ) -> dict[str, Any]:
        """Build the RunnableConfig with this turn's core callbacks.

        Note: tool-execution ``ActivityContext`` binding is NOT done via a
        callback — LangChain isolates callback context from the tool body, so a
        callback bind never reaches the hooks. It is done at the ToolNode request
        seam (see ``tool_activity_binding``); this method threads the per-turn
        ids down to those wrappers via ``config["metadata"]`` (a per-invocation
        channel, concurrency-safe — unlike a shared module ContextVar).

        When ``self._activity_bridge`` is armed (C1 — same condition as its
        construction in ``__init__``), BOTH the async and sync pure-LangChain-Core
        callbacks (Phase 2, ``openbox_langchain``) are appended to
        ``cfg["callbacks"]``. The sync handler is what makes the sync-only-tool
        corner fail-closed PRE-body (C2 — the async handler alone is swallowed
        there by ``BaseTool.run``'s sync callback manager); installing both lets
        LangGraph's own cross-dispatch (Phase 0-measured) exercise the
        evaluate-once/enforce-from-stash contract on every tool call, not just
        that corner. ``record_less_ok=False`` on both (C8) — this handler's
        bridge always prepares a record before a governed tool runs, so a
        record-less callback fire only happens on an unbound nested-subgraph
        ToolNode, which must NOT send (the consumer, which DOES see that inner
        event, remains the sole sender).

        Phase 5 — the SAME two callback instances now also own the LLM
        lifecycle (``send_llm_start_event``/``send_llm_end_event=True``):
        pre-screen reuse for call 1 (``pre_screen_response`` mapped to the
        base SDK's ``EvaluationResult``, M18), redaction, trace registration,
        and the same-id LLMCompleted close (H11). This retires
        ``_GuardrailsCallbackHandler``'s LLM duties in one atomic cutover — a
        handler with NO bridge (injected client, or a subagent-gated handler
        per the C1 blackout above) installs NO LLM-owning callback either,
        exactly mirroring the pre-existing tool-ownership fallback: the
        consumer's own ``_pre_screen_input`` (enforcement) and
        ``_map_event``'s ``on_chat_model_start``/``on_chat_model_end``
        (telemetry) already cover that path unconditionally and are
        untouched by this phase — see ``_process_event``'s LLMCompleted
        fallback branch.
        """
        callbacks: list[Any] = []
        if self._activity_bridge is not None and self._core_runtime is not None:
            registry = get_trace_registry(self._core_runtime)
            callback_options = OpenBoxLangChainCoreCallbackOptions(
                runtime=self._core_runtime,
                bridge=self._activity_bridge,
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or "LangGraphRun",
                task_queue=self._config.task_queue,
                session_id=self._config.session_id,
                agent_name=self._config.agent_name,
                send_tool_start_event=self._config.send_tool_start_event,
                send_tool_end_event=self._config.send_tool_end_event,
                send_llm_start_event=self._config.send_llm_start_event,
                send_llm_end_event=self._config.send_llm_end_event,
                # P3: a bound lambda, not `self._resolve_tool_type(name, None)` —
                # `name` is out of scope here; the resolver receives only the
                # tool name argument the callback passes it.
                tool_type_resolver=lambda n: self._resolve_tool_type(n, None),
                pre_screen_response=(
                    pre_screen_response.to_evaluation_result()
                    if pre_screen_response is not None
                    else None
                ),
                pre_screen_activity_id=(
                    f"{run_id}-pre" if pre_screen_response is not None else None
                ),
                # Registry-backed, not the base SDK's process-wide default
                # (M21) — the SAME workflow-scoped `TraceContextRegistry` this
                # handler's tool path and `_cleanup_turn`'s sweep already use,
                # so a turn-exit sweep also drops LLM trace bindings and an
                # exact-trace hook lookup resolves via the SAME tier the tool
                # path relies on. Bound methods, not the raw registry, to
                # match the `Callable[[int|str, ActivityContext], None]` /
                # `Callable[[int|str], None]` option shapes exactly.
                register_trace=registry.register,
                unregister_trace=registry.unregister,
                record_less_ok=False,
            )
            callbacks.append(OpenBoxLangChainCoreAsyncCallbackHandler(callback_options))
            callbacks.append(OpenBoxLangChainCoreSyncCallbackHandler(callback_options))
        cfg = dict(config or {})
        cfg["callbacks"] = [*list(cfg.get("callbacks") or []), *callbacks]
        # Carry this turn's ids to the wrapped tools via their own RunnableConfig
        # (LangGraph propagates metadata down to each tool). Only when a core
        # runtime owns a store to bind on; merged so user metadata survives.
        if self._core_runtime is not None:
            cfg["metadata"] = {**(cfg.get("metadata") or {}), **turn_metadata(workflow_id, run_id)}
        return cfg

    # ─────────────────────────────────────────────────────────────
    # Public invoke / ainvoke
    # ─────────────────────────────────────────────────────────────

    async def ainvoke(
        self,
        input: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Invoke the governed graph and return the final state.

        Streams events via `astream_events` (governance applied inline) and
        returns the final graph output from the root `on_chain_end` event.
        Does NOT call `ainvoke` on the underlying graph a second time.

        Args:
            input: The initial graph state (e.g. `{"messages": [...]}`)
            config: LangGraph RunnableConfig — must include
                `{"configurable": {"thread_id": "..."}}` for session tracking.
        """
        thread_id = _extract_thread_id(config)
        # Generate fresh workflow_id + run_id per turn, matching Temporal SDK:
        #   workflow_id = stable logical session ID (unique per-turn)
        #   run_id      = unique execution attempt ID (distinct from workflow_id)
        # Core seals a workflow after WorkflowCompleted — reusing the same
        # workflow_id causes HALT: "fully attested and sealed".
        _turn = uuid.uuid4().hex
        workflow_id = f"{thread_id}-{_turn[:8]}"
        run_id = f"{thread_id}-run-{_turn[8:16]}"
        root_tracker = _RootRunTracker()
        buffer = _RunBufferManager()
        final_output: dict[str, Any] = {}

        # Pre-screen: enforce guardrails BEFORE stream starts so exceptions
        # propagate to the caller (LangGraph runner swallows callback exceptions).
        # Returns (workflow_started_sent, pre_screen_response) — response reused
        # by the shared core callback for PII redaction (call 1) to avoid a
        # duplicate ActivityStarted.
        workflow_started_sent, pre_screen_response = await self._pre_screen_input(
            input, workflow_id, run_id
        )
        pre_screen_claim = _PreScreenClaim(
            f"{run_id}-pre" if pre_screen_response is not None else None
        )

        cfg = self._governed_config(
            config,
            workflow_id=workflow_id,
            run_id=run_id,
            thread_id=thread_id,
            pre_screen_response=pre_screen_response,
        )

        try:
            async for event in self._graph.astream_events(
                input, config=cfg, version="v2", **kwargs
            ):
                stream_event = LangGraphStreamEvent.from_dict(event)
                await self._process_event(
                    stream_event, thread_id, workflow_id, run_id, root_tracker, buffer,
                    workflow_started_sent=workflow_started_sent,
                    pre_screen_claim=pre_screen_claim,
                )
                # Capture the root graph's final output from on_chain_end
                if (
                    stream_event.event == "on_chain_end"
                    and root_tracker.root_run_id == stream_event.run_id
                ):
                    output = stream_event.data.get("output")
                    if isinstance(output, dict):
                        final_output = output
        except GovernanceBlockedError as hook_err:
            if hook_err.verdict != "require_approval":
                raise
            _logger.info("[OpenBox] Hook REQUIRE_APPROVAL during ainvoke, polling")
            poll_activity_id = _approval_poll_activity_id(hook_err, run_id)
            self._mark_bridge_abort(workflow_id, poll_activity_id)
            await poll_until_decision(
                self._client,
                HITLPollParams(
                    workflow_id=workflow_id,
                    run_id=run_id,
                    activity_id=poll_activity_id,
                    activity_type="hook",
                ),
                self._config.hitl,
            )
            _logger.info("[OpenBox] Approval granted, retrying ainvoke")
            self._reset_after_approval(workflow_id)
            final_output = await self._graph.ainvoke(input, config=cfg, **kwargs)
        except Exception as exc:
            hook_err = _extract_governance_blocked(exc)
            if hook_err is None or hook_err.verdict != "require_approval":
                raise
            _logger.info("[OpenBox] Hook REQUIRE_APPROVAL (wrapped) during ainvoke, polling")
            poll_activity_id = _approval_poll_activity_id(hook_err, run_id)
            self._mark_bridge_abort(workflow_id, poll_activity_id)
            await poll_until_decision(
                self._client,
                HITLPollParams(
                    workflow_id=workflow_id,
                    run_id=run_id,
                    activity_id=poll_activity_id,
                    activity_type="hook",
                ),
                self._config.hitl,
            )
            _logger.info("[OpenBox] Approval granted, retrying ainvoke")
            self._reset_after_approval(workflow_id)
            final_output = await self._graph.ainvoke(input, config=cfg, **kwargs)
        finally:
            # Outermost `finally` on purpose: an approval retry re-runs the
            # graph INSIDE the `except` blocks above, so this only fires once
            # the (possibly retried) turn is fully done — the retry keeps its
            # dual-write context intact instead of racing a cleanup that
            # unregisters it mid-retry. See `_cleanup_turn`.
            self._cleanup_turn(workflow_id)

        return final_output

    async def astream_governed(
        self,
        input: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream governed graph updates, yielding each update chunk.

        Governance is applied inline as events are streamed. The caller
        receives graph state update chunks identically to `astream_events`.

        Args:
            input: The initial graph state.
            config: LangGraph RunnableConfig with `thread_id`.
        """
        thread_id = _extract_thread_id(config)
        _turn = uuid.uuid4().hex
        workflow_id = f"{thread_id}-{_turn[:8]}"
        run_id = f"{thread_id}-run-{_turn[8:16]}"
        root_tracker = _RootRunTracker()
        buffer = _RunBufferManager()

        workflow_started_sent, pre_screen_response = await self._pre_screen_input(
            input, workflow_id, run_id
        )
        pre_screen_claim = _PreScreenClaim(
            f"{run_id}-pre" if pre_screen_response is not None else None
        )

        cfg = self._governed_config(
            config,
            workflow_id=workflow_id,
            run_id=run_id,
            thread_id=thread_id,
            pre_screen_response=pre_screen_response,
        )

        _debug = os.environ.get("OPENBOX_DEBUG") == "1"
        try:
            async for event in self._graph.astream_events(
                input, config=cfg, version="v2", **kwargs
            ):
                stream_event = LangGraphStreamEvent.from_dict(event)
                if _debug and "_stream" not in stream_event.event:
                    sys.stderr.write(
                        f"[OBX_EVENT] {stream_event.event:<25} name={stream_event.name!r:<35} "
                        f"node={stream_event.metadata.get('langgraph_node')!r}\n"
                    )
                await self._process_event(
                    stream_event, thread_id, workflow_id, run_id, root_tracker, buffer,
                    workflow_started_sent=workflow_started_sent,
                    pre_screen_claim=pre_screen_claim,
                )
                yield event
        finally:
            # Outermost `finally` on an async generator fires on normal
            # exhaustion, an exception raised through the loop, OR the
            # caller closing/abandoning this generator (`aclose()` / GC) —
            # covering the mid-stream-raise and early-break cases the plain
            # `try/except` in `ainvoke` doesn't need to (it has no yield).
            self._cleanup_turn(workflow_id)

    async def astream(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Graph-compatible astream — delegates to astream_governed.

        Provided so `langgraph dev` and other LangGraph tooling that calls
        ``graph.astream(...)`` can use this handler as a drop-in replacement
        for a ``CompiledStateGraph``.
        """
        # This method mints no turn of its own (astream_governed does) — the
        # `contextlib.aclosing` here exists ONLY so an abandoned/early-broken
        # `astream` generator still closes the inner `astream_governed`
        # generator it delegates to. Without it, `GeneratorExit` thrown into
        # THIS generator's suspended `yield chunk` below never reaches the
        # `async for` over `inner`, so `astream_governed`'s own turn cleanup
        # would never run — verified empirically: `GeneratorExit` on an outer
        # generator's `aclose()` does NOT propagate into an inner generator
        # merely being iterated via `async for`, only into an explicit
        # `aclose()` call on that inner generator from a `finally` here.
        # `cast` is safe: `astream_governed` has a `yield` in its body, so its
        # runtime type is always an async generator — `AsyncIterator` is only
        # its PUBLIC return annotation (matching `astream`'s own, for a
        # drop-in `CompiledStateGraph` surface), not what `aclosing` needs.
        governed = cast(
            "AsyncGenerator[dict[str, Any], None]",
            self.astream_governed(input, config=config, **kwargs),
        )
        async with contextlib.aclosing(governed) as inner:
            async for chunk in inner:
                yield chunk

    async def astream_events(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
        *,
        version: str = "v2",
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Graph-compatible astream_events — runs governance and re-yields raw events.

        Provided so tooling that calls ``graph.astream_events(...)`` works
        transparently with the governed handler.
        """
        thread_id = _extract_thread_id(config)
        _turn = uuid.uuid4().hex
        workflow_id = f"{thread_id}-{_turn[:8]}"
        run_id = f"{thread_id}-run-{_turn[8:16]}"
        root_tracker = _RootRunTracker()
        buffer = _RunBufferManager()

        workflow_started_sent, pre_screen_response = await self._pre_screen_input(
            input, workflow_id, run_id
        )
        pre_screen_claim = _PreScreenClaim(
            f"{run_id}-pre" if pre_screen_response is not None else None
        )

        cfg = self._governed_config(
            config,
            workflow_id=workflow_id,
            run_id=run_id,
            thread_id=thread_id,
            pre_screen_response=pre_screen_response,
        )

        try:
            async for event in self._graph.astream_events(
                input, config=cfg, version=version, **kwargs
            ):
                stream_event = LangGraphStreamEvent.from_dict(event)
                await self._process_event(
                    stream_event, thread_id, workflow_id, run_id, root_tracker, buffer,
                    pre_screen_claim=pre_screen_claim,
                    workflow_started_sent=workflow_started_sent,
                )
                yield event
        finally:
            # See astream_governed's identical finally for why this covers
            # normal exhaustion, mid-stream exceptions, AND an abandoned
            # generator (aclose()/GC) alike.
            self._cleanup_turn(workflow_id)

    # ─────────────────────────────────────────────────────────────
    # Event processing
    # ─────────────────────────────────────────────────────────────

    async def _process_event(
        self,
        event: LangGraphStreamEvent,
        thread_id: str,
        workflow_id: str,
        run_id: str,
        root_tracker: _RootRunTracker,
        buffer: _RunBufferManager,
        *,
        workflow_started_sent: bool = False,
        pre_screen_claim: _PreScreenClaim | None = None,
    ) -> None:
        """Process a single LangGraph stream event through governance."""
        # ── C1/C7 — callback ownership, checked BEFORE `_map_event` runs its
        # side effects (trace registration, span creation). `event.run_id` IS
        # the canonical activity id for tool events (the ToolNode-seam wrapper
        # writes it into `config["run_id"]` before `execute()` mints the
        # LangChain run — see `tool_activity_binding.py`), so it is also
        # exactly the key the bridge/callback used. Ownership is checked on
        # SENT flags only (`is_callback_owned`), never record-existence — a
        # prepared-but-never-started record (e.g. the callback wasn't
        # installed for this turn, C1 blackout) is NOT owned and falls through
        # to full consumer governance below, unchanged from pre-phase-4
        # behavior.
        #
        # LLM events (Phase 5) resolve their bridge key via the H11
        # `event_run_id` alias FIRST: the callback's `on_chat_model_start`
        # calls `bridge.prepare_llm(..., event_run_id=...)` unconditionally,
        # so `get_by_event_run_id` finds the record even when the first
        # call's activity_id diverges from `event.run_id` (the pre-screen
        # `"{run_id}-pre"` row) — falling through to a direct `event.run_id`
        # lookup when no alias was ever registered (the callback never ran,
        # C1 blackout).
        tool_owned_start = False
        tool_owned_complete = False
        llm_owned_start = False
        llm_owned_complete = False
        bridge = self._activity_bridge
        if bridge is not None and event.event in ("on_tool_start", "on_tool_end"):
            event_type: EventType = (
                "tool_start" if event.event == "on_tool_start" else "tool_complete"
            )
            if bridge.is_callback_owned(workflow_id, event.run_id, event_type):
                if event_type == "tool_start":
                    tool_owned_start = True
                else:
                    tool_owned_complete = True
        llm_activity_id = event.run_id
        if bridge is not None and event.event in ("on_chat_model_start", "on_chat_model_end"):
            llm_record = bridge.get_by_event_run_id(workflow_id, event.run_id)
            if llm_record is not None:
                llm_activity_id = llm_record.activity_id
            llm_event_type: EventType = (
                "llm_start" if event.event == "on_chat_model_start" else "llm_complete"
            )
            if bridge.is_callback_owned(workflow_id, llm_activity_id, llm_event_type):
                if llm_event_type == "llm_start":
                    llm_owned_start = True
                else:
                    llm_owned_complete = True

        gov_event, is_root, is_start, event_type_label = self._map_event(
            event, thread_id, workflow_id, run_id, root_tracker, buffer,
            skip_consumer_side_effects=(
                tool_owned_start or tool_owned_complete or llm_owned_start or llm_owned_complete
            ),
            pre_screen_claim=pre_screen_claim,
        )

        # ── Callback-owned ToolCompleted (P1): the callback already SENT
        # ActivityCompleted (telemetry-only, C4 — gate.aevaluate, never
        # adapter-enforcing) and stashed the verdict on the bridge record.
        # Read it back and drive the SAME enforce + poll-and-continue the
        # consumer runs today for an unowned ToolCompleted, so a callback-owned
        # BLOCK/HALT/REQUIRE_APPROVAL is never silently downgraded to
        # telemetry-only. Never re-SEND (the callback already did).
        if tool_owned_complete:
            record = bridge.get(workflow_id, event.run_id) if bridge is not None else None
            stashed = record.completion_result if record is not None else None
            if stashed is not None:
                # Named distinctly from `response` below — mypy infers a
                # single type for a repeated variable name across a whole
                # function body, and `response` elsewhere in this method is
                # `GovernanceVerdictResponse | None` (evaluate_event's return
                # type), while `.from_result` always returns non-Optional.
                stashed_response = GovernanceVerdictResponse.from_result(stashed)
                context = lang_graph_event_to_context(event.event, is_root=is_root)
                result = enforce_verdict(stashed_response, context)
                if result.requires_hitl:
                    try:
                        await poll_until_decision(
                            self._client,
                            HITLPollParams(
                                workflow_id=workflow_id,
                                run_id=run_id,
                                activity_id=event.run_id,
                                activity_type=(gov_event.activity_type if gov_event else None)
                                or event.name,
                            ),
                            self._config.hitl,
                        )
                    except (ApprovalRejectedError, ApprovalExpiredError, ApprovalTimeoutError) as e:
                        raise GovernanceHaltError(str(e)) from e
            return

        # Callback-owned ToolStarted: the callback already sent+enforced this
        # (a stop-shaped/REQUIRE_APPROVAL verdict would have raised out of the
        # callback itself, before this event was ever produced) — nothing left
        # for the consumer to send or enforce.
        if tool_owned_start:
            return

        # ── Callback-owned LLMCompleted (Phase 5, mirrors tool_owned_complete
        # above): the callback already sent ActivityCompleted (gate.aevaluate
        # only, C4 — never adapter-enforcing) on the SAME id as its
        # LLMStarted, and stashed the verdict on the bridge record keyed by
        # THAT id (`llm_activity_id`, resolved via the H11 alias above — NOT
        # `event.run_id` for a pre-screened first call). Checked BEFORE the
        # `gov_event is None` guard below: the consumer's OWN `_map_event`
        # empty-prompt skip (M14) returns `None` independent of callback
        # ownership, but the callback (M14-documented: sends even an empty
        # prompt) still governed this call and may have stashed a
        # REQUIRE_APPROVAL/BLOCK/HALT verdict that must not be silently
        # dropped just because the consumer's own mapping had nothing to add.
        # Enforce from the stash so a BLOCK/HALT/REQUIRE_APPROVAL is never
        # silently downgraded to telemetry-only; never re-SEND.
        if llm_owned_complete:
            record = bridge.get(workflow_id, llm_activity_id) if bridge is not None else None
            stashed = record.completion_result if record is not None else None
            if stashed is not None:
                stashed_response = GovernanceVerdictResponse.from_result(stashed)
                context = lang_graph_event_to_context(event.event, is_root=is_root)
                result = enforce_verdict(stashed_response, context)
                if result.requires_hitl:
                    try:
                        await poll_until_decision(
                            self._client,
                            HITLPollParams(
                                workflow_id=workflow_id,
                                run_id=run_id,
                                activity_id=llm_activity_id,
                                activity_type=(gov_event.activity_type if gov_event else None)
                                or "llm_call",
                            ),
                            self._config.hitl,
                        )
                    except (ApprovalRejectedError, ApprovalExpiredError, ApprovalTimeoutError) as e:
                        raise GovernanceHaltError(str(e)) from e
            return

        # Callback-owned LLMStarted: the callback already sent this (pre-screen
        # reuse or a real evaluate) — nothing left for the consumer to send.
        # Same "checked before gov_event is None" rationale as above.
        if llm_owned_start:
            return

        if gov_event is None:
            return

        # ── Skip events disabled in config
        if is_start:
            if event_type_label == "ChainStarted" and not self._config.send_chain_start_event:
                return
            # _pre_screen_input already sent WorkflowStarted — skip duplicate from on_chain_start
            if event_type_label == "ChainStarted" and is_root and workflow_started_sent:
                return
            if event_type_label == "ToolStarted" and not self._config.send_tool_start_event:
                return
            if event_type_label == "LLMStarted" and not self._config.send_llm_start_event:
                return
            # ── LLMStarted FALLBACK (Phase 5, M13 non-goal preserved): reached
            # only when no callback owns this LLM call's start (injected
            # client, subagent-gated handler, callback disabled). Mirrors the
            # retired `_GuardrailsCallbackHandler.on_chat_model_start` EXACTLY:
            # call 1 (this run claimed the pre-screen's `-pre` activity_id via
            # `_PreScreenClaim` — checked on the buffer, NOT `gov_event
            # .activity_id`, which is always the raw `event_run_id` for a
            # START event regardless of the claim) was ALREADY sent and
            # enforced by `_pre_screen_input` before the stream started — skip
            # it here to avoid a duplicate ActivityStarted. Call 2+ (no claim)
            # is SENT here as telemetry (a real `evaluate_event` call, so Core
            # still gets a row for it) but deliberately NEVER enforced
            # (`enforce_verdict`/HITL are skipped) — 2nd+ prompts stay
            # UNENFORCED by design (M13, documented Non-Goal, not closed by
            # this phase; the retired class had this exact comment: "LangGraph's
            # graph runner catches callback exceptions... enforcement is done
            # in _pre_screen_input()").
            if event_type_label == "LLMStarted":
                buf = buffer.get(event.run_id)
                claimed_pre_screen = buf is not None and buf.fallback_llm_activity_id is not None
                if not claimed_pre_screen:
                    await self._client.evaluate_event(gov_event)
                return
        else:
            if event_type_label == "ChainCompleted" and not self._config.send_chain_end_event:
                return
            if event_type_label == "ToolCompleted" and not self._config.send_tool_end_event:
                return
            # ── LLMCompleted FALLBACK (Phase 5): reached only when no callback
            # owns this LLM call's completion — injected-client handlers (no
            # core runtime), a subagent-gated handler (no bridge, C1
            # blackout), or the callback simply disabled. Mirrors the
            # pre-phase-5 consumer close exactly, on the SAME id
            # `_map_event`'s `on_chat_model_start` returned as this event's
            # `activity_id` — the `-pre` row for call 1 (via `_PreScreenClaim`,
            # the fallback-path replacement for the retired
            # `_GuardrailsCallbackHandler`'s `llm_activity_map`), or the raw
            # `event_run_id` for every later call on this handler.
            if event_type_label == "LLMCompleted":
                if self._config.send_llm_start_event and gov_event.activity_id:
                    llm_activity_type = gov_event.activity_type or "llm_call"
                    completed_activity_id = f"{gov_event.activity_id}-c"
                    completed_event = LangChainGovernanceEvent(
                        source="workflow-telemetry",
                        event_type="LLMCompleted",
                        workflow_id=workflow_id,
                        run_id=run_id,
                        workflow_type=self._config.agent_name or "LangGraphRun",
                        task_queue=self._config.task_queue,
                        timestamp=rfc3339_now(),
                        session_id=self._config.session_id,
                        activity_id=completed_activity_id,
                        activity_type=llm_activity_type,
                        activity_output=gov_event.activity_output,
                        status="completed",
                        duration_ms=gov_event.duration_ms,
                        llm_model=gov_event.llm_model,
                        input_tokens=gov_event.input_tokens,
                        output_tokens=gov_event.output_tokens,
                        total_tokens=gov_event.total_tokens,
                        has_tool_calls=gov_event.has_tool_calls,
                        completion=gov_event.completion,
                        langgraph_node=gov_event.langgraph_node,
                        langgraph_step=gov_event.langgraph_step,
                    )

                    response = await self._client.evaluate_event(completed_event)
                    if response is not None:
                        context = lang_graph_event_to_context(event.event, is_root=is_root)
                        result = enforce_verdict(response, context)
                        if result.requires_hitl:
                            await poll_until_decision(
                                self._client,
                                HITLPollParams(
                                    workflow_id=workflow_id,
                                    run_id=run_id,
                                    activity_id=completed_activity_id,
                                    activity_type=llm_activity_type,
                                ),
                                self._config.hitl,
                            )
                return

        # ── Send to OpenBox Core
        response = await self._client.evaluate_event(gov_event)

        if response is None:
            return

        # ── Determine context and enforce verdict
        context = lang_graph_event_to_context(event.event, is_root=is_root)
        try:
            result = enforce_verdict(response, context)
        except (GovernanceBlockedError, GovernanceHaltError, GuardrailsValidationError):
            raise

        # ── HITL polling
        if result.requires_hitl:
            activity_id = gov_event.activity_id or event.run_id
            activity_type = gov_event.activity_type or event.name
            try:
                await poll_until_decision(
                    self._client,
                    HITLPollParams(
                        workflow_id=workflow_id,
                        run_id=run_id,
                        activity_id=activity_id,
                        activity_type=activity_type,
                    ),
                    self._config.hitl,
                )
            except (ApprovalRejectedError, ApprovalExpiredError, ApprovalTimeoutError) as e:
                raise GovernanceHaltError(str(e)) from e

    # ─────────────────────────────────────────────────────────────
    # Tool type classification
    # ─────────────────────────────────────────────────────────────

    def _resolve_tool_type(self, tool_name: str, subagent_name: str | None) -> str | None:
        """Resolve the semantic tool_type for a given tool.

        Priority:
        1. Explicit entry in tool_type_map
        2. "a2a" if subagent_name is set
        3. None for unknown tools (no classification prefix in the label)
        """
        if tool_name in self._config.tool_type_map:
            return self._config.tool_type_map[tool_name]
        if subagent_name:
            return "a2a"
        return None

    def _enrich_activity_input(
        self,
        base_input: list[Any] | None,
        tool_type: str | None,
        subagent_name: str | None,
    ) -> list[Any] | None:
        """Append an ``__openbox`` metadata entry to activity_input for Rego policy use.

        Core forwards ``activity_input`` as-is to ``input.activity_input`` in OPA.
        By appending a sentinel object, Rego policies can classify tools without
        any Core changes:

        .. code-block:: rego

            some item in input.activity_input
            meta := item["__openbox"]
            meta.subagent_name == "writer"

        Only appended when tool_type or subagent_name is set (skips for unclassified tools).
        """
        if tool_type is None and subagent_name is None:
            return base_input
        meta: dict[str, Any] = {}
        if tool_type is not None:
            meta["tool_type"] = tool_type
        if subagent_name is not None:
            meta["subagent_name"] = subagent_name
        result = list(base_input) if base_input else []
        result.append({"__openbox": meta})
        return result

    # ─────────────────────────────────────────────────────────────
    # Event mapping (LangGraph event → governance event)
    # ─────────────────────────────────────────────────────────────

    def _map_event(
        self,
        event: LangGraphStreamEvent,
        thread_id: str,
        workflow_id: str,
        run_id: str,
        root_tracker: _RootRunTracker,
        buffer: _RunBufferManager,
        *,
        skip_consumer_side_effects: bool = False,
        pre_screen_claim: _PreScreenClaim | None = None,
    ) -> tuple[LangChainGovernanceEvent | None, bool, bool, str]:
        """Map a LangGraph stream event to a governance event.

        ``skip_consumer_side_effects`` (C7, extended to LLM events in Phase
        5): True for a callback-owned `on_tool_start`/`on_tool_end` OR
        `on_chat_model_start`/`on_chat_model_end` event. Skips the OTel span
        creation/teardown and `register_activity`/`unregister_activity`
        trace-only dual-write — consumer-only bookkeeping that exists so the
        base hook runtime can resolve an activity via the EXACT trace-id
        tier. It is redundant (and would be WRONG to duplicate) for a
        callback-owned tool/LLM call: the callback runs `run_inline=True`
        (tools: INSIDE the ToolNode-seam's OWN `activity_scope`; LLM calls:
        registers its OWN trace via the injected `register_trace` callable
        BEFORE the provider HTTP call) — a SECOND, later-registered OTel
        span/trace binding from this method would only be pure overhead and,
        for LLM calls, would also disagree with the callback's activity_id
        (the pre-screen `-pre` id for call 1) on which trace maps to which
        activity. The `_RunBufferManager` registration/duration/removal
        bookkeeping is NEVER skipped — the governance event this method
        still returns (when not `None`) and the buffer's `duration_ms`
        calculation are used unconditionally.

        Returns:
            A 4-tuple of (governance_event | None, is_root, is_start, event_type_label).
        """
        ev = event.event
        event_run_id = event.run_id   # LangGraph internal run UUID for this node/tool/llm
        name = event.name
        metadata = event.metadata
        data = event.data

        langgraph_node = metadata.get("langgraph_node")
        langgraph_step = metadata.get("langgraph_step")

        subagent_name = (
            self._resolve_subagent_name(event) if self._resolve_subagent_name else None
        )

        def base(
            event_type: str,
            *,
            is_start: bool,
            **extra: Any,
        ) -> tuple[LangChainGovernanceEvent, bool, bool, str]:
            is_root = root_tracker.root_run_id == event_run_id or (
                ev == "on_chain_start" and root_tracker.is_root(event_run_id)
            )
            gov = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type=event_type,
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                langgraph_node=langgraph_node,
                langgraph_step=langgraph_step,
                **extra,
            )
            return gov, is_root, is_start, event_type

        if ev == "on_chain_start":
            is_root = root_tracker.is_root(event_run_id)
            if is_root:
                buffer.register(
                    event_run_id, "graph", name, thread_id, langgraph_node, langgraph_step
                )
                gov = LangChainGovernanceEvent(
                    source="workflow-telemetry",
                    event_type="ChainStarted",
                    workflow_id=workflow_id,
                    run_id=run_id,
                    workflow_type=self._config.agent_name or name or "LangGraphRun",
                    task_queue=self._config.task_queue,
                    timestamp=rfc3339_now(),
                    session_id=self._config.session_id,
                    activity_id=event_run_id,
                    activity_type=name,
                    activity_input=(
                        [safe_serialize(data.get("input"))]
                        if data.get("input") is not None
                        else None
                    ),
                    langgraph_node=langgraph_node,
                    langgraph_step=langgraph_step,
                )
                return gov, True, True, "ChainStarted"
            # Non-root chain = subgraph node
            if name in self._config.skip_chain_types:
                return None, False, True, "ChainStarted"
            # Skip non-subagent chains — LangGraph fires BOTH on_chain_start
            # (BaseTool's Runnable layer) AND on_tool_start (Tool layer) for
            # the same tool invocation with different run_ids, creating
            # duplicate ActivityStarted events.  on_tool_start handles tools
            # with proper span hook context; only subagent chains need this.
            if not subagent_name:
                return None, False, True, "ChainStarted"
            buffer.register(
                event_run_id,
                "subgraph" if subagent_name else "chain",
                name,
                thread_id,
                langgraph_node,
                langgraph_step,
                subagent_name,
            )
            # Use ToolStarted so to_server_event_type maps to ActivityStarted,
            # NOT WorkflowStarted — sending WorkflowCompleted for a sub-chain
            # seals the session in Core and causes all subsequent requests to HALT.
            chain_tool_type = self._resolve_tool_type(name, subagent_name)
            chain_base_input = (
                [safe_serialize(data.get("input"))] if data.get("input") is not None else None
            )
            gov = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type="ToolStarted",
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                activity_id=event_run_id,
                activity_type=name,
                activity_input=self._enrich_activity_input(
                    chain_base_input, chain_tool_type, subagent_name
                ),
                tool_name=name,
                tool_type=chain_tool_type,
                langgraph_node=langgraph_node,
                langgraph_step=langgraph_step,
                subagent_name=subagent_name,
            )
            return gov, False, True, "ChainStarted"

        if ev == "on_chain_end":
            is_root = root_tracker.root_run_id == event_run_id
            dur = buffer.duration_ms(event_run_id)
            buffer.remove(event_run_id)
            output = data.get("output")
            serialized_output = (
                safe_serialize({"result": output})
                if isinstance(output, str)
                else safe_serialize(output)
            )
            if is_root:
                gov = LangChainGovernanceEvent(
                    source="workflow-telemetry",
                    event_type="ChainCompleted",
                    workflow_id=workflow_id,
                    run_id=run_id,
                    workflow_type=self._config.agent_name or name or "LangGraphRun",
                    task_queue=self._config.task_queue,
                    timestamp=rfc3339_now(),
                    session_id=self._config.session_id,
                    activity_id=event_run_id,
                    activity_type=name,
                    workflow_output=safe_serialize(output),
                    activity_output=serialized_output,
                    status="completed",
                    duration_ms=dur,
                    langgraph_node=langgraph_node,
                    langgraph_step=langgraph_step,
                )
                return gov, True, False, "ChainCompleted"
            if name in self._config.skip_chain_types:
                return None, False, False, "ChainCompleted"
            # Skip non-subagent chains (mirrors on_chain_start skip above)
            if not subagent_name:
                return None, False, False, "ChainCompleted"
            # Use ToolCompleted → ActivityCompleted (not WorkflowCompleted)
            chain_tool_type = self._resolve_tool_type(name, subagent_name)
            gov = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type="ToolCompleted",
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                activity_id=event_run_id,
                activity_type=name,
                activity_output=serialized_output,
                tool_name=name,
                tool_type=chain_tool_type,
                status="completed",
                duration_ms=dur,
                langgraph_node=langgraph_node,
                langgraph_step=langgraph_step,
                subagent_name=subagent_name,
            )
            return gov, False, False, "ChainCompleted"

        if ev == "on_tool_start":
            if name in self._config.skip_tool_types:
                return None, False, True, "ToolStarted"

            buffer.register(event_run_id, "tool", name, thread_id, langgraph_node, langgraph_step)
            buf = buffer.get(event_run_id)
            if buf is not None:
                buf.subagent_name = subagent_name

            tool_type = self._resolve_tool_type(name, subagent_name)

            # Create OTel span to propagate trace context across asyncio.Task boundaries.
            # Tool execution happens in a spawned Task with a new OTel context — this span
            # bridges the gap so base-instrumentation child spans (httpx/db/file) inherit
            # the correct trace_id, which the base runtime resolves back to this activity
            # via its private TraceContextRegistry (the dual-write below). Only armed when
            # this handler owns a core runtime (never the injected-client lifecycle path).
            # Skipped entirely for a callback-owned tool (C7) — its ToolNode-seam
            # `activity_scope` already provides ContextVar-tier resolution.
            if should_dual_write(self._core_runtime) and not skip_consumer_side_effects:
                parent_ctx = otel_context.get_current()
                tool_span = _otel_tracer.start_span(
                    f"tool.{name}", context=parent_ctx, kind=otel_trace.SpanKind.INTERNAL,
                )
                token = otel_context.attach(otel_trace.set_span_in_context(tool_span))
                trace_id = tool_span.get_span_context().trace_id
                if trace_id:
                    # Trace-only registration into the base runtime's private
                    # trace registry — never a ContextVar bind (see
                    # activity_context_binding module docstring).
                    register_activity(
                        self._core_runtime,
                        trace_id,
                        build_activity_context(
                            config=self._config,
                            workflow_id=workflow_id,
                            run_id=run_id,
                            activity_id=event_run_id,
                            activity_type=name,
                            activity_input=safe_serialize(data.get("input")),
                            langgraph_node=langgraph_node,
                            langgraph_step=langgraph_step,
                            tool_type=tool_type,
                            tool_name=name,
                            subagent_name=subagent_name,
                            parent_ids=event.parent_ids,
                        ),
                    )
                if buf is not None:
                    buf.otel_span = tool_span
                    buf.otel_token = token
            tool_input = _unwrap_tool_input(data.get("input"))
            # NOTE: No internal span here. In the Temporal SDK, @traced spans
            # fire DURING activity execution — after ActivityStarted is stored
            # in Core.  Firing here would race with the ToolStarted event below
            # (the hook span may arrive at Core before the parent event exists).
            # The "completed" internal span fires at on_tool_end instead.
            gov = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type="ToolStarted",
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                activity_id=event_run_id,
                activity_type=name,
                activity_input=self._enrich_activity_input(
                    [safe_serialize(tool_input)], tool_type, subagent_name
                ),
                tool_name=name,
                tool_type=tool_type,
                tool_input=safe_serialize(data.get("input")),
                subagent_name=subagent_name,
                langgraph_node=langgraph_node,
                langgraph_step=langgraph_step,
            )
            return gov, False, True, "ToolStarted"

        if ev == "on_tool_end":
            if name in self._config.skip_tool_types:
                return None, False, False, "ToolCompleted"
            dur = buffer.duration_ms(event_run_id)
            buf = buffer.get(event_run_id)
            tool_type = self._resolve_tool_type(name, subagent_name)
            # One activity id for the whole tool lifecycle: ActivityCompleted
            # closes the SAME activity_id ActivityStarted opened. Core keys the
            # started/completed pair on event_type (not an id suffix), and the
            # client dedup key includes event type, so both survive distinctly.
            completed_activity_id = event_run_id

            # End OTel span created in on_tool_start and detach context
            if buf is not None and buf.otel_span is not None:
                trace_id = buf.otel_span.get_span_context().trace_id
                if buf.otel_token is not None:
                    otel_context.detach(buf.otel_token)
                buf.otel_span.end()
                # Update the same trace-id binding to the COMPLETED activity
                # identity for the brief window a completion hook could still
                # resolve it, then unregister — the trace's underlying OTel
                # span is now ended, so nothing else will ever look it up again.
                if trace_id and should_dual_write(self._core_runtime):
                    register_activity(
                        self._core_runtime,
                        trace_id,
                        build_activity_context(
                            config=self._config,
                            workflow_id=workflow_id,
                            run_id=run_id,
                            activity_id=completed_activity_id,
                            activity_type=name,
                            activity_input=data.get("output"),
                            langgraph_node=langgraph_node,
                            langgraph_step=langgraph_step,
                            tool_type=tool_type,
                            tool_name=name,
                            subagent_name=subagent_name,
                            parent_ids=event.parent_ids,
                        ),
                    )
                    unregister_activity(self._core_runtime, trace_id)

            buffer.remove(event_run_id)
            tool_output = data.get("output")
            serialized_output = (
                safe_serialize({"result": tool_output})
                if isinstance(tool_output, str)
                else safe_serialize(tool_output)
            )
            gov = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type="ToolCompleted",
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                activity_id=completed_activity_id,
                activity_type=name,
                activity_output=serialized_output,
                tool_name=name,
                tool_type=tool_type,
                subagent_name=subagent_name,
                status="completed",
                duration_ms=dur,
                langgraph_node=langgraph_node,
                langgraph_step=langgraph_step,
            )
            return gov, False, False, "ToolCompleted"

        if ev == "on_chat_model_start":
            buffer.register(event_run_id, "llm", name, thread_id, langgraph_node, langgraph_step)
            # Create OTel span to propagate trace context across asyncio.Task
            # boundaries so base-instrumentation child spans inherit this
            # trace_id, resolved back to the llm_call activity via the base
            # runtime's private trace registry. Only armed when this handler
            # owns a core runtime (never the injected-client lifecycle path)
            # AND the call is NOT callback-owned (Phase 5, C7 extended to
            # LLM) — a callback-owned call already registered ITS OWN trace
            # via the injected `register_trace` callable (same registry) in
            # `on_chat_model_start`, keyed by the callback's activity_id
            # (the pre-screen `-pre` id for call 1, not necessarily
            # `event_run_id`). A second, later registration here would either
            # be pure overhead or — worse — bind the SAME trace id to the
            # WRONG activity_id for call 1.
            if should_dual_write(self._core_runtime) and not skip_consumer_side_effects:
                parent_ctx = otel_context.get_current()
                llm_span = _otel_tracer.start_span(
                    "llm.call", context=parent_ctx, kind=otel_trace.SpanKind.INTERNAL,
                )
                token = otel_context.attach(otel_trace.set_span_in_context(llm_span))
                trace_id = llm_span.get_span_context().trace_id
                if trace_id:
                    # Registered regardless of whether LLMStarted ends up sent
                    # below (empty-prompt subagent-internal LLM calls still get
                    # span-hook governance during the call).
                    register_activity(
                        self._core_runtime,
                        trace_id,
                        build_activity_context(
                            config=self._config,
                            workflow_id=workflow_id,
                            run_id=run_id,
                            activity_id=event_run_id,
                            activity_type="llm_call",
                            langgraph_node=langgraph_node,
                            langgraph_step=langgraph_step,
                            subagent_name=subagent_name,
                            parent_ids=event.parent_ids,
                        ),
                    )
                buf = buffer.get(event_run_id)
                if buf is not None:
                    buf.otel_span = llm_span
                    buf.otel_token = token
            messages = (data.get("input") or {}).get("messages", [])
            prompt_text = _extract_prompt_from_messages(messages)
            # Skip sending empty prompts — subagent-internal LLM calls have only
            # system/tool messages, no human turn. Core's guardrail JSON-parses the
            # prompt field and returns a parse error ("Expecting value ... char 0") → block.
            if not prompt_text.strip():
                return None, False, True, "LLMStarted"
            # Mark that LLMStarted will be sent — on_chat_model_end uses this to
            # decide whether to fire the LLM span hook.  Without this guard, span
            # hooks fire for internal subagent LLM calls that have no row in Core,
            # causing Core to create orphan empty rows (duplicate with no data).
            buf = buffer.get(event_run_id)
            if buf is not None:
                buf.llm_started = True
                # Fallback-path pre-screen claim (no callback owns this call —
                # see `_PreScreenClaim`'s docstring): the FIRST non-empty-prompt
                # LLM call in the turn claims the pre-screen's `"{run_id}-pre"`
                # id as ITS fallback completion base, mirroring the retired
                # `_GuardrailsCallbackHandler`'s `llm_activity_map`. A no-op
                # (returns None) when no pre-screen response exists, or it was
                # already claimed by an earlier call this turn.
                if pre_screen_claim is not None:
                    buf.fallback_llm_activity_id = pre_screen_claim.claim()
            model_name = _extract_model_name_from_event(event) or name
            gov = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type="LLMStarted",
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                activity_id=event_run_id,
                activity_type="llm_call",
                activity_input=[{"prompt": prompt_text}],
                llm_model=model_name,
                prompt=prompt_text,
                langgraph_node=langgraph_node,
                langgraph_step=langgraph_step,
            )
            return gov, False, True, "LLMStarted"

        if ev == "on_chat_model_end":
            dur = buffer.duration_ms(event_run_id)
            buf = buffer.get(event_run_id)
            llm_started = buf.llm_started if buf else False
            # Fallback-path pre-screen claim (see `on_chat_model_start` above
            # and `_PreScreenClaim`'s docstring): call 1's completion base id
            # is the `-pre` row it claimed, not its own `event_run_id` — this
            # is what lets the fallback close below land on the SAME row
            # `_pre_screen_input` opened, matching pre-phase-5 behavior.
            fallback_activity_id = (
                buf.fallback_llm_activity_id if buf is not None else None
            ) or event_run_id

            # End OTel span created in on_chat_model_start and detach context.
            # A callback-owned call never had a span created above (skipped by
            # `skip_consumer_side_effects`), so `buf.otel_span` is None there
            # and this whole block is a no-op — the callback's OWN
            # `unregister_trace` call (`on_llm_end`/`on_llm_error`) is what
            # unregisters ITS trace binding, on ITS activity_id.
            if buf is not None and buf.otel_span is not None:
                llm_trace_id = buf.otel_span.get_span_context().trace_id
                if buf.otel_token is not None:
                    otel_context.detach(buf.otel_token)
                buf.otel_span.end()
                # Cleanup: the LLM's OWN OTel span/trace_id (distinct from the
                # parent tool's, if any) is ending now — unregister it from the
                # base registry so it never resolves after this point.
                if llm_trace_id and should_dual_write(self._core_runtime):
                    unregister_activity(self._core_runtime, llm_trace_id)

            buffer.remove(event_run_id)
            # Skip if LLMStarted was never sent (empty/no human-turn prompt).
            # Firing a hook_trigger span for a non-existent row creates an
            # orphan empty ActivityStarted row in Core.
            if not llm_started:
                return None, False, False, "LLMCompleted"
            llm_output = data.get("output") or {}
            input_tokens, output_tokens, total_tokens = _extract_token_usage(llm_output)
            completion_text = _extract_completion_text(llm_output)
            model_name = (
                _extract_model_name_from_output(llm_output)
                or _extract_model_name_from_event(event)
                or name
            )
            has_tool_calls = bool(_extract_tool_calls(llm_output))
            # NOTE: No span hook for LLM calls.  The user's hard rule:
            # "every activity started … not the LLM prompt should have a span call"
            # LLM events are explicitly excluded from the span requirement.
            # Additionally, the LLMStarted activity_id (from _pre_screen_input or
            # the shared core callback) doesn't reliably match event.run_id here,
            # so a span hook would create an orphan governance event (duplicate).
            gov = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type="LLMCompleted",
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                activity_id=fallback_activity_id,
                activity_output=safe_serialize(llm_output),
                status="completed",
                duration_ms=dur,
                llm_model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                has_tool_calls=has_tool_calls,
                completion=completion_text,
                langgraph_node=langgraph_node,
                langgraph_step=langgraph_step,
            )
            return gov, False, False, "LLMCompleted"

        # Streaming chunks and other events — not governed
        return None, False, False, ""


# ═══════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════

def create_openbox_graph_handler(
    graph: Any,
    *,
    api_url: str,
    api_key: str,
    governance_timeout: float = 30.0,
    validate: bool = True,
    enable_telemetry: bool = True,
    sqlalchemy_engine: Any = None,
    agent_did: str | None = None,
    agent_private_key: str | None = None,
    **handler_kwargs: Any,
) -> OpenBoxLangGraphHandler:
    """Create a fully configured `OpenBoxLangGraphHandler` wrapping a compiled LangGraph graph.

    Calls the synchronous `initialize()` to validate credentials and set up global config,
    then returns a ready-to-use `OpenBoxLangGraphHandler`.

    Args:
        graph: A compiled LangGraph graph (e.g. `StateGraph.compile()`).
        api_url: Base URL of your OpenBox Core instance.
        api_key: API key in `obx_live_*` or `obx_test_*` format.
        governance_timeout: HTTP timeout in **seconds** for governance calls (default 30.0).
        validate: If True, validates the API key against the server on startup.
        enable_telemetry: Reserved for future HTTP-span telemetry patching.
        sqlalchemy_engine: Optional SQLAlchemy Engine instance to instrument for DB
            governance. Required when the engine is created before the handler.
        agent_did: Optional OpenBox agent DID. Falls back to `OPENBOX_AGENT_DID`.
        agent_private_key: Optional raw Ed25519 private key seed. Falls back to
            `OPENBOX_AGENT_PRIVATE_KEY`.
        **handler_kwargs: Additional keyword arguments forwarded to
            `OpenBoxLangGraphHandlerOptions`.

    Returns:
        A configured `OpenBoxLangGraphHandler` ready to govern the graph.

    Example:
        >>> governed = create_openbox_graph_handler(
        ...     graph=my_graph,
        ...     api_url=os.environ["OPENBOX_URL"],
        ...     api_key=os.environ["OPENBOX_API_KEY"],
        ...     agent_name="MyAgent",
        ...     hitl={"enabled": True, "poll_interval_ms": 5000, "max_wait_ms": 300000},
        ... )
    """
    from openbox_langgraph.config import initialize
    initialize(
        api_url=api_url,
        api_key=api_key,
        governance_timeout=governance_timeout,
        validate=validate,
        agent_did=agent_did,
        agent_private_key=agent_private_key,
    )

    options = OpenBoxLangGraphHandlerOptions(
        api_timeout=governance_timeout,
        sqlalchemy_engine=sqlalchemy_engine,
        **{k: v for k, v in handler_kwargs.items() if hasattr(OpenBoxLangGraphHandlerOptions, k)},
    )
    return OpenBoxLangGraphHandler(graph, options)


# ═══════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════

def _extract_thread_id(config: dict[str, Any] | None) -> str:
    """Extract thread_id from a LangGraph RunnableConfig dict."""
    if not config:
        return "default"
    configurable = config.get("configurable") or {}
    return configurable.get("thread_id") or "default"


def _unwrap_tool_input(raw: Any) -> Any:
    """Unwrap potentially double-encoded JSON tool input."""
    import json

    if not isinstance(raw, str):
        return raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            if list(parsed.keys()) == ["input"] and isinstance(parsed["input"], str):
                return json.loads(parsed["input"])
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return raw


def _extract_prompt_from_messages(messages: Any) -> str:
    """Extract the last human/user message text from a LangChain messages structure."""
    if not isinstance(messages, (list, tuple)):
        return ""
    flat: list[Any] = []
    for item in messages:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    for msg in reversed(flat):
        if hasattr(msg, "content"):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get("content", "")
        else:
            continue
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            return " ".join(parts)
    return ""


def _extract_model_name_from_event(event: LangGraphStreamEvent) -> str | None:
    """Extract model name from event metadata."""
    return (
        event.metadata.get("ls_model_name")
        or event.metadata.get("model_name")
        or None
    )


def _extract_model_name_from_output(output: Any) -> str | None:
    """Extract model name from LLM output dict."""
    if not isinstance(output, dict):
        return None
    meta = output.get("response_metadata") or {}
    return meta.get("model_name") or meta.get("model") or output.get("model") or None


def _extract_token_usage(output: Any) -> tuple[int | None, int | None, int | None]:
    """Extract (input_tokens, output_tokens, total_tokens) from an LLM output dict."""
    if not isinstance(output, dict):
        return None, None, None
    usage = (
        output.get("usage_metadata") or output.get("response_metadata", {}).get("usage", {}) or {}
    )
    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
    total = usage.get("total_tokens") or (
        (input_tokens or 0) + (output_tokens or 0) if (input_tokens or output_tokens) else None
    )
    return input_tokens, output_tokens, total


def _extract_completion_text(output: Any) -> str | None:
    """Extract the assistant completion text from an LLM output dict."""
    if not isinstance(output, dict):
        return None
    # LangChain AIMessage structure
    content = output.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return " ".join(parts) if parts else None
    return None


def _extract_tool_calls(output: Any) -> list[Any]:
    """Return tool_calls list from an LLM output dict (empty list if none)."""
    if not isinstance(output, dict):
        return []
    tool_calls = output.get("tool_calls") or []
    if tool_calls:
        return tool_calls
    # LangChain AIMessage wraps tool_calls in additional_kwargs
    additional = output.get("additional_kwargs") or {}
    return additional.get("tool_calls") or []
