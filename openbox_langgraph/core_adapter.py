# openbox_langgraph/core_adapter.py
"""LangGraph ``FrameworkAdapter`` for the opt-in ``openbox_core`` hook runtime.

Maps base-SDK governance verdicts onto the LangGraph-native error types
(``openbox_langgraph.errors.GovernanceBlockedError``/``GovernanceHaltError``)
the handler's existing ``ainvoke``/``astream_governed`` catch blocks already
understand — no new error vocabulary, no new control flow at the call site.

RAISE-ONLY approval, mirroring the legacy hook path exactly: an inline
blocking poller on the event-loop thread (``time.sleep`` in
``ApprovalPoller.wait_for_decision``) would freeze every other coroutine on
that loop — LangGraph tools/LLM calls run as concurrent ``asyncio.Task``s, so
blocking the loop thread blocks ALL of them, not just the approval-pending
one. The handler's OUTER catch/poll/retry in ``ainvoke``/``_pre_screen_input``
is the single HITL driver today (catches ``GovernanceBlockedError`` with
``verdict == "require_approval"``, awaits ``poll_until_decision``, retries);
this adapter reproduces that same shape for hook-level (started-stage
HTTP/DB/file/function) verdicts so the SAME outer loop drives them too.
Defining ``handle_approval_sync`` here also pre-empts
``openbox_core.hooks.preflight.HookRuntime._sync_approval``'s fallback to its
own inline ``ApprovalPoller`` (adapter-native flow always wins when present).
"""

from __future__ import annotations

from typing import NoReturn

from openbox_core.context import ContextStore
from openbox_core.contracts.context import ActivityContext
from openbox_core.contracts.results import EvaluationResult, Verdict

from openbox_langgraph.errors import GovernanceBlockedError, GovernanceHaltError

__all__ = ["LangGraphFrameworkAdapter"]


class LangGraphFrameworkAdapter:
    """Maps base-SDK governance outcomes onto LangGraph-native errors.

    Args:
        context_store: The runtime's private ``ContextStore`` (registration
            target). Falls back to ``ContextStore.current_activity_context()``
            when no explicit context is passed (started-hook / lifecycle
            paths never receive one — see ``handle_approval_sync``'s docstring
            for why the sync-approval path DOES).
    """

    name = "langgraph"

    def __init__(
        self,
        *,
        context_store: ContextStore | None = None,
    ) -> None:
        self._store = context_store if context_store is not None else ContextStore()

    # ── Lifecycle verdicts (WorkflowStarted/LLMStarted pre-screen, etc.) ───

    def raise_lifecycle_blocked(self, result: EvaluationResult) -> NoReturn:
        """HALT/BLOCK on a lifecycle event -> the SDK-level error shape.

        No hook-identifier context here (lifecycle events carry no
        URL/file-path ``identifier``) — uses the 3-arg SDK-level
        ``GovernanceBlockedError(reason, policy_id, risk_score)`` convention,
        matching ``verdict_handler.enforce_verdict``'s existing HALT/BLOCK
        branches exactly.
        """
        reason = result.reason or "Blocked by governance policy"
        if result.verdict is Verdict.HALT:
            raise GovernanceHaltError(
                reason, policy_id=result.policy_id, risk_score=result.risk_score
            )
        raise GovernanceBlockedError(reason, result.policy_id, result.risk_score)

    # ── Started-hook verdicts (HTTP/DB/file/function preflight) ────────────

    def raise_hook_blocked(self, result: EvaluationResult) -> NoReturn:
        """HALT/BLOCK on a started hook -> hook-shape error + abort-mark.

        Self-marks the abort flag (idempotent — ``mark_activity_aborted`` adds
        to a set) rather than relying solely on the caller
        (``HookRuntime._mark_stopped`` already marks it upstream), so this
        adapter holds under direct unit-test construction too, not only when
        driven through the full ``HookRuntime`` call chain.
        """
        ctx = self._store.current_activity_context()
        if ctx is not None:
            self._store.mark_activity_aborted(ctx.workflow_id, ctx.activity_id)
        identifier = _resolve_identifier(result, ctx)
        reason = result.reason or "Blocked by governance"
        raise GovernanceBlockedError(result.verdict.value, reason, identifier)

    # ── Approval (RAISE-ONLY — never an inline blocking wait) ──────────────

    async def handle_approval(self, result: EvaluationResult) -> None:
        """Async started-hook REQUIRE_APPROVAL -> raise, never await inline.

        The base ``HookRuntime._adecide_started`` treats a normal RETURN as
        "approved, proceed" — raising here is the correct "not approved yet"
        signal for a ``requires_approval()`` verdict, so the handler's outer
        catch/poll/retry loop drives the approval flow.
        """
        self._raise_pending_approval(result, self._store.current_activity_context())

    def handle_approval_sync(
        self, result: EvaluationResult, *, context: ActivityContext | None = None
    ) -> None:
        """Sync started-hook REQUIRE_APPROVAL -> raise, never poll inline.

        ``context`` is the span-resolved ``ActivityContext``
        ``HookRuntime._sync_approval`` passes explicitly — it can differ from
        the ambient ``ContextStore.current_activity_context()`` (e.g. a sync
        tool running inside ``run_in_executor``, where the ContextVar bound on
        the async stream-consumer never reached the worker thread). Preferring
        the passed context over the ambient lookup keeps the abort-mark keyed
        on the SAME workflow/activity id the operation is actually running
        under.

        Defining this method is what stops
        ``HookRuntime._sync_approval`` from falling back to its own inline
        ``ApprovalPoller.wait_for_decision`` (a blocking ``time.sleep`` loop on
        whatever thread called this) — an adapter-native
        ``handle_approval_sync`` always takes priority when present.
        """
        ctx = context if context is not None else self._store.current_activity_context()
        self._raise_pending_approval(result, ctx)

    def _raise_pending_approval(
        self, result: EvaluationResult, ctx: ActivityContext | None
    ) -> NoReturn:
        if ctx is not None:
            self._store.mark_activity_aborted(ctx.workflow_id, ctx.activity_id)
        # The identifier here becomes the HITL poll key — must be the bound
        # activity_id (what the backend matches on), NOT a raw-echoed resource id.
        identifier = _resolve_approval_activity_id(result, ctx)
        reason = result.reason or "Approval required"
        raise GovernanceBlockedError("require_approval", reason, identifier)

    # ── Completed-hook telemetry (never undoes the operation) ──────────────

    def on_completed_hook_result(self, result: EvaluationResult) -> None:
        """Completed verdicts affect FUTURE execution only — the operation
        already ran. ``HookRuntime._after_completed`` has already marked the
        abort/halt flags on the base ``ContextStore`` for a stop-shaped
        result; nothing further to do here (parity with the Temporal
        adapter's ``on_completed_hook_result``, which is also a no-op)."""
        return None

    # ── Post-approval reset (clears the base store before the retry) ───────

    def reset_after_approval(self, workflow_id: str | None) -> None:
        """Clear every abort mark registered under ``workflow_id`` on the base
        store, so the caller's retry (already GRANTED by
        ``poll_until_decision``) runs GOVERNED instead of short-circuiting on
        a stale abort flag from the blocked first pass.

        Scoped to ``workflow_id`` (the per-TURN id), not a single
        ``activity_id``: an approved retry re-invokes the underlying graph
        directly (bypassing this SDK's own registration), so the exact
        ``activity_id`` its operations will use is not knowable up front — but
        every activity a hook could have aborted THIS turn shares the SAME
        ``workflow_id``. See ``TraceContextRegistry.clear_aborted_for_workflow``
        for why the store needed a new, narrower-than-``sweep`` method for this
        (``sweep`` also drops trace state the retry still needs).

        Call this AFTER ``poll_until_decision`` resolves and BEFORE
        re-invoking the graph — never before the poll (that would let a
        second concurrent hook evaluation ignore the still-pending approval).

        Clears the base store — via the ``TraceContextRegistry`` published on
        the store as ``store.registry`` (``create_core_runtime`` sets it to the
        runtime's registry; duck-typed by attribute, not import, to avoid a
        circular import), which sweeps every activity key registered this turn.
        Falls back to a direct ``clear_activity_aborted`` on the ambient bound
        context when no registry is published (a store built outside
        ``create_core_runtime``).

        No-op when ``workflow_id`` is falsy (nothing is ever keyed on it).
        """
        if not workflow_id:
            return
        registry = getattr(self._store, "registry", None)
        if registry is not None and hasattr(registry, "clear_aborted_for_workflow"):
            registry.clear_aborted_for_workflow(workflow_id)
        else:
            ctx = self._store.current_activity_context()
            if ctx is not None and ctx.workflow_id == workflow_id:
                self._store.clear_activity_aborted(ctx.workflow_id, ctx.activity_id)


def _resolve_identifier(result: EvaluationResult, ctx: ActivityContext | None) -> str:
    """Best-effort resource identifier for a hook-shape BLOCK error's 3rd arg.

    Used for the DISPLAY/log identifier on ``raise_hook_blocked`` — a
    server-echoed ``result.raw["identifier"]`` (e.g. the offending URL/path)
    is the meaningful value there and takes precedence; falls back to the bound
    activity id, then empty.

    NOTE: this is NOT the approval poll key. Approval polling matches on
    ``activity_id`` (see ``_resolve_approval_activity_id``), so the approval
    path must not use this raw-first resolution.
    """
    if result.raw.get("identifier"):
        return str(result.raw["identifier"])
    if ctx is not None and ctx.activity_id:
        return ctx.activity_id
    return ""


def _resolve_approval_activity_id(result: EvaluationResult, ctx: ActivityContext | None) -> str:
    """Poll key for a REQUIRE_APPROVAL raise — the backend matches a pending
    approval strictly on ``activity_id``, so the bound activity id is the
    known-correct key and takes precedence. A server-echoed
    ``result.raw["identifier"]`` may be an unrelated resource/policy id; using
    it would poll a key the backend never resolves and hang the unbounded
    poller. Only when no bound activity id exists do we fall back to any echoed
    identifier, then empty (caller then uses its own synthetic-hook fallback).
    """
    if ctx is not None and ctx.activity_id:
        return ctx.activity_id
    if result.raw.get("identifier"):
        return str(result.raw["identifier"])
    return ""
