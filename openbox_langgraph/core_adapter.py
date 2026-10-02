# openbox_langgraph/core_adapter.py
"""Map base-SDK verdicts to native errors and wait at pending operations.

A handler configures an activity waiter for each governed turn. Standalone
adapters without a waiter still raise REQUIRE_APPROVAL to fail closed.
"""

from __future__ import annotations

from typing import NoReturn

from openbox_core.context import ContextStore
from openbox_core.contracts.context import ActivityContext
from openbox_core.contracts.results import EvaluationResult, Verdict

from openbox_langgraph.activity_approval import ActivityApprovalWaiter
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
        self.approval_waiter: ActivityApprovalWaiter | None = None

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

    # ── Approval: resume the same operation after its decision ────────────

    async def handle_approval(
        self, result: EvaluationResult, context: ActivityContext | None = None
    ) -> None:
        """Suspend this coroutine without unwinding the graph or blocking the loop."""
        ctx = context if context is not None else self._store.current_activity_context()
        if self.approval_waiter is None or ctx is None:
            self._raise_pending_approval(result, ctx)
        try:
            await self.approval_waiter.wait(result, ctx)
        except Exception:
            self._store.mark_activity_aborted(ctx.workflow_id, ctx.activity_id)
            raise

    def handle_approval_sync(
        self, result: EvaluationResult, *, context: ActivityContext | None = None
    ) -> None:
        """Suspend the sync tool's worker thread, preserving its current stack."""
        ctx = context if context is not None else self._store.current_activity_context()
        if self.approval_waiter is None or ctx is None:
            self._raise_pending_approval(result, ctx)
        try:
            self.approval_waiter.wait_sync(result, ctx)
        except Exception:
            self._store.mark_activity_aborted(ctx.workflow_id, ctx.activity_id)
            raise

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

    def on_completed_hook_result(
        self, result: EvaluationResult, context: ActivityContext | None = None
    ) -> None:
        """Completed verdicts affect FUTURE execution only — the operation
        already ran. ``HookRuntime._after_completed`` has already marked the
        abort/halt flags on the base ``ContextStore`` for a stop-shaped
        result; nothing further to do here (parity with the Temporal
        adapter's ``on_completed_hook_result``, which is also a no-op)."""
        return None

    # ── Compatibility helper for callers managing abort marks themselves ──

    def reset_after_approval(self, workflow_id: str | None) -> None:
        """Clear a workflow's abort marks for legacy callers.

        Normal governed turns wait in place and do not use this helper.
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
