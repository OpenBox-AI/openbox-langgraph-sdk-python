"""Unit tests for `LangGraphFrameworkAdapter` — verdict-to-error translation,
raise-only approval, abort-marking, and the post-approval reset — in
isolation from any real LangGraph graph or HookRuntime call chain.

Exercises the adapter's public FrameworkAdapter-protocol surface directly
against hand-built `EvaluationResult`s, so a translation bug surfaces here
without needing a real Core response or OTel-instrumented operation (that
coverage lives in test_core_conformance_suite.py instead).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openbox_core")

from openbox_core.context import ContextStore
from openbox_core.contracts.context import ActivityContext
from openbox_core.contracts.results import EvaluationResult, Verdict

from openbox_langgraph.core_adapter import LangGraphFrameworkAdapter
from openbox_langgraph.errors import GovernanceBlockedError, GovernanceHaltError

_CTX = ActivityContext(
    workflow_id="wf-adapter-test",
    run_id="run-adapter-test",
    workflow_type="AdapterTestWorkflow",
    task_queue="adapter-test-queue",
    activity_id="act-adapter-test",
    activity_type="http_request",
)


def _bound_adapter(
    store: ContextStore | None = None,
) -> tuple[LangGraphFrameworkAdapter, ContextStore]:
    store = store if store is not None else ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    store.bind(_CTX)
    return adapter, store


# ── raise_lifecycle_blocked ─────────────────────────────────────────────────


class TestRaiseLifecycleBlocked:
    def test_halt_raises_governance_halt_error(self) -> None:
        adapter, _ = _bound_adapter()
        result = EvaluationResult(verdict=Verdict.HALT, reason="emergency stop", policy_id="p1")
        with pytest.raises(GovernanceHaltError) as exc_info:
            adapter.raise_lifecycle_blocked(result)
        assert str(exc_info.value) == "emergency stop"
        assert exc_info.value.policy_id == "p1"

    def test_block_raises_governance_blocked_error_sdk_shape(self) -> None:
        adapter, _ = _bound_adapter()
        result = EvaluationResult(verdict=Verdict.BLOCK, reason="policy violation", risk_score=0.9)
        with pytest.raises(GovernanceBlockedError) as exc_info:
            adapter.raise_lifecycle_blocked(result)
        assert exc_info.value.verdict == "block"
        assert str(exc_info.value) == "policy violation"
        assert exc_info.value.risk_score == 0.9

    def test_missing_reason_falls_back_to_default_message(self) -> None:
        adapter, _ = _bound_adapter()
        with pytest.raises(GovernanceBlockedError) as exc_info:
            adapter.raise_lifecycle_blocked(EvaluationResult(verdict=Verdict.BLOCK))
        assert str(exc_info.value) == "Blocked by governance policy"


# ── raise_hook_blocked ───────────────────────────────────────────────────────


class TestRaiseHookBlocked:
    def test_block_raises_hook_shape_error_and_marks_abort(self) -> None:
        adapter, store = _bound_adapter()
        result = EvaluationResult(verdict=Verdict.BLOCK, reason="bad request")
        with pytest.raises(GovernanceBlockedError) as exc_info:
            adapter.raise_hook_blocked(result)
        assert exc_info.value.verdict == "block"
        assert str(exc_info.value) == "bad request"
        assert store.is_activity_aborted(_CTX.workflow_id, _CTX.activity_id)

    def test_halt_raises_hook_shape_error_with_halt_verdict(self) -> None:
        adapter, store = _bound_adapter()
        result = EvaluationResult(verdict=Verdict.HALT, reason="stop everything")
        with pytest.raises(GovernanceBlockedError) as exc_info:
            adapter.raise_hook_blocked(result)
        assert exc_info.value.verdict == "halt"
        assert store.is_activity_aborted(_CTX.workflow_id, _CTX.activity_id)

    def test_no_bound_context_still_raises_without_marking(self) -> None:
        store = ContextStore()
        adapter = LangGraphFrameworkAdapter(context_store=store)
        with pytest.raises(GovernanceBlockedError):
            adapter.raise_hook_blocked(EvaluationResult(verdict=Verdict.BLOCK, reason="x"))
        # Nothing bound -> nothing to key the abort mark on; must not raise
        # a second, unrelated error trying to mark a None workflow/activity.
        assert not store.is_activity_aborted(None, None)

    def test_identifier_prefers_raw_identifier_over_activity_id(self) -> None:
        adapter, _ = _bound_adapter()
        result = EvaluationResult(
            verdict=Verdict.BLOCK, reason="x", raw={"identifier": "https://evil.example.com"}
        )
        with pytest.raises(GovernanceBlockedError) as exc_info:
            adapter.raise_hook_blocked(result)
        assert exc_info.value.identifier == "https://evil.example.com"

    def test_identifier_falls_back_to_bound_activity_id(self) -> None:
        adapter, _ = _bound_adapter()
        with pytest.raises(GovernanceBlockedError) as exc_info:
            adapter.raise_hook_blocked(EvaluationResult(verdict=Verdict.BLOCK, reason="x"))
        assert exc_info.value.identifier == _CTX.activity_id


# ── handle_approval / handle_approval_sync (RAISE-ONLY) ─────────────────────


class TestApprovalIsRaiseOnly:
    async def test_async_handle_approval_raises_require_approval_shape(self) -> None:
        adapter, store = _bound_adapter()
        result = EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL, reason="needs sign-off")
        with pytest.raises(GovernanceBlockedError) as exc_info:
            await adapter.handle_approval(result)
        assert exc_info.value.verdict == "require_approval"
        assert store.is_activity_aborted(_CTX.workflow_id, _CTX.activity_id)

    def test_sync_handle_approval_raises_require_approval_shape(self) -> None:
        adapter, store = _bound_adapter()
        result = EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL, reason="needs sign-off")
        with pytest.raises(GovernanceBlockedError) as exc_info:
            adapter.handle_approval_sync(result)
        assert exc_info.value.verdict == "require_approval"
        assert store.is_activity_aborted(_CTX.workflow_id, _CTX.activity_id)

    def test_sync_handle_approval_prefers_explicit_context_over_ambient(self) -> None:
        """`context=` passed explicitly (the span-resolved context
        `HookRuntime._sync_approval` supplies) must win over whatever is
        ambiently bound — proves a sync tool in a different thread than the
        bind still gets marked under the RIGHT workflow/activity id."""
        store = ContextStore()
        adapter = LangGraphFrameworkAdapter(context_store=store)
        other_ctx = ActivityContext(
            workflow_id="wf-other", run_id="run-other", activity_id="act-other",
            activity_type="file_operation",
        )
        with pytest.raises(GovernanceBlockedError):
            adapter.handle_approval_sync(
                EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL, reason="x"),
                context=other_ctx,
            )
        assert store.is_activity_aborted("wf-other", "act-other")
        assert not store.is_activity_aborted(_CTX.workflow_id, _CTX.activity_id)

    async def test_handle_approval_never_returns_normally(self) -> None:
        """A normal return from `handle_approval` means APPROVED to the base
        HookRuntime — this adapter must NEVER do that (raise-only contract),
        confirmed by exhaustiveness: every verdict this method is called for
        is REQUIRE_APPROVAL by construction, and it always raises."""
        adapter, _ = _bound_adapter()
        with pytest.raises(GovernanceBlockedError):
            await adapter.handle_approval(
                EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL, reason="x")
            )

    async def test_approval_poll_key_ignores_server_echoed_identifier(self) -> None:
        """C5 hang guard: the approval identifier is the HITL POLL KEY, and Core
        matches a pending approval on activity_id. Even if a Core response echoes
        an unrelated `identifier` (a policy/resource id), the raise must carry the
        bound activity_id — polling the echoed value would target a key Core never
        resolves and hang the unbounded poller. (Unlike raise_hook_blocked, whose
        DISPLAY identifier legitimately prefers the echoed value.)"""
        adapter, _ = _bound_adapter()
        result = EvaluationResult(
            verdict=Verdict.REQUIRE_APPROVAL,
            reason="needs sign-off",
            raw={"identifier": "policy-42-not-an-activity-id"},
        )
        with pytest.raises(GovernanceBlockedError) as exc_info:
            await adapter.handle_approval(result)
        assert exc_info.value.identifier == _CTX.activity_id

    def test_sync_approval_poll_key_ignores_server_echoed_identifier(self) -> None:
        """Sync counterpart of the C5 approval poll-key guard."""
        adapter, _ = _bound_adapter()
        result = EvaluationResult(
            verdict=Verdict.REQUIRE_APPROVAL,
            reason="needs sign-off",
            raw={"identifier": "policy-42-not-an-activity-id"},
        )
        with pytest.raises(GovernanceBlockedError) as exc_info:
            adapter.handle_approval_sync(result)
        assert exc_info.value.identifier == _CTX.activity_id


# ── on_completed_hook_result ─────────────────────────────────────────────────


class TestOnCompletedHookResult:
    def test_completed_result_never_raises_and_never_undoes(self) -> None:
        adapter, _ = _bound_adapter()
        # A stop-shaped completed result must not raise — the operation
        # already ran; only future execution can be affected, and that
        # marking is the base HookRuntime's job (_after_completed), not this
        # callback's.
        adapter.on_completed_hook_result(EvaluationResult(verdict=Verdict.BLOCK, reason="x"))
        adapter.on_completed_hook_result(EvaluationResult(verdict=Verdict.ALLOW))


# ── reset_after_approval (the post-approval reset) ───────────────────────────


class TestResetAfterApproval:
    def test_clears_base_store_abort_mark_for_workflow(self) -> None:
        adapter, store = _bound_adapter()
        with pytest.raises(GovernanceBlockedError):
            adapter.raise_hook_blocked(EvaluationResult(verdict=Verdict.BLOCK, reason="x"))
        assert store.is_activity_aborted(_CTX.workflow_id, _CTX.activity_id)

        adapter.reset_after_approval(_CTX.workflow_id)

        assert not store.is_activity_aborted(_CTX.workflow_id, _CTX.activity_id)

    def test_falsy_workflow_id_is_a_noop(self) -> None:
        adapter, store = _bound_adapter()
        with pytest.raises(GovernanceBlockedError):
            adapter.raise_hook_blocked(EvaluationResult(verdict=Verdict.BLOCK, reason="x"))
        adapter.reset_after_approval(None)
        adapter.reset_after_approval("")
        # Still aborted — a falsy workflow_id must never accidentally clear
        # everything.
        assert store.is_activity_aborted(_CTX.workflow_id, _CTX.activity_id)
