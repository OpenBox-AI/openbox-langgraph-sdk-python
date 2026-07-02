"""Run the base-SDK conformance kit against `LangGraphFrameworkAdapter`.

Mirrors the Temporal migration's `tests/test_core_conformance_suite.py`:
proves the LangGraph adapter drives REAL base instrumentation (not a
re-implementation of it) with LangGraph-native semantics — started BLOCK/HALT
prevent the real operation and surface as `GovernanceBlockedError`/
`GovernanceHaltError`; REQUIRE_APPROVAL RAISES (never an inline blocking
wait — the LangGraph event loop must stay responsive); completed verdicts
only mark future execution, never undo the operation that already ran.
"""

from __future__ import annotations

import pytest
import requests

pytest.importorskip("openbox_core")

from openbox_core.conformance.fake_core import FakeCore, assert_hook_wire_shape
from openbox_core.conformance.instrumentation import (
    LocalCountingServer,
    installed_conformance_runtime,
)
from openbox_core.context import ContextStore, activity_scope
from openbox_core.contracts.context import ActivityContext

from openbox_langgraph.core_adapter import LangGraphFrameworkAdapter
from openbox_langgraph.errors import GovernanceBlockedError, GovernanceHaltError

_CONTEXT = ActivityContext(
    workflow_id="wf-langgraph-conf",
    run_id="run-langgraph-conf",
    workflow_type="LangGraphConfWorkflow",
    task_queue="langgraph-conf-queue",
    activity_id="act-langgraph-conf",
    activity_type="http_request",
)


@pytest.fixture(scope="module")
def server():
    srv = LocalCountingServer()
    yield srv
    srv.stop()


class TestLangGraphAdapterConformance:
    def test_http_block_prevents_the_real_request_and_raises_blocked(self, server) -> None:
        fake_core = FakeCore({"verdict": "block", "reason": "policy"})
        store = ContextStore()
        adapter = LangGraphFrameworkAdapter(context_store=store)
        with installed_conformance_runtime(fake_core, adapter, store):
            with activity_scope(_CONTEXT, store=store):
                before = server.hits
                with pytest.raises(GovernanceBlockedError) as exc_info:
                    requests.get(server.url, timeout=5)
                assert server.hits == before  # blocked BEFORE the request ran
        assert exc_info.value.verdict == "block"
        assert_hook_wire_shape(fake_core.started_payloads[0])

    def test_http_halt_prevents_the_real_request_and_sets_halt_flag(self, server) -> None:
        fake_core = FakeCore({"verdict": "halt", "reason": "emergency"})
        store = ContextStore()
        adapter = LangGraphFrameworkAdapter(context_store=store)
        with installed_conformance_runtime(fake_core, adapter, store):
            with activity_scope(_CONTEXT, store=store):
                before = server.hits
                with pytest.raises(GovernanceBlockedError) as exc_info:
                    requests.get(server.url, timeout=5)
                assert server.hits == before
                assert store.halt_requested  # adapter/handler decides how to stop future work
        assert exc_info.value.verdict == "halt"

    async def test_require_approval_raises_never_polls_inline(self, server) -> None:
        """Async path: REQUIRE_APPROVAL must RAISE (the LangGraph outer
        catch/poll/retry loop drives HITL) — never await an inline poller,
        which would block the event loop LangGraph's concurrent tool/LLM
        tasks share."""
        import httpx

        fake_core = FakeCore({"verdict": "require_approval", "approval_id": "app-1"})
        store = ContextStore()
        adapter = LangGraphFrameworkAdapter(context_store=store)
        with installed_conformance_runtime(fake_core, adapter, store):
            with activity_scope(_CONTEXT, store=store):
                before = server.hits
                async with httpx.AsyncClient() as client:
                    with pytest.raises(GovernanceBlockedError) as exc_info:
                        await client.get(server.url)
                assert server.hits == before  # never ran — still pending approval
        assert exc_info.value.verdict == "require_approval"
        # No approval poll ever reached Core — raise-only, no inline wait.
        assert fake_core.approval_requests == []

    def test_sync_require_approval_raises_never_polls_inline(self, server) -> None:
        """Sync path: defining `handle_approval_sync` pre-empts
        `HookRuntime._sync_approval`'s fallback to its own inline
        `ApprovalPoller` — confirmed by NO approval request ever reaching the
        fake Core (an inline poller would have sent one after the first
        sleep interval)."""
        fake_core = FakeCore({"verdict": "require_approval", "approval_id": "app-2"})
        store = ContextStore()
        adapter = LangGraphFrameworkAdapter(context_store=store)
        with installed_conformance_runtime(fake_core, adapter, store):
            with activity_scope(_CONTEXT, store=store):
                before = server.hits
                with pytest.raises(GovernanceBlockedError) as exc_info:
                    requests.get(server.url, timeout=5)
                assert server.hits == before
        assert exc_info.value.verdict == "require_approval"
        assert fake_core.approval_requests == []

    def test_completed_block_runs_the_operation_and_never_undoes_it(self, server) -> None:
        fake_core = FakeCore(
            {"verdict": "allow"},
            {"verdict": "block", "reason": "post-hoc"},
        )
        store = ContextStore()
        adapter = LangGraphFrameworkAdapter(context_store=store)
        with installed_conformance_runtime(fake_core, adapter, store):
            with activity_scope(_CONTEXT, store=store):
                response = requests.get(server.url, timeout=5)  # runs; NOT undone
                assert response.status_code == 200
        assert store.is_activity_aborted(_CONTEXT.workflow_id, _CONTEXT.activity_id)

    def test_self_instrumentation_impossible(self, server) -> None:
        """The evaluate call to Core's own URL must never re-govern itself —
        confirmed here via the SAME `should_ignore_url`/`api_url` guard the
        base InstrumentationManager wires automatically on install; a
        governed request to the REAL target still fires exactly once."""
        fake_core = FakeCore({"verdict": "allow"})
        store = ContextStore()
        adapter = LangGraphFrameworkAdapter(context_store=store)
        with installed_conformance_runtime(fake_core, adapter, store):
            with activity_scope(_CONTEXT, store=store):
                before_hits = server.hits
                before_payloads = len(fake_core.payloads)
                requests.get(server.url, timeout=5)
        # Exactly one hook-triggered evaluation for the ONE real request —
        # no recursive re-evaluation of the evaluate call itself.
        assert server.hits == before_hits + 1
        assert len(fake_core.payloads) == before_payloads + 2  # started + completed

    def test_wire_shape_activity_started_hook_trigger_nonempty_spans(self, server) -> None:
        fake_core = FakeCore({"verdict": "allow"})
        store = ContextStore()
        adapter = LangGraphFrameworkAdapter(context_store=store)
        with installed_conformance_runtime(fake_core, adapter, store):
            with activity_scope(_CONTEXT, store=store):
                requests.get(server.url, timeout=5)
        assert fake_core.started_payloads, "expected a started-stage hook payload"
        assert_hook_wire_shape(fake_core.started_payloads[0])
        assert fake_core.completed_payloads, "expected a completed-stage hook payload"
        assert_hook_wire_shape(fake_core.completed_payloads[0])

    def test_lifecycle_halt_raises_governance_halt_error(self) -> None:
        from openbox_core.contracts.results import EvaluationResult, Verdict

        adapter = LangGraphFrameworkAdapter()
        with pytest.raises(GovernanceHaltError) as exc_info:
            adapter.raise_lifecycle_blocked(EvaluationResult(verdict=Verdict.HALT, reason="stop"))
        assert str(exc_info.value) == "stop"

    def test_lifecycle_block_raises_governance_blocked_sdk_shape(self) -> None:
        from openbox_core.contracts.results import EvaluationResult, Verdict

        adapter = LangGraphFrameworkAdapter()
        with pytest.raises(GovernanceBlockedError) as exc_info:
            adapter.raise_lifecycle_blocked(
                EvaluationResult(verdict=Verdict.BLOCK, reason="denied")
            )
        assert exc_info.value.verdict == "block"
        assert str(exc_info.value) == "denied"
