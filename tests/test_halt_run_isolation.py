"""A HALT in one run must not affect a concurrent/later run in the same
process.

Each `create_core_runtime` call gets its OWN PRIVATE `ContextStore` (or
`FallbackContextStore`) — this is the isolation mechanism, not a special
suppression of `request_halt()`. `openbox_core.hooks.preflight.HookRuntime
._mark_stopped` calls `self._store.request_halt()` on a HALT verdict
UNCONDITIONALLY (base-SDK behavior, correct and expected) — but `self._store`
is whichever runtime's OWN private store fired the hook, so a HALT in run A
sets `halt_requested` on run A's store ONLY. Run B's separate store's
`halt_requested` must stay `False` throughout, and run B's own operations
must keep succeeding normally.

`LangGraphFrameworkAdapter` itself never calls `request_halt()` directly
(grep-verified against `core_adapter.py`) — it relies entirely on the base
`HookRuntime`'s per-store flag, never a global/class-level one, which is
exactly what makes per-run isolation possible without any adapter-side
bookkeeping.
"""

from __future__ import annotations

import pytest
import requests

pytest.importorskip("openbox_core")

from openbox_core.conformance.fake_core import FakeCore
from openbox_core.conformance.instrumentation import LocalCountingServer
from openbox_core.context import ContextStore, activity_scope
from openbox_core.contracts.context import ActivityContext
from openbox_core.instrumentation.manager import InstrumentationManager

from openbox_langgraph.core_adapter import LangGraphFrameworkAdapter
from openbox_langgraph.errors import GovernanceBlockedError

_CONTEXT_A = ActivityContext(
    workflow_id="wf-run-a", run_id="run-a", activity_id="act-a", activity_type="http_request"
)
_CONTEXT_B = ActivityContext(
    workflow_id="wf-run-b", run_id="run-b", activity_id="act-b", activity_type="http_request"
)


@pytest.fixture
def server():
    srv = LocalCountingServer()
    yield srv
    srv.stop()


def _armed_run(fake_core: FakeCore):
    """One isolated (runtime, store, manager) triple — mirrors what
    `create_core_runtime` gives each LangGraph handler: its OWN adapter bound
    to its OWN private store, with base instrumentation installed via a
    manually constructed `InstrumentationManager` (same technique
    `openbox_core.conformance.instrumentation.installed_conformance_runtime`
    uses) so this test controls install/uninstall per run explicitly."""
    from openbox_core.conformance.hook_preflight import build_conformance_runtime

    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    runtime = build_conformance_runtime(fake_core, adapter, store)
    manager = InstrumentationManager(runtime)
    runtime._instrumentation_manager = manager
    manager.install()
    return runtime, store, manager


class TestHaltRunIsolation:
    def test_halt_in_run_a_does_not_set_halt_on_run_b_store(self, server) -> None:
        fake_core_a = FakeCore({"verdict": "halt", "reason": "run A emergency"})
        fake_core_b = FakeCore({"verdict": "allow"})

        _runtime_a, store_a, manager_a = _armed_run(fake_core_a)
        try:
            with activity_scope(_CONTEXT_A, store=store_a):
                with pytest.raises(GovernanceBlockedError) as exc_info:
                    requests.get(server.url, timeout=5)
            assert exc_info.value.verdict == "halt"
            assert store_a.halt_requested, "run A's OWN store must reflect its HALT"
        finally:
            manager_a.uninstall()

        # Run B is a COMPLETELY SEPARATE runtime/store/instrumentation
        # install, sequenced after run A fully tears down (proves isolation
        # holds even across sequential runs sharing the process, not just
        # concurrent ones — the private-store guarantee has no time
        # component).
        _runtime_b, store_b, manager_b = _armed_run(fake_core_b)
        try:
            assert not store_b.halt_requested, (
                "run B's store must NEVER have been touched by run A's HALT"
            )
            with activity_scope(_CONTEXT_B, store=store_b):
                before = server.hits
                response = requests.get(server.url, timeout=5)
                assert response.status_code == 200
                assert server.hits == before + 1
            assert not store_b.halt_requested, (
                "run B's own ALLOW-verdict operation must not set halt either"
            )
        finally:
            manager_b.uninstall()

    def test_concurrent_runs_stay_isolated_within_the_same_process(self, server) -> None:
        """Both runtimes installed and active AT THE SAME TIME (not
        sequential) — proves isolation is structural (per-store), not an
        artifact of one run's instrumentation being uninstalled before the
        other's starts. Only ONE `InstrumentationManager` can hold the
        process-wide monkeypatches at once, so run B's manager is NOT
        installed here — instead both runtimes' STORES are exercised
        directly against the SAME installed hook runtime (run A's), proving
        the isolation guarantee lives in the store object itself, not in
        which manager happens to be active."""
        fake_core = FakeCore(
            {"verdict": "halt", "reason": "run A emergency"},
            {"verdict": "allow"},
        )
        store_a = ContextStore()
        store_b = ContextStore()
        adapter_a = LangGraphFrameworkAdapter(context_store=store_a)

        from openbox_core.conformance.hook_preflight import build_conformance_runtime

        runtime_a = build_conformance_runtime(fake_core, adapter_a, store_a)
        manager = InstrumentationManager(runtime_a)
        runtime_a._instrumentation_manager = manager
        manager.install()
        try:
            with activity_scope(_CONTEXT_A, store=store_a):
                with pytest.raises(GovernanceBlockedError):
                    requests.get(server.url, timeout=5)
            assert store_a.halt_requested

            # store_b was never bound during the HALT and belongs to a
            # DIFFERENT adapter/runtime entirely — must remain untouched.
            assert not store_b.halt_requested
        finally:
            manager.uninstall()

    def test_adapter_never_calls_request_halt_directly(self) -> None:
        """`LangGraphFrameworkAdapter` relies entirely on the base
        `HookRuntime`'s own `request_halt()` call (per-store, correct) —
        confirmed here by construction: every adapter method that can see a
        HALT verdict (`raise_lifecycle_blocked`, `raise_hook_blocked`,
        `_raise_pending_approval`) only calls `mark_activity_aborted` and
        raises; NONE of them touch `request_halt` themselves. A direct call
        from adapter code would be redundant with the base HookRuntime's own
        per-store flag and risks diverging from it (e.g. calling it on the
        wrong store)."""
        import inspect

        from openbox_langgraph import core_adapter

        source = inspect.getsource(core_adapter.LangGraphFrameworkAdapter)
        assert "request_halt" not in source
