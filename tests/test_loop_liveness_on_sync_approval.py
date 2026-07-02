"""A sync-hook REQUIRE_APPROVAL must never freeze the event loop.

`LangGraphFrameworkAdapter.handle_approval_sync` is RAISE-ONLY (no inline
blocking wait). Defining it pre-empts
`openbox_core.hooks.preflight.HookRuntime._sync_approval`'s fallback to its
own inline `ApprovalPoller.wait_for_decision` — a `time.sleep`-based loop
with a 5-SECOND default `poll_interval_ms` (`openbox_core.config.HitlConfig`).
If that fallback ever ran on the event loop thread, EVERY concurrent
coroutine sharing that loop (LangGraph's tool/LLM tasks are exactly that)
would stall for the full sleep duration before the hook even asks Core once.

Proves it two ways:
1. A concurrent tick-counter task keeps advancing DURING the sync-hook
   evaluation, and the whole operation completes in well under one second —
   an inline poller's first sleep alone would take 5 seconds, an unmistakable
   signal if the raise-only contract ever regressed.
2. Directly: NO approval-poll request ever reaches Core. This is the
   AIRTIGHT check — verified empirically that the fallback poller's FIRST
   attempt can resolve terminal-ALLOW-shaped immediately against
   `FakeCore`'s single-queued-response default (an empty queue answers ALLOW
   on any subsequent call), which would make a regressed adapter ALSO finish
   near-instantly and pass check #1 by luck. Check #2 has no such blind spot:
   the fallback poller sends at least one `/governance/approval` POST before
   it can ever decide anything, regardless of how fast that decision resolves.
"""

from __future__ import annotations

import asyncio
import time

import pytest
import requests

pytest.importorskip("openbox_core")

from openbox_core.conformance.fake_core import FakeCore
from openbox_core.conformance.instrumentation import (
    LocalCountingServer,
    installed_conformance_runtime,
)
from openbox_core.context import ContextStore, activity_scope
from openbox_core.contracts.context import ActivityContext

from openbox_langgraph.core_adapter import LangGraphFrameworkAdapter
from openbox_langgraph.errors import GovernanceBlockedError

_CONTEXT = ActivityContext(
    workflow_id="wf-liveness",
    run_id="run-liveness",
    activity_id="act-liveness",
    activity_type="http_request",
)

# Generous upper bound: the raise-only path should complete in low
# milliseconds; an inline 5s-poll-interval fallback blows past this by 10x.
_MAX_EXPECTED_SECONDS = 0.5
# Minimum ticks a healthy loop accumulates while the sync call runs — a
# frozen loop produces (near) zero regardless of how long the freeze lasts,
# since ticking resumes only after the freeze ends.
_MIN_EXPECTED_TICKS = 3


@pytest.fixture
def server():
    srv = LocalCountingServer()
    yield srv
    srv.stop()


async def test_concurrent_task_keeps_ticking_during_sync_hook_approval(server) -> None:
    fake_core = FakeCore({"verdict": "require_approval", "approval_id": "app-1"})
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)

    tick_count = 0
    stop = asyncio.Event()

    async def ticker() -> None:
        nonlocal tick_count
        while not stop.is_set():
            tick_count += 1
            await asyncio.sleep(0.01)

    with installed_conformance_runtime(fake_core, adapter, store):
        with activity_scope(_CONTEXT, store=store):
            ticker_task = asyncio.create_task(ticker())
            await asyncio.sleep(0.03)  # let the ticker establish a baseline rate
            ticks_before = tick_count

            started_at = time.monotonic()
            with pytest.raises(GovernanceBlockedError) as exc_info:
                # A SYNC requests call made directly on the loop thread (no
                # executor offload) — the realistic worst case for loop
                # freezing, since nothing here yields control to other tasks
                # except whatever the hook itself does.
                requests.get(server.url, timeout=5)
            elapsed = time.monotonic() - started_at

            await asyncio.sleep(0.03)  # give the ticker a chance to resume/continue
            stop.set()
            await ticker_task

    assert exc_info.value.verdict == "require_approval"
    assert elapsed < _MAX_EXPECTED_SECONDS, (
        f"sync approval took {elapsed:.3f}s — expected near-instant (raise-only); "
        "an inline poller fallback would take >= 5s (default poll_interval_ms)"
    )
    ticks_after = tick_count
    assert ticks_after - ticks_before >= _MIN_EXPECTED_TICKS, (
        f"only {ticks_after - ticks_before} ticks around the sync approval — "
        "the loop may have been blocked"
    )
    # The blocked request never reached the server.
    assert server.hits == 0


async def test_sync_approval_does_not_send_a_poll_request_to_core(server) -> None:
    """Direct evidence the inline poller never ran: NO approval-poll request
    ever reached the fake Core — only the initial evaluate call did."""
    fake_core = FakeCore({"verdict": "require_approval", "approval_id": "app-2"})
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)

    with installed_conformance_runtime(fake_core, adapter, store):
        with activity_scope(_CONTEXT, store=store):
            with pytest.raises(GovernanceBlockedError):
                requests.get(server.url, timeout=5)

    assert fake_core.approval_requests == []
    assert len(fake_core.started_payloads) == 1
