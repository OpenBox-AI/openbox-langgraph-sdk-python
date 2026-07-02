"""Regression: a source hook span's ActivityContext is pinned at STARTED and
reused at COMPLETED, so the trace-lookup fallback drifting to a later activity
cannot split one span across two activity_ids.

Drives `LangGraphHookRuntime` directly (no real HTTP): the store's trace map is
mutated between the started and completed stages to reproduce the exact drift
seen in Core logs — started under activity A, then the run moves on and the
trace resolves to activity B before the completed callback fires. The pin must
win, keeping both stages on A. `(trace_id, span_id, hook_type)` is the key, so
the same OTel span object is reused across both stages.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openbox_core")

from openbox_core.conformance.fake_core import FakeCore
from openbox_core.conformance.instrumentation import installed_conformance_runtime
from openbox_core.contracts.context import ActivityContext
from openbox_core.contracts.otel_spans import HookType
from opentelemetry import trace as otel_trace

from openbox_langgraph.core_adapter import LangGraphFrameworkAdapter
from openbox_langgraph.fallback_context_store import FallbackContextStore
from openbox_langgraph.langgraph_hook_runtime import LangGraphHookRuntime

_HOOK = HookType.HTTP_REQUEST


def _ctx(activity_id: str) -> ActivityContext:
    return ActivityContext(
        workflow_id="wf-pin",
        run_id="run-pin",
        workflow_type="PinWorkflow",
        task_queue="langgraph",
        activity_id=activity_id,
        activity_type="http_request",
    )


def _real_span(name: str):
    """A real OTel span (conftest installs a recording TracerProvider) so it
    carries non-zero trace_id/span_id for the pin key."""
    span = otel_trace.get_tracer("test-hook-pin").start_span(name)
    assert span.get_span_context().trace_id != 0, "test needs a real TracerProvider"
    return span


def test_completed_reuses_pinned_context_despite_store_drift() -> None:
    fake = FakeCore()  # empty queue → every verdict ALLOW
    store = FallbackContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    with installed_conformance_runtime(fake, adapter, store) as rt:
        hookrt = LangGraphHookRuntime(rt)
        span = _real_span("http_call")
        trace_id = span.get_span_context().trace_id

        store.register_trace(trace_id, _ctx("act-A"))  # STARTED resolves A
        hookrt.preflight(span, hook_type=_HOOK)

        store.register_trace(trace_id, _ctx("act-B"))  # DRIFT: trace now → B
        span.end()
        hookrt.completed(span, hook_type=_HOOK)

    assert fake.started_payloads and fake.completed_payloads
    assert fake.started_payloads[0]["activity_id"] == "act-A"
    # The pin — not the drifted store (now B) — decides the completed activity:
    assert fake.completed_payloads[0]["activity_id"] == "act-A"


def test_completed_without_started_pin_uses_resolver() -> None:
    """No prior preflight pin → completed falls back to normal resolution."""
    fake = FakeCore()
    store = FallbackContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    with installed_conformance_runtime(fake, adapter, store) as rt:
        hookrt = LangGraphHookRuntime(rt)
        span = _real_span("orphan_completed")
        store.register_trace(span.get_span_context().trace_id, _ctx("act-B"))
        span.end()
        hookrt.completed(span, hook_type=_HOOK)

    assert fake.completed_payloads[0]["activity_id"] == "act-B"


async def test_acompleted_reuses_pinned_context_despite_store_drift() -> None:
    fake = FakeCore()
    store = FallbackContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    with installed_conformance_runtime(fake, adapter, store) as rt:
        hookrt = LangGraphHookRuntime(rt)
        span = _real_span("http_call_async")
        trace_id = span.get_span_context().trace_id

        store.register_trace(trace_id, _ctx("act-A"))
        await hookrt.apreflight(span, hook_type=_HOOK)

        store.register_trace(trace_id, _ctx("act-B"))
        span.end()
        await hookrt.acompleted(span, hook_type=_HOOK)

    assert fake.completed_payloads[0]["activity_id"] == "act-A"
