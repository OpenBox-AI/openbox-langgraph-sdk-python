"""Base-hook context lookup under direct-await, `asyncio.create_task`, and
`run_in_executor` — proving the trace-only dual-write (no ContextVar bind)
is sufficient for the base SDK's OWN exact-trace-id resolution tier to find
governance context regardless of which of the three LangGraph execution
shapes a tool/LLM call runs under.

Uses the REAL conformance kit (`openbox_core.conformance`) so the assertions
exercise ACTUAL installed httpx/requests instrumentation and the base SDK's
OWN `resolve_context`/`build_hook_event` — not a re-implementation of them.

Why no ContextVar bind is exercised here: `openbox_core.context.ContextStore.bind`
sets a ContextVar that `asyncio.create_task`/`run_in_executor` COPY at spawn
time (proven in `test_contextvars_propagation.py`) — a bind made on the
stream-consumer coroutine can never reach code running inside a task spawned
either before OR after that bind, once the task is already scheduled.
`langgraph_handler.py`'s dual-write therefore never calls `ContextStore.bind`
for LangGraph; it registers ONLY the trace-id -> context mapping, which is
looked up by the OTel trace id carried on the ACTUAL span the operation runs
under — a mechanism OTel's own context propagation (not Python's raw
ContextVar copy semantics) keeps correct across task/thread boundaries,
because the span is created (and its OTel context attached) BEFORE the task
is spawned, exactly like `langgraph_handler.py`'s `on_tool_start`/
`on_chat_model_start` sites do.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace

pytest.importorskip("openbox_core")

from openbox_core.conformance.fake_core import FakeCore
from openbox_core.conformance.instrumentation import (
    LocalCountingServer,
    installed_conformance_runtime,
)
from openbox_core.context import ContextStore
from openbox_core.contracts.context import ActivityContext

from openbox_langgraph.core_runtime import TraceContextRegistry

_TRACER = otel_trace.get_tracer("openbox-langgraph-fallback-test")

_REFERENCE_CONTEXT = ActivityContext(
    workflow_id="wf-fallback",
    run_id="run-fallback",
    workflow_type="FallbackTestWorkflow",
    task_queue="fallback-queue",
    activity_id="act-fallback",
    activity_type="fallback_activity",
)


@pytest.fixture(scope="module")
def server():
    srv = LocalCountingServer()
    yield srv
    srv.stop()


def _register_and_start_span(registry: TraceContextRegistry, name: str) -> tuple[int, object]:
    """Create a REAL OTel span, register its trace id (trace-only, no bind),
    attach the span to the current OTel context, and return (trace_id, token)
    so the caller can run an operation "under" this span and detach after."""
    span = _TRACER.start_span(name, kind=otel_trace.SpanKind.INTERNAL)
    token = otel_context.attach(otel_trace.set_span_in_context(span))
    trace_id = span.get_span_context().trace_id
    registry.register(trace_id, _REFERENCE_CONTEXT)
    return trace_id, (span, token)


def _end_span(handle: object) -> None:
    span, token = handle  # type: ignore[misc]
    otel_context.detach(token)
    span.end()


@pytest.mark.asyncio
async def test_direct_await_resolves_context_via_registered_trace(server) -> None:
    """Baseline: an operation directly awaited under the registered span's
    OTel context resolves — ContextVar propagation trivially works here too,
    so this proves the exact-trace tier ALONE (no ContextVar assist needed)
    already succeeds, before the harder create_task/executor cases below."""
    fake_core = FakeCore({"verdict": "allow"})
    store = ContextStore()
    registry = TraceContextRegistry(store)

    with installed_conformance_runtime(fake_core, store=store):
        _trace_id, handle = _register_and_start_span(registry, "direct-await-op")
        try:
            before = server.hits
            response = await asyncio.to_thread(requests.get, server.url, timeout=5)
            assert response.status_code == 200
            assert server.hits == before + 1
        finally:
            _end_span(handle)

    assert fake_core.started_payloads, "expected a hook-triggered ActivityStarted payload"
    payload = fake_core.started_payloads[0]
    assert payload["activity_id"] == _REFERENCE_CONTEXT.activity_id
    assert payload["workflow_id"] == _REFERENCE_CONTEXT.workflow_id


@pytest.mark.asyncio
async def test_create_task_resolves_context_via_registered_trace_not_contextvar(
    server,
) -> None:
    """The exact scenario `langgraph_handler.py` hits: a tool's HTTP call runs
    inside an `asyncio.create_task` spawned AFTER the span/trace is
    registered. `ContextStore.current_activity_context()` (the ContextVar
    tier) is proven empty here — nothing ever called `.bind()` — so this ONLY
    passes if the exact-trace-id tier (fed by `TraceContextRegistry.register`)
    is what resolves it, not an incidental ContextVar carry-through."""
    fake_core = FakeCore({"verdict": "allow"})
    store = ContextStore()
    registry = TraceContextRegistry(store)

    with installed_conformance_runtime(fake_core, store=store):
        _trace_id, handle = _register_and_start_span(registry, "create-task-op")
        try:
            assert store.current_activity_context() is None, (
                "ContextVar tier must be empty — this test proves the "
                "trace-id tier alone resolves context, not a ContextVar bind"
            )

            async def tool_task() -> requests.Response:
                return await asyncio.to_thread(requests.get, server.url, timeout=5)

            before = server.hits
            response = await asyncio.create_task(tool_task())
            assert response.status_code == 200
            assert server.hits == before + 1
        finally:
            _end_span(handle)

    assert fake_core.started_payloads
    payload = fake_core.started_payloads[0]
    assert payload["activity_id"] == _REFERENCE_CONTEXT.activity_id


@pytest.mark.asyncio
async def test_run_in_executor_loses_otel_context_so_base_lookup_misses(server) -> None:
    """Empirical prerequisite for the NEXT test: `loop.run_in_executor` does
    NOT propagate the submitting coroutine's `contextvars.Context` (verified
    directly against `contextvars` — unlike `asyncio.create_task`, which DOES
    copy context automatically per the asyncio API contract). OTel's own
    current-span tracking is itself ContextVar-based, so the HTTP call
    running inside the executor thread gets a BRAND NEW, unrelated trace_id —
    `opentelemetry-instrumentation-requests` starts a fresh ROOT span because
    it sees no parent context, not a child of the span registered before the
    executor call was submitted. The base SDK's exact-trace-id tier therefore
    CANNOT match here by construction: this is exactly why the legacy
    `WorkflowSpanProcessor` (and this adapter's `TraceContextRegistry`) carry
    a single-active/last-registered fallback tier — see the next test."""
    fake_core = FakeCore({"verdict": "allow"})
    store = ContextStore()
    registry = TraceContextRegistry(store)

    with installed_conformance_runtime(fake_core, store=store):
        _trace_id, handle = _register_and_start_span(registry, "executor-op")
        try:
            assert store.current_activity_context() is None

            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as pool:
                before_hits = server.hits
                before_payloads = len(fake_core.payloads)
                response = await loop.run_in_executor(pool, requests.get, server.url)
                assert response.status_code == 200
                assert server.hits == before_hits + 1
                # The base SDK's exact-trace tier misses (unrelated trace_id
                # from a fresh root span) -> silent skip, matching
                # `resolve_context`'s documented "no bound context = not
                # governed" behavior. No payload sent for this operation.
                assert len(fake_core.payloads) == before_payloads
        finally:
            _end_span(handle)


def test_run_in_executor_resolves_via_single_active_fallback_tier() -> None:
    """The adapter-level shim's single-active fallback tier is what actually
    covers the `run_in_executor` gap the previous test proved exists: with
    EXACTLY ONE activity registered on this registry, `resolve()` returns it
    for ANY trace_id — including the executor thread's unrelated fresh one —
    mirroring `WorkflowSpanProcessor.get_activity_context_by_trace`'s own
    "exactly one active context" tier. No conformance runtime/server needed:
    this is a direct, synchronous unit test of `TraceContextRegistry.resolve`."""
    store = ContextStore()
    registry = TraceContextRegistry(store)
    trace_id, handle = _register_and_start_span(registry, "executor-op-2")
    unrelated_trace_id = trace_id + 1  # guaranteed not to exact-match

    try:
        resolved = registry.resolve(unrelated_trace_id)
    finally:
        _end_span(handle)

    assert resolved is not None
    assert resolved.workflow_id == _REFERENCE_CONTEXT.workflow_id
    assert resolved.activity_id == _REFERENCE_CONTEXT.activity_id
    assert registry.metrics.miss_count == 0, "single-active fallback must not count as a miss"


@pytest.mark.asyncio
async def test_context_miss_fails_loud_not_silent(server, caplog) -> None:
    """No trace was ever registered for this span — the base SDK's OWN
    `build_hook_event` skips SILENTLY (DEBUG-level log, no hook fires, the
    HTTP call still runs ungoverned — that miss is the pre-existing base-SDK
    behavior this phase does not change). `TraceContextRegistry.resolve`,
    called directly against the SAME unpopulated registry, must instead be
    LOUD: a WARNING-level log line plus an incremented `ContextMissMetrics`
    counter, precisely because an ungoverned operation under
    `use_core_instrumentation=True` must be observable, never silent."""
    fake_core = FakeCore({"verdict": "allow"})
    store = ContextStore()
    registry = TraceContextRegistry(store)

    with installed_conformance_runtime(fake_core, store=store):
        span = _TRACER.start_span("unregistered-op", kind=otel_trace.SpanKind.INTERNAL)
        token = otel_context.attach(otel_trace.set_span_in_context(span))
        trace_id = span.get_span_context().trace_id
        try:
            before_hits = server.hits
            before_payloads = len(fake_core.payloads)

            # Base SDK path: silent skip. The HTTP call still succeeds (fail-open
            # default for an unresolved hook is "let it run"), but NO governance
            # payload is ever sent for it — this is the pre-existing behavior.
            response = await asyncio.to_thread(requests.get, server.url, timeout=5)
            assert response.status_code == 200
            assert server.hits == before_hits + 1
            assert len(fake_core.payloads) == before_payloads, (
                "no hook payload should have been sent — nothing was registered"
            )
        finally:
            otel_context.detach(token)
            span.end()

    # Adapter-level shim path: the SAME miss, but fail-loud.
    with caplog.at_level("WARNING", logger="openbox_langgraph.core_runtime"):
        resolved = registry.resolve(trace_id)

    assert resolved is None
    assert registry.metrics.miss_count == 1
    assert registry.metrics.last_miss_trace_id == trace_id
    assert any(
        "no ActivityContext resolved" in record.message for record in caplog.records
    ), "expected a WARNING-level log line for the unresolved trace"
