"""Base-hook context lookup under direct-await, `asyncio.create_task`, and
`run_in_executor` — proving EXACT trace registration resolves governance
context where OTel context propagates, and that where it does NOT propagate
(and no ContextVar is bound) the operation is left intentionally UNBOUND, never
guessed onto some other active activity.

Uses the REAL conformance kit (`openbox_core.conformance`) so the assertions
exercise ACTUAL installed httpx/requests instrumentation and the base SDK's
OWN `resolve_context`/`build_hook_event` — not a re-implementation of them.

This SDK registers ONLY exact bindings: the trace id of an OTel span it owns,
mapped to a known `ActivityContext`. There is no single-active/last-registered
fallback tier — a hook span resolves to the activity it can be proven to belong
to (the ContextVar tier bound at the ToolNode seam, then this exact trace tier),
or it stays unbound. The `run_in_executor` case below is exactly where a
guessing fallback used to fire; it now correctly resolves nothing rather than
mis-attributing the call to the one registered activity.
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

_TRACER = otel_trace.get_tracer("openbox-langgraph-exact-trace-test")

_REFERENCE_CONTEXT = ActivityContext(
    workflow_id="wf-exact",
    run_id="run-exact",
    workflow_type="ExactTraceWorkflow",
    task_queue="exact-queue",
    activity_id="act-exact",
    activity_type="exact_activity",
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
    OTel context resolves via the EXACT trace-id tier."""
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
async def test_create_task_resolves_context_via_registered_trace(server) -> None:
    """A tool's HTTP call inside an `asyncio.create_task` spawned AFTER the
    span/trace is registered. `create_task` copies the OTel context (itself
    ContextVar-based), so the EXACT trace-id tier resolves it — not an
    incidental ContextVar activity bind (proven empty below)."""
    fake_core = FakeCore({"verdict": "allow"})
    store = ContextStore()
    registry = TraceContextRegistry(store)

    with installed_conformance_runtime(fake_core, store=store):
        _trace_id, handle = _register_and_start_span(registry, "create-task-op")
        try:
            assert store.current_activity_context() is None, (
                "ContextVar tier must be empty — this test proves the "
                "exact trace-id tier resolves context, not a ContextVar bind"
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
async def test_run_in_executor_without_bind_stays_unbound_never_guessed(server) -> None:
    """`loop.run_in_executor` does NOT propagate the submitting coroutine's
    `contextvars.Context` (unlike `asyncio.create_task`). OTel's own
    current-span tracking is ContextVar-based, so the HTTP call in the executor
    thread starts a fresh ROOT span with an unrelated trace_id — the exact
    trace-id tier cannot match it.

    With EXACTLY ONE activity registered, a single-active fallback tier (removed)
    would have mis-attributed this call to `act-exact`. It must instead resolve
    NOTHING: no governance payload is sent, the call runs unbound. This is the
    zero-fallback guarantee — proven, not guessed."""
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
                # No exact trace match + no ContextVar bind → NO payload, even
                # though exactly one activity is registered. Never guessed.
                assert len(fake_core.payloads) == before_payloads
        finally:
            _end_span(handle)


@pytest.mark.asyncio
async def test_unregistered_trace_resolves_nothing(server) -> None:
    """No trace was ever registered for this span — the base SDK's own
    `build_hook_event` resolves no context and skips the hook (the HTTP call
    still runs, ungoverned — the base SDK's fail-open default for an unresolved
    hook). No governance payload is sent. There is no louder adapter-side
    fallback that would resolve or count this differently: unbound is unbound."""
    fake_core = FakeCore({"verdict": "allow"})
    store = ContextStore()

    with installed_conformance_runtime(fake_core, store=store):
        span = _TRACER.start_span("unregistered-op", kind=otel_trace.SpanKind.INTERNAL)
        token = otel_context.attach(otel_trace.set_span_in_context(span))
        try:
            before_hits = server.hits
            before_payloads = len(fake_core.payloads)
            response = await asyncio.to_thread(requests.get, server.url, timeout=5)
            assert response.status_code == 200
            assert server.hits == before_hits + 1
            assert len(fake_core.payloads) == before_payloads, (
                "no hook payload should have been sent — nothing was registered"
            )
        finally:
            otel_context.detach(token)
            span.end()
