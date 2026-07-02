"""Trace-lookup fallback shim: LangGraph-adapter bookkeeping over a private
base-SDK ``ContextStore``, plus the fail-loud miss-observability surface.

LangGraph spawns tool/LLM execution as ``asyncio.Task``s (and, for sync
tools, ``run_in_executor`` worker threads). Both copy/lose the current
``ContextVar`` at spawn time, so a context bound on the stream-consumer
coroutine never reaches code running inside the spawned task — proven
empirically in ``tests/test_contextvars_propagation.py``. The base SDK's own
``openbox_core.hooks.events.resolve_context`` only tries the ContextVar, then
one exact trace-id lookup; on a miss it skips the hook SILENTLY (correct for
the base SDK's framework-agnostic default, but LangGraph's task-spawn pattern
makes that miss the common case, not the exception).

``TraceContextRegistry.resolve`` mirrors the legacy
``WorkflowSpanProcessor.get_activity_context_by_trace`` fallback ladder
(exact-trace -> single-active -> last-registered) over the SAME private
``ContextStore`` the dual-write in ``langgraph_handler.py`` populates —
without reaching into ``ContextStore`` internals, since the base store
deliberately does not expose enumeration/order of its trace map. The
bookkeeping needed for tiers 2/3 lives here, on the LangGraph adapter side,
exactly where the legacy processor keeps its own ``_activity_context``/
``_last_activity_key`` bookkeeping today.

A resolution miss is NEVER a silent skip: it is logged at WARNING (visible in
default log configs, unlike the base SDK's DEBUG) and counted in
``ContextMissMetrics`` so callers/tests can assert on it. An ungoverned
operation under ``use_core_instrumentation=True`` must be observable.
"""

from __future__ import annotations

import logging
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass

from openbox_core.context import ContextStore, canonical_trace_key
from openbox_core.contracts.context import ActivityContext
from openbox_core.runtime import OpenBoxRuntime

_logger = logging.getLogger(__name__)

__all__ = [
    "ContextMissMetrics",
    "TraceContextRegistry",
    "get_context_store",
    "get_trace_registry",
]


@dataclass
class ContextMissMetrics:
    """Observable counters for trace-lookup misses — the fail-loud surface.

    Incremented by :meth:`TraceContextRegistry.resolve` whenever every
    fallback tier is exhausted with no registered context. Tests assert on
    ``miss_count``; a real deployment would export it as a metric alongside
    the WARNING log line.
    """

    miss_count: int = 0
    last_miss_trace_id: int | None = None

    def record_miss(self, trace_id: int) -> None:
        self.miss_count += 1
        self.last_miss_trace_id = trace_id


class TraceContextRegistry:
    """LangGraph-adapter trace bookkeeping layered on top of a private ``ContextStore``.

    Every ``register`` call dual-writes: the base ``ContextStore.register_trace``
    (so ``openbox_core.hooks.events.resolve_context``'s own exact trace-id tier
    keeps working unchanged) AND a local ``OrderedDict`` this class owns, which
    is what makes the single-active / last-registered fallback tiers possible
    without reaching into ``ContextStore`` internals (it has no public API to
    enumerate or order its trace map — by design, see ``context.py``).

    Thread-safe: LangGraph sync tools run inside ``run_in_executor`` worker
    threads that register/resolve concurrently with the async stream consumer.
    """

    def __init__(self, store: ContextStore) -> None:
        self._store = store
        self._lock = threading.Lock()
        self._by_trace: OrderedDict[int, ActivityContext] = OrderedDict()
        # Every (workflow_id, activity_id) a `register()` call has EVER bound
        # in the current turn — kept independent of `_by_trace` (which loses
        # entries on `unregister`) so a completed activity's abort mark is
        # still swept even though its trace binding is long gone by turn-exit.
        self._activity_keys: set[tuple[str | None, str | None]] = set()
        self.metrics = ContextMissMetrics()

    @property
    def store(self) -> ContextStore:
        return self._store

    def register(self, trace_id: int | str, ctx: ActivityContext) -> None:
        """Register a trace-only binding (mirrors the legacy span processor).

        No ContextVar bind here by design: LangGraph tool/LLM execution runs
        in a spawned ``asyncio.Task`` or executor thread that already copied
        its ContextVar snapshot before this call can run, so a ContextVar bind
        at this call site can never reach that code — see module docstring.
        """
        key = canonical_trace_key(trace_id)
        self._store.register_trace(key, ctx)
        with self._lock:
            # Re-insert to move to the end — "most recently registered" for
            # the last-registered fallback tier stays accurate on re-registration.
            self._by_trace.pop(key, None)
            self._by_trace[key] = ctx
            self._activity_keys.add((ctx.workflow_id, ctx.activity_id))

    def unregister(self, trace_id: int | str) -> None:
        """Drop a single trace binding (activity completion)."""
        key = canonical_trace_key(trace_id)
        self._store.unregister_trace(key)
        with self._lock:
            self._by_trace.pop(key, None)

    def sweep(self, workflow_id: str | None = None) -> None:
        """Drop ALL bindings + turn-scoped bookkeeping (turn-exit cleanup).

        Only clears the trace map and abort marks THIS registry tracked —
        never ``store.clear()``, which would also drop the runtime's
        ``halt_requested`` flag and any other turn's abort marks, both of
        which belong to the runtime's full lifetime, not a single turn.

        When ``workflow_id`` is given, also clears every abort mark this
        registry ever registered an activity under for that turn (see
        ``_activity_keys``) — the base ``ContextStore`` has no prefix-sweep
        of its own (unlike the legacy ``WorkflowSpanProcessor.unregister_workflow``),
        so this registry supplies the missing per-turn sweep instead.
        """
        with self._lock:
            keys = list(self._by_trace.keys())
            self._by_trace.clear()
            activity_keys = (
                [k for k in self._activity_keys if k[0] == workflow_id]
                if workflow_id is not None
                else list(self._activity_keys)
            )
            for k in activity_keys:
                self._activity_keys.discard(k)
        for key in keys:
            self._store.unregister_trace(key)
        for wf_id, activity_id in activity_keys:
            self._store.clear_activity_aborted(wf_id, activity_id)

    def resolve(self, trace_id: int | str) -> ActivityContext | None:
        """Exact-trace -> single-active -> last-registered, fail-loud on total miss.

        Mirrors ``WorkflowSpanProcessor.get_activity_context_by_trace``'s
        fallback ladder against this registry's own bookkeeping (see class
        docstring for why the base ``ContextStore`` cannot supply tiers 2/3).
        A miss across every tier is logged at WARNING and counted in
        ``self.metrics`` — never a silent skip, since a governed operation
        that resolves no context runs ungoverned.
        """
        key = canonical_trace_key(trace_id)
        with self._lock:
            exact = self._by_trace.get(key)
            if exact is not None:
                return exact
            if len(self._by_trace) == 1:
                return next(iter(self._by_trace.values()))
            if self._by_trace:
                return next(reversed(self._by_trace.values()))
        _logger.warning(
            "openbox_langgraph.trace_context_registry: no ActivityContext resolved "
            "for trace_id=%s across exact/single-active/last-registered fallback "
            "tiers — this operation is running WITHOUT core-instrumentation "
            "governance context",
            key,
        )
        self.metrics.record_miss(key)
        return None


# Runtime -> registry, so `langgraph_handler.py` can dual-write via
# `get_trace_registry(runtime)` without the handler owning registry lifetime
# bookkeeping itself. A `WeakKeyDictionary` — NOT a plain `dict[id(runtime), ...]`
# — is required here: CPython reuses a garbage-collected object's `id()` for a
# later, unrelated object of similar size, so an `id()`-keyed cache can hand a
# brand-new runtime the STALE registry (wrapping a DEAD runtime's dead
# ContextStore) of whichever earlier runtime happened to get GC'd into the
# same address. Caught empirically: short-lived runtimes created back-to-back
# (e.g. one per test) intermittently resolved a PRIOR runtime's registry.
# `WeakKeyDictionary` keys on actual object identity via a weak-reference
# callback, immune to address reuse, and self-prunes when a runtime is GC'd —
# no explicit cleanup call needed from `OpenBoxLangGraphHandler`.
_registries: weakref.WeakKeyDictionary[OpenBoxRuntime, TraceContextRegistry] = (
    weakref.WeakKeyDictionary()
)
_registries_lock = threading.Lock()


def get_context_store(runtime: OpenBoxRuntime) -> ContextStore:
    """The runtime's PRIVATE ``ContextStore`` — never the process-global default.

    Thin accessor so callers outside this module never need to reach past the
    runtime's public ``context_store`` attribute directly; kept as a function
    (not a re-export) so future runtimes that resolve the store lazily do not
    require a call-site change.
    """
    return runtime.context_store


def get_trace_registry(runtime: OpenBoxRuntime) -> TraceContextRegistry:
    """The (lazily created) ``TraceContextRegistry`` for ``runtime``'s private store.

    One registry per runtime, created on first use and reused for the
    runtime's lifetime — matches the one-runtime-per-handler, one-private-store
    -per-runtime isolation guarantee ``core_runtime.create_core_runtime``
    provides. Keyed on the runtime OBJECT itself (weakly) — see the
    `_registries` module comment for why an `id()`-keyed cache is unsafe here.
    """
    with _registries_lock:
        registry = _registries.get(runtime)
        if registry is None:
            registry = TraceContextRegistry(runtime.context_store)
            _registries[runtime] = registry
        return registry
