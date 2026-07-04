"""EXACT trace registration + per-turn abort bookkeeping over a private
base-SDK ``ContextStore``.

LangGraph spawns tool/LLM execution as ``asyncio.Task``s (and, for sync
tools, ``run_in_executor`` worker threads). The ContextVar tier bound at the
ToolNode seam (``tool_activity_binding``) reaches most of that spawned work,
but not code the SDK does not own the execution context of (e.g. a raw thread
a tool spawns). For those, this SDK registers the trace id of an OTel parent
span it EXPLICITLY creates for a known activity, so
``openbox_core.hooks.events.resolve_context``'s exact trace-id tier can resolve
that activity — a known trace id mapped to a known ``ActivityContext``.

This is EXACT registration ONLY. There is deliberately no single-active /
last-registered guessing: a hook span resolves to the activity it can be proven
to belong to (ContextVar tier, then exact trace tier), or it stays unbound.
This registry therefore keeps only what exact registration and per-turn cleanup
need — the trace-id -> context map (for ``sweep``) and the set of activity keys
ever registered this turn (for abort-mark cleanup).

Thread-safe: LangGraph sync tools run inside ``run_in_executor`` worker
threads that register concurrently with the async stream consumer.
"""

from __future__ import annotations

import logging
import threading
import weakref

from openbox_core.context import ContextStore, canonical_trace_key
from openbox_core.contracts.context import ActivityContext
from openbox_core.runtime import OpenBoxRuntime

_logger = logging.getLogger(__name__)

__all__ = [
    "TraceContextRegistry",
    "get_context_store",
    "get_trace_registry",
]


class TraceContextRegistry:
    """LangGraph-adapter EXACT trace registration over a private ``ContextStore``.

    Every ``register`` call dual-writes: the base ``ContextStore.register_trace``
    (so ``openbox_core.hooks.events.resolve_context``'s exact trace-id tier
    resolves it) AND a local ``OrderedDict`` this class owns, which ``sweep``
    uses to drop exactly this turn's bindings at turn-exit (the base store has
    no public per-turn enumeration of its trace map — by design, see
    ``context.py``).

    Thread-safe: LangGraph sync tools run inside ``run_in_executor`` worker
    threads that register concurrently with the async stream consumer.
    """

    def __init__(self, store: ContextStore) -> None:
        self._store = store
        self._lock = threading.Lock()
        # Trace keys registered this turn — the context VALUE lives in the base
        # ``ContextStore`` (exact resolution reads it there); this set exists
        # only so ``sweep`` knows which trace keys to unregister at turn-exit.
        self._by_trace: set[int] = set()
        # Every (workflow_id, activity_id) a `register()` call has EVER bound
        # in the current turn — kept independent of `_by_trace` (which loses
        # entries on `unregister`) so a completed activity's abort mark is
        # still swept even though its trace binding is long gone by turn-exit.
        self._activity_keys: set[tuple[str | None, str | None]] = set()

    @property
    def store(self) -> ContextStore:
        return self._store

    def register(self, trace_id: int | str, ctx: ActivityContext) -> None:
        """Register an EXACT trace binding: a known trace id -> a known activity.

        No ContextVar bind here by design: LangGraph tool/LLM execution runs
        in a spawned ``asyncio.Task`` or executor thread that already copied
        its ContextVar snapshot before this call can run, so a ContextVar bind
        at this call site can never reach that code — see module docstring.
        """
        key = canonical_trace_key(trace_id)
        self._store.register_trace(key, ctx)
        with self._lock:
            self._by_trace.add(key)
            self._activity_keys.add((ctx.workflow_id, ctx.activity_id))

    def unregister(self, trace_id: int | str) -> None:
        """Drop a single trace binding (activity completion)."""
        key = canonical_trace_key(trace_id)
        self._store.unregister_trace(key)
        with self._lock:
            self._by_trace.discard(key)

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
            keys = list(self._by_trace)
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

    def clear_aborted_for_workflow(self, workflow_id: str | None) -> None:
        """Clear the base ``ContextStore`` abort mark for every activity THIS
        registry has ever tracked under ``workflow_id`` — WITHOUT touching
        trace bindings or ``_activity_keys`` bookkeeping (unlike ``sweep``).

        Used by the opt-in hook runtime's post-approval reset (an approved
        REQUIRE_APPROVAL retry re-runs the SAME turn's ``workflow_id`` but
        bypasses this SDK's own dual-write registration entirely — it calls
        the underlying graph's ``ainvoke`` directly — so the retry's tool/LLM
        calls can only resolve context via the FIRST pass's still-registered
        trace bindings; dropping those here, as ``sweep`` would, leaves the
        retry ungoverned instead of governed).
        """
        with self._lock:
            keys = [k for k in self._activity_keys if k[0] == workflow_id]
        for wf_id, activity_id in keys:
            self._store.clear_activity_aborted(wf_id, activity_id)


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
