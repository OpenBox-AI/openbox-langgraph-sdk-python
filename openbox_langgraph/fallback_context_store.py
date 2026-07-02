# openbox_langgraph/fallback_context_store.py
"""Wires ``TraceContextRegistry``'s fallback ladder into the base SDK's OWN
context-resolution seam, so a real hook evaluation benefits from it — not
just direct unit tests of the registry.

``openbox_core.hooks.events.resolve_context(store, span)`` is a free function
hardcoded to call ``store.current_activity_context()`` then
``store.context_for_trace(trace_id)`` on whatever object the runtime was
constructed with (duck-typed — never ``isinstance``-checked; see
``openbox_core.contracts.context.ActivityContextProvider``, which documents
this exact substitution but is not itself consumed anywhere as an injectable
seam). ``HookRuntime`` reads ``self._store = runtime.context_store`` and also
calls ``is_activity_aborted``/``mark_activity_aborted``/``request_halt`` on
it — methods OUTSIDE ``ActivityContextProvider`` — so only a ``ContextStore``
SUBCLASS (never a bare protocol-shaped object) can stand in here without
breaking abort/halt tracking.

``FallbackContextStore`` is exactly that subclass: it inherits every
``ContextStore`` method unchanged and overrides only ``context_for_trace`` to
fall through to a ``TraceContextRegistry``'s single-active/last-registered
tiers on an exact-trace miss — no base-SDK code changes required.
"""

from __future__ import annotations

from openbox_core.context import ContextStore
from openbox_core.contracts.context import ActivityContext

from openbox_langgraph.trace_context_registry import TraceContextRegistry

__all__ = ["FallbackContextStore"]


class FallbackContextStore(ContextStore):
    """``ContextStore`` whose ``context_for_trace`` also consults a
    ``TraceContextRegistry``'s fallback ladder on an exact-match miss.

    The registry is constructed bound to THIS store (not a separate one) so
    ``TraceContextRegistry.register``'s dual-write into
    ``ContextStore.register_trace`` and this override's ``super()`` exact
    lookup always agree — the exact-match tier is answered ONCE, by the base
    implementation; only a miss falls through to tiers 2/3.
    """

    def __init__(self) -> None:
        super().__init__()
        # Bound to `self` — `TraceContextRegistry.register()`'s dual-write
        # into `self.register_trace()` keeps the exact-match tier (handled by
        # `super().context_for_trace()` below) and the fallback tiers (handled
        # by `self.registry.resolve()`) reading from the SAME underlying map.
        self.registry = TraceContextRegistry(self)

    def context_for_trace(self, trace_id: int | str) -> ActivityContext | None:
        """Exact match first (base behavior, unchanged); on a miss, fall
        through to the registry's single-active/last-registered tiers.

        A registry-tier resolution is intentionally NOT re-registered as an
        exact match here — the fallback is a best-effort correlation for
        THIS lookup, not a promise that the guessed context is correct for
        every future trace_id (a later operation may legitimately resolve to
        a different single-active/last-registered context).
        """
        exact = super().context_for_trace(trace_id)
        if exact is not None:
            return exact
        return self.registry.resolve(trace_id)
