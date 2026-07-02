"""LangGraph-owned hook runtime that pins a source span's ActivityContext.

Why this exists — the drift bug it fixes:

The base :class:`~openbox_core.hooks.preflight.HookRuntime` resolves the bound
``ActivityContext`` INDEPENDENTLY at the started (preflight) and completed
stages, via ``resolve_context`` → ``store.context_for_trace(trace_id)``. In
Temporal that is stable because a `core_activity_scope` binds the context around
the actual activity execution. LangGraph has no such scope: it reconstructs
context from ``astream_events`` + a trace-lookup fallback ladder
(:class:`~openbox_langgraph.trace_context_registry.TraceContextRegistry`:
exact → single-active → last-registered). That ladder resolves to whatever
activity is *currently* active — so a hook span that STARTED while ``load_skill``
was active can COMPLETE after the run has moved on to ``llm_call``, and the
completed stage is mis-attributed to ``llm_call``. Core then sees one source
span split across two ``activity_id``s.

Fix — and the boundary it respects: this runtime does NOT re-implement the hook
path. It only supplies the RIGHT context. It pins the context resolved at
STARTED, keyed by the SOURCE span's identity ``(trace_id, span_id, hook_type)``
— NOT ``trace_id`` alone, because nested LangGraph tool/LLM activity spans share
one OTel ``trace_id`` and a trace-only pin would collide across activities. At
COMPLETED it BINDS that pinned context into the store (``activity_scope``) and
delegates to the base ``HookRuntime.completed``, so Core builds, evaluates, and
marks abort/halt all against the pinned activity — never the drifted store
context. A completed stage with no matching pin defers to the base resolver
unchanged.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from openbox_core.context import activity_scope
from openbox_core.contracts.context import ActivityContext
from openbox_core.contracts.otel_spans import HookType
from openbox_core.hooks.events import resolve_context
from openbox_core.hooks.preflight import HookRuntime
from openbox_core.otel.trace_context import raw_trace_id

__all__ = ["LangGraphHookRuntime"]

# Bound on in-flight pins. A pin is normally cleared at completed (or when
# preflight blocks); this cap only protects against an unbounded leak from
# started-without-completed operations (e.g. a hard crash between stages).
# Evicts oldest-first — a stale abandoned pin is the correct thing to drop.
_MAX_PENDING = 4096

# (trace_id: int, span_id: int, hook_type: str)
_PinKey = tuple[int, int, str]


class LangGraphHookRuntime(HookRuntime):
    """`HookRuntime` that pins started-stage context and reuses it at completed.

    Drop-in replacement installed via
    ``openbox_core.instrumentation.shared.set_hook_runtime`` after the base
    ``InstrumentationManager`` has installed the wrappers. Every base behavior
    (event assembly, gate evaluation, abort short-circuit, adapter enforcement,
    approval flow, completed future-marks) is inherited unchanged; the only
    override is WHICH ``ActivityContext`` the base runtime resolves at the
    completed stage — the pinned one, bound via ``activity_scope`` so the base
    ``resolve_context``'s ContextVar tier returns it before the drifting
    trace-map fallback is ever consulted.
    """

    def __init__(self, runtime: Any) -> None:
        super().__init__(runtime)
        self._pending: OrderedDict[_PinKey, ActivityContext] = OrderedDict()
        self._pending_lock = threading.Lock()

    # ── Pin bookkeeping ───────────────────────────────────────────────────

    @staticmethod
    def _pin_key(span: Any, hook_type: HookType) -> _PinKey | None:
        """Identity of a SOURCE hook span: (trace_id, span_id, hook_type).

        Returns None when the span carries no usable trace/span id — such a
        span cannot be pinned, so it degrades to the base resolver.
        """
        trace_id = raw_trace_id(span)
        if not trace_id:  # None or 0
            return None
        get_ctx = getattr(span, "get_span_context", None)
        span_ctx = get_ctx() if callable(get_ctx) else None
        span_id = getattr(span_ctx, "span_id", None)
        if not isinstance(span_id, int) or not span_id:
            return None
        ht = hook_type.value if isinstance(hook_type, HookType) else str(hook_type)
        return (trace_id, span_id, ht)

    def _pin(self, key: _PinKey, ctx: ActivityContext) -> None:
        with self._pending_lock:
            self._pending[key] = ctx
            self._pending.move_to_end(key)
            while len(self._pending) > _MAX_PENDING:
                self._pending.popitem(last=False)

    def _unpin(self, key: _PinKey | None) -> ActivityContext | None:
        if key is None:
            return None
        with self._pending_lock:
            return self._pending.pop(key, None)

    def _pin_started(self, span: Any, hook_type: HookType) -> _PinKey | None:
        """Resolve + pin the started-stage context; return the key (or None).

        Only pins when preflight telemetry is enabled — a disabled preflight
        sends no STARTED hook, so there is no started stage to correlate a
        completed stage back to, and a pin would have no counterpart. Also
        only pins a context that carries an activity binding — an unbound
        context would be skipped by the base builder anyway, so leaving it
        unpinned keeps completed on the identical (skip) path.
        """
        if not self._runtime.config.instrumentation.preflight_enabled:
            return None
        key = self._pin_key(span, hook_type)
        if key is None:
            return None
        ctx = resolve_context(self._store, span)
        if ctx is None or not ctx.activity_id or not ctx.activity_type:
            return None
        self._pin(key, ctx)
        return key

    # ── Preflight (started): pin, then defer to base evaluation ───────────

    def preflight(
        self,
        span: Any,
        *,
        hook_type: HookType,
        identifier: str = "",
        fields: Mapping[str, Any] | None = None,
    ) -> bool:
        key = self._pin_started(span, hook_type)
        try:
            proceed = super().preflight(
                span, hook_type=hook_type, identifier=identifier, fields=fields
            )
        except BaseException:
            # Blocked / halted / approval-rejected: no completed callback will
            # follow, so drop the pin rather than leak it.
            self._unpin(key)
            raise
        if not proceed:
            self._unpin(key)
        return proceed

    async def apreflight(
        self,
        span: Any,
        *,
        hook_type: HookType,
        identifier: str = "",
        fields: Mapping[str, Any] | None = None,
    ) -> bool:
        key = self._pin_started(span, hook_type)
        try:
            proceed = await super().apreflight(
                span, hook_type=hook_type, identifier=identifier, fields=fields
            )
        except BaseException:
            self._unpin(key)
            raise
        if not proceed:
            self._unpin(key)
        return proceed

    # ── Completed: bind the pinned context, then defer to the base runtime ─

    def completed(
        self,
        span: Any,
        *,
        hook_type: HookType,
        fields: Mapping[str, Any] | None = None,
    ) -> None:
        # Unpin FIRST — before the base runtime's completed-telemetry-enabled
        # early return — so a pin can never leak on any exit path.
        ctx = self._unpin(self._pin_key(span, hook_type))
        if ctx is None:
            super().completed(span, hook_type=hook_type, fields=fields)
            return
        with activity_scope(ctx, store=self._store):
            super().completed(span, hook_type=hook_type, fields=fields)

    async def acompleted(
        self,
        span: Any,
        *,
        hook_type: HookType,
        fields: Mapping[str, Any] | None = None,
    ) -> None:
        ctx = self._unpin(self._pin_key(span, hook_type))
        if ctx is None:
            await super().acompleted(span, hook_type=hook_type, fields=fields)
            return
        with activity_scope(ctx, store=self._store):
            await super().acompleted(span, hook_type=hook_type, fields=fields)
