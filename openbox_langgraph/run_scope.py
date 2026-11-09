"""One agent session per matter or tenant (PROD-1109).

Core keeps one exposure history per agent and ``scope``. With no scope, every
run is its own session, so what an agent saw on Monday for Client A is
forgotten by Tuesday's run, and two matters handled by one graph would share
nothing either way. The adapter therefore takes the scope from the run itself,
for example ``{"matter_id": "0417"}`` -> ``matter:0417``, and applies it to
every governance event in the run: the pre-screen, model calls, tools and
retrievals. Runs on the same matter share one session; runs on different
matters never do.

A scope the caller already set with ``openbox_core.exposure.exposure_scope``
wins; the adapter never overrides it.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Mapping
from typing import Any

try:
    from openbox_core import exposure as _exposure  # type: ignore[attr-defined,unused-ignore]
except ImportError:  # openbox-sdk-python before exposure fields
    _exposure = None  # type: ignore[assignment,unused-ignore]

logger = logging.getLogger(__name__)

ScopeResolver = Callable[[Any, Mapping[str, Any] | None], "str | None"]


def scope_namespace_for(key: str) -> str:
    """``matter_id`` -> ``matter``; ``tenant`` -> ``tenant``."""
    return key[:-3] if key.endswith("_id") and len(key) > 3 else key


def scope_value(value: Any, namespace: str) -> str | None:
    """A scope string for a raw value, or None when there is none.

    A value that already names its namespace (``matter:0417``) is kept as is.
    """
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (str, int)):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text if ":" in text else f"{namespace}:{text}"


def _lookup(source: Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


async def resolve_run_scope(
    graph: Any,
    input: Any,
    config: Mapping[str, Any] | None,
    *,
    key: str | None,
    namespace: str | None,
    resolver: ScopeResolver | None,
) -> str | None:
    """The scope for one run: the resolver's answer, else ``key`` from the run
    input, then ``config["configurable"]``, then the thread's saved state (a
    follow-up turn on a checkpointed thread usually omits it)."""
    if resolver is not None:
        return (resolver(input, config) or "").strip() or None
    if not key:
        return None
    ns = namespace or scope_namespace_for(key)
    for source in (input, (config or {}).get("configurable") or {}):
        found = scope_value(_lookup(source, key), ns)
        if found:
            return found
    if config and getattr(graph, "checkpointer", None) and hasattr(graph, "aget_state"):
        try:
            snapshot = await graph.aget_state(config)
        except Exception:  # no thread yet, or a checkpointer that cannot answer
            return None
        return scope_value(_lookup(getattr(snapshot, "values", None) or {}, key), ns)
    return None


def caller_scope_set() -> bool:
    """Whether the caller already scoped this run themselves."""
    return _exposure is not None and bool(_exposure.current_exposure_fields().get("scope"))


def enter_scope(scope: str | None, *, force: bool = False) -> contextlib.ExitStack:
    """Enter ``exposure_scope(scope)`` unless there is nothing to do, or the
    caller already scoped the run (``force`` overrides that, for a handoff
    whose scope is the parent's). Close the stack with :func:`exit_scope`."""
    stack = contextlib.ExitStack()
    if scope and _exposure is not None and (force or not caller_scope_set()):
        stack.enter_context(_exposure.exposure_scope(scope))
    return stack


def exit_scope(stack: contextlib.ExitStack) -> None:
    """Leave the scope. An async generator closed from another task resets the
    context variable from a context it was not set in; that context is gone
    anyway, so the error is not worth raising."""
    try:
        stack.close()
    except ValueError:
        logger.debug("OpenBox: exposure scope closed from another context")
