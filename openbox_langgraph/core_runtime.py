"""Opt-in construction of a base-SDK ``OpenBoxRuntime`` for LangGraph.

Builds an :class:`openbox_core.runtime.OpenBoxRuntime` from the LangGraph
configuration surface WITHOUT changing any default behavior. Nothing here runs
unless a caller explicitly constructs a runtime; the legacy ``GovernanceClient``
plus in-repo hooks remain the only default execution path.

Two isolation guarantees matter here:

* Each runtime is given a PRIVATE :class:`~openbox_core.context.ContextStore`.
  The base default is a process-global store, and ``runtime.close()`` clears its
  store — so sharing one would let a handler's shutdown/instrumentation blast
  another handler in the same process. One runtime per handler, one private
  store per runtime.
* DID identity is preserved end to end. The runtime builds its
  ``EvaluationClient`` with ``config.load_identity()``, so a signed LangGraph
  configuration keeps signing base-SDK requests — never a silent downgrade to
  bare-Bearer at the trust boundary.

The trace-lookup fallback shim (``TraceContextRegistry``, ``ContextMissMetrics``,
``get_trace_registry``, ``get_context_store``) that resolves a runtime's
private ``ContextStore`` for a spawned LangGraph tool/LLM task lives in
``trace_context_registry.py`` — re-exported here so existing callers of
``openbox_langgraph.core_runtime`` do not need an import-path change.

``config.use_core_instrumentation=True`` additionally wires the
``LangGraphFrameworkAdapter`` + ``FallbackContextStore`` and installs base
instrumentation via a manually constructed ``InstrumentationManager`` (not
the opaque ``runtime.install_instrumentation()``) so ``extra_ignored_urls``
can carry the SAME URL prefixes the legacy OTel setup ignores — see
``create_core_runtime``'s docstring for the single-provider / ignored-URL
union this makes possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openbox_core.config import OpenBoxConfig
from openbox_core.context import ContextStore
from openbox_core.runtime import OpenBoxRuntime

from openbox_langgraph.config import GovernanceConfig
from openbox_langgraph.trace_context_registry import (
    ContextMissMetrics,
    TraceContextRegistry,
    get_context_store,
    get_trace_registry,
)

if TYPE_CHECKING:
    from openbox_langgraph.span_processor import WorkflowSpanProcessor

# SDK-specific env namespace. Resolution order is explicit > OPENBOX_LANGGRAPH_*
# > OPENBOX_* > defaults (handled by OpenBoxConfig.resolve).
CORE_ENV_PREFIX = "OPENBOX_LANGGRAPH"

__all__ = [
    "CORE_ENV_PREFIX",
    "ContextMissMetrics",
    "TraceContextRegistry",
    "create_core_runtime",
    "get_context_store",
    "get_trace_registry",
]


def create_core_runtime(
    config: GovernanceConfig,
    *,
    api_url: str | None = None,
    api_key: str | None = None,
    governance_timeout: float | None = None,
    agent_did: str | None = None,
    agent_private_key: str | None = None,
    legacy_span_processor: WorkflowSpanProcessor | None = None,
    extra_ignored_urls: set[str] | None = None,
) -> OpenBoxRuntime:
    """Resolve base-SDK config and build an isolated ``OpenBoxRuntime``.

    ``validate=True`` runs the base normalizer: URL-security (rejects
    non-localhost ``http://``), timeout float coercion, API-key format, and the
    DID/private-key both-or-neither rule. Explicit ``None`` arguments fall
    through to the env layers rather than overriding them.

    ``config.use_core_instrumentation=False`` (the default): no adapter and no
    instrumentation are wired — construction performs no network I/O and the
    returned runtime is inert until a caller does something with it (matches
    every caller that predates the opt-in hook runtime).

    ``config.use_core_instrumentation=True``: the runtime is fully armed —

    * ``context_store`` is a :class:`~openbox_langgraph.fallback_context_store.FallbackContextStore`
      (exact-trace lookup, falling through to the single-active/last-registered
      tiers on a miss — see that module for why the base SDK needs this).
    * ``adapter`` is a :class:`~openbox_langgraph.core_adapter.LangGraphFrameworkAdapter`
      bound to the SAME store (+ ``legacy_span_processor``, when given, for the
      post-approval reset that keeps both governance stores in sync).
    * Base instrumentation (HTTP/DB/file/function wrappers) is installed
      before this function returns, via an ``InstrumentationManager``
      constructed directly (mirroring ``openbox_core.conformance.instrumentation``)
      rather than ``runtime.install_instrumentation()``, so ``extra_ignored_urls``
      can be threaded through — the manager's self-instrumentation guard then
      ignores every URL EITHER side already ignores, a true union rather than
      whichever set happened to be configured last.
    * ``runtime.config.api_url`` reuses ``get_or_create_tracer_provider``'s
      existing-provider-wins behavior: the legacy ``setup_opentelemetry_for_governance``
      call in ``langgraph_handler.py`` runs immediately after this function
      returns and installs the SAME check (create only if the global provider
      isn't already an SDK ``TracerProvider``) — so exactly one provider ever
      exists in the process regardless of which of the two setup calls runs
      first.
    """
    core_config = OpenBoxConfig.resolve(
        env_prefix=CORE_ENV_PREFIX,
        api_url=api_url,
        api_key=api_key,
        timeout_seconds=governance_timeout,
        on_api_error=config.on_api_error,
        agent_name=config.agent_name,
        agent_did=agent_did,
        agent_private_key=agent_private_key,
        validate=True,
    )
    if not config.use_core_instrumentation:
        return OpenBoxRuntime(core_config, context_store=ContextStore())

    from openbox_core.instrumentation.manager import InstrumentationManager

    from openbox_langgraph.core_adapter import LangGraphFrameworkAdapter
    from openbox_langgraph.fallback_context_store import FallbackContextStore

    store = FallbackContextStore()
    adapter = LangGraphFrameworkAdapter(legacy_span_processor, context_store=store)
    runtime = OpenBoxRuntime(core_config, adapter, context_store=store)
    manager = InstrumentationManager(runtime, extra_ignored_urls=extra_ignored_urls)
    # Same pattern the base SDK's own conformance kit uses
    # (openbox_core.conformance.instrumentation.installed_conformance_runtime)
    # to get extra_ignored_urls past the no-arg runtime.install_instrumentation()
    # facade; runtime.close()/aclose() still call uninstall_instrumentation(),
    # which uses whichever manager is attached here.
    runtime._instrumentation_manager = manager
    manager.install()
    # manager.install() published a base HookRuntime to the shared
    # instrumentation state; swap in the LangGraph-owned one that pins a source
    # span's ActivityContext at STARTED and reuses it at COMPLETED, so a hook
    # span can't be split across two activities when the trace-lookup fallback
    # drifts to a later activity. manager.uninstall() (via runtime.close/aclose)
    # resets the shared runtime to None, so no teardown change is needed here.
    from openbox_core.instrumentation.shared import set_hook_runtime

    from openbox_langgraph.langgraph_hook_runtime import LangGraphHookRuntime

    set_hook_runtime(LangGraphHookRuntime(runtime))
    return runtime
