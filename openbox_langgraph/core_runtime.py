"""Construction of a base-SDK ``OpenBoxRuntime`` for LangGraph.

Builds an :class:`openbox_core.runtime.OpenBoxRuntime` from the LangGraph
configuration surface. The base ``openbox_core`` InstrumentationManager is the
ONLY hook runtime — there is no legacy in-repo hook fallback. A handler that
owns its own runtime (non-injected-client path) always routes hook governance
through this runtime; ``create_core_runtime`` fails fast when hook governance
is opted out rather than silently arming nothing.

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

The EXACT trace-registration helper (``TraceContextRegistry``,
``get_trace_registry``, ``get_context_store``) that the handler uses to dual-write
a self-owned OTel parent span's ``ActivityContext`` into a runtime's private
``ContextStore`` lives in ``trace_context_registry.py`` — re-exported here so
existing callers of ``openbox_langgraph.core_runtime`` do not need an
import-path change. That path is EXACT only (a known trace id -> a known
activity); there is no single-active / last-registered guessing.

``config.use_core_instrumentation=True`` additionally wires the
``LangGraphFrameworkAdapter`` and installs base instrumentation via a manually
constructed ``InstrumentationManager`` (not the opaque
``runtime.install_instrumentation()``) so ``extra_ignored_urls`` can carry the
SAME URL prefixes the legacy OTel setup ignores — see ``create_core_runtime``'s
docstring for the single-provider / ignored-URL union this makes possible.
"""

from __future__ import annotations

from openbox_core.config import OpenBoxConfig
from openbox_core.runtime import OpenBoxRuntime

from openbox_langgraph.config import GovernanceConfig
from openbox_langgraph.errors import OpenBoxConfigError
from openbox_langgraph.trace_context_registry import (
    TraceContextRegistry,
    get_context_store,
    get_trace_registry,
)

# SDK-specific env namespace. Resolution order is explicit > OPENBOX_LANGGRAPH_*
# > OPENBOX_* > defaults (handled by OpenBoxConfig.resolve).
CORE_ENV_PREFIX = "OPENBOX_LANGGRAPH"

__all__ = [
    "CORE_ENV_PREFIX",
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
    extra_ignored_urls: set[str] | None = None,
) -> OpenBoxRuntime:
    """Resolve base-SDK config and build an isolated ``OpenBoxRuntime``.

    ``validate=True`` runs the base normalizer: URL-security (rejects
    non-localhost ``http://``), timeout float coercion, API-key format, and the
    DID/private-key both-or-neither rule. Explicit ``None`` arguments fall
    through to the env layers rather than overriding them.

    ``config.use_core_instrumentation=False``: raises ``OpenBoxConfigError``.
    Legacy in-repo hook governance has been removed, so a runtime that arms no
    hook instrumentation would silently govern nothing — refused rather than
    returned inert.

    ``config.use_core_instrumentation=True`` (the default): the runtime is fully armed —

    * ``context_store`` is a plain :class:`~openbox_core.context.ContextStore`.
      Hook context resolves EXACTLY: the ContextVar tier (bound around tool
      execution at the ToolNode seam — see ``tool_activity_binding``) first,
      then an EXACT trace-id lookup for a trace this SDK explicitly registered.
      There is no single-active / last-registered fallback — a hook span is
      attached to the activity it can be proven to belong to, or left unbound.
    * ``adapter`` is a :class:`~openbox_langgraph.core_adapter.LangGraphFrameworkAdapter`
      bound to the SAME store.
    * Base instrumentation (HTTP/DB/file/function wrappers) is installed
      before this function returns, via an ``InstrumentationManager``
      constructed directly (mirroring ``openbox_core.conformance.instrumentation``)
      rather than ``runtime.install_instrumentation()``, so ``extra_ignored_urls``
      can be threaded through — the manager's self-instrumentation guard then
      ignores every URL EITHER side already ignores, a true union rather than
      whichever set happened to be configured last.
    """
    if not config.use_core_instrumentation:
        # No legacy hook fallback exists anymore — the base openbox_core
        # InstrumentationManager is the ONLY hook runtime. Opting out cannot
        # silently degrade to legacy hooks, so refuse to build a runtime that
        # would arm no hook governance at all.
        raise OpenBoxConfigError(
            "use_core_instrumentation=False is no longer supported: legacy in-repo "
            "hook governance has been removed and openbox_core base instrumentation "
            "is the only hook runtime. Set use_core_instrumentation=True (the default)."
        )

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

    from openbox_core.context import ContextStore
    from openbox_core.instrumentation.manager import InstrumentationManager

    from openbox_langgraph.core_adapter import LangGraphFrameworkAdapter

    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    runtime = OpenBoxRuntime(core_config, adapter, context_store=store)
    # The adapter's post-approval abort sweep (reset_after_approval ->
    # clear_aborted_for_workflow) needs the SAME per-runtime registry the
    # handler dual-writes activity keys into, to clear every abort mark for a
    # turn's workflow_id (the base ContextStore has no workflow-scoped clear).
    # The adapter holds the store, not the runtime, so publish that one registry
    # on the store for it to reach via `store.registry` (a dynamic attribute the
    # adapter reads with getattr — the base ContextStore does not declare it).
    store.registry = get_trace_registry(runtime)  # type: ignore[attr-defined]
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
    # span's started and completed stages can't resolve to two different
    # activities if the resolvable context changes between them.
    # manager.uninstall() (via runtime.close/aclose) resets the shared runtime
    # to None, so no teardown change is needed here.
    from openbox_core.instrumentation.shared import set_hook_runtime

    from openbox_langgraph.langgraph_hook_runtime import LangGraphHookRuntime

    set_hook_runtime(LangGraphHookRuntime(runtime))
    return runtime
