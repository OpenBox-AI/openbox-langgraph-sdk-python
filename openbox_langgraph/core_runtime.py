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
"""

from __future__ import annotations

from openbox_core.config import OpenBoxConfig
from openbox_core.context import ContextStore
from openbox_core.runtime import OpenBoxRuntime

from openbox_langgraph.config import GovernanceConfig

# SDK-specific env namespace. Resolution order is explicit > OPENBOX_LANGGRAPH_*
# > OPENBOX_* > defaults (handled by OpenBoxConfig.resolve).
CORE_ENV_PREFIX = "OPENBOX_LANGGRAPH"


def create_core_runtime(
    config: GovernanceConfig,
    *,
    api_url: str | None = None,
    api_key: str | None = None,
    governance_timeout: float | None = None,
    agent_did: str | None = None,
    agent_private_key: str | None = None,
) -> OpenBoxRuntime:
    """Resolve base-SDK config and build an isolated ``OpenBoxRuntime``.

    ``validate=True`` runs the base normalizer: URL-security (rejects
    non-localhost ``http://``), timeout float coercion, API-key format, and the
    DID/private-key both-or-neither rule. Explicit ``None`` arguments fall
    through to the env layers rather than overriding them.

    No adapter and no instrumentation are wired here — that is the opt-in hook
    runtime's responsibility. Construction performs no network I/O.
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
    return OpenBoxRuntime(core_config, context_store=ContextStore())
