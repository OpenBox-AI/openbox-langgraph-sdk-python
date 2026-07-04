"""OpenBox LangGraph SDK — Configuration & initialization."""

# NOTE: No module-level logging import — lazy-loaded to avoid sandbox restrictions
# where applicable (mirrors openbox-temporal-sdk-python pattern).

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from openbox_langgraph.errors import (
    OpenBoxAuthError,
    OpenBoxConfigError,
    OpenBoxInsecureURLError,
    OpenBoxNetworkError,
)
from openbox_langgraph.identity import (
    AgentIdentityConfig,
    parse_optional_agent_identity_config,
)
from openbox_langgraph.types import DEFAULT_HITL_CONFIG, HITLConfig

if TYPE_CHECKING:
    from logging import Logger

# API key format pattern (obx_live_... or obx_test_...)
_API_KEY_PATTERN = re.compile(r"^obx_(live|test)_[a-zA-Z0-9_]+$")


def _get_logger() -> Logger:
    """Lazy logger import."""
    import logging

    return logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# API key / URL validation
# ═══════════════════════════════════════════════════════════════════


def validate_api_key_format(api_key: str) -> bool:
    """Return True if the API key matches the expected `obx_live_*` / `obx_test_*` format."""
    return bool(_API_KEY_PATTERN.match(api_key))


def validate_url_security(api_url: str) -> None:
    """Raise `OpenBoxInsecureURLError` if the URL uses HTTP on a non-localhost host."""
    from urllib.parse import urlparse  # lazy — avoids sandbox os.stat

    parsed = urlparse(api_url)
    is_localhost = parsed.hostname in ("localhost", "127.0.0.1", "::1")
    if parsed.scheme == "http" and not is_localhost:
        msg = (
            f"Insecure HTTP URL detected: {api_url}. "
            "Use HTTPS for non-localhost URLs to protect API keys in transit."
        )
        raise OpenBoxInsecureURLError(msg)


# ═══════════════════════════════════════════════════════════════════
# GovernanceConfig
# ═══════════════════════════════════════════════════════════════════


@dataclass
class GovernanceConfig:
    """Full resolved governance configuration for the handler."""

    on_api_error: str = "fail_open"  # "fail_open" | "fail_closed"
    api_timeout: float = 30.0  # seconds (not ms)
    send_chain_start_event: bool = True
    send_chain_end_event: bool = True
    send_tool_start_event: bool = True
    send_tool_end_event: bool = True
    send_llm_start_event: bool = True
    send_llm_end_event: bool = True
    skip_chain_types: set[str] = field(default_factory=set)
    skip_tool_types: set[str] = field(default_factory=set)
    hitl: HITLConfig = field(default_factory=HITLConfig)
    session_id: str | None = None
    multi_agent_session_id: str | None = None
    """Optional multi-agent session correlation id, distinct from `session_id`.

    Threaded onto the base SDK's `ActivityContext.multi_agent_session_id` at
    activity-boundary dual-write registration — never mixed into `session_id`,
    which stays the single-agent chat-session identifier the legacy governance
    events already carry."""
    agent_name: str | None = None
    task_queue: str = "langgraph"
    use_native_interrupt: bool = False
    use_core_instrumentation: bool = True
    """Route hook governance through the shared ``openbox_core`` base
    instrumentation — the only hook runtime. Default ``True``. Setting it
    ``False`` fails fast (``OpenBoxConfigError``): legacy in-repo hook
    governance has been removed, so there is nothing to fall back to."""
    strict_activity_context: bool = False
    """Fail loudly when a tool cannot be bound to a proven ActivityContext.

    A tool run is bindable only inside a governed turn — the handler supplies
    the turn's ``workflow_id``/``run_id`` down each tool's ``RunnableConfig``.
    When that turn identity is absent (e.g. a tool driven outside the handler),
    the ToolNode binder cannot prove which activity a hook span belongs to.

    Default ``False``: log a warning once per such invocation and execute the
    tool UNBOUND (its hook spans stay intentionally unattached — never guessed).
    ``True`` (a test/debug aid): raise ``OpenBoxConfigError`` BEFORE the tool
    body runs, so the tool never executes unbound. Note the raise is thrown from
    inside the ``ToolNode`` call, so a ``ToolNode`` with the default
    ``handle_tool_errors=True`` converts it into an error ``ToolMessage``
    (loud and visible, tool still not run) rather than propagating; build the
    ``ToolNode`` with ``handle_tool_errors=False`` to make it hard-fail the run."""
    root_node_names: set[str] = field(default_factory=set)
    tool_type_map: dict[str, str] = field(default_factory=dict)
    """Optional mapping of tool name → tool_type for execution tree classification.

    Example: {"search_web": "http", "query_db": "database", "write_file": "builtin"}
    Supported values: "http", "database", "builtin", "a2a".
    If a tool is not listed, no prefix is added to the label (bare tool name shown).
    A2A is set automatically when subagent_name is resolved — no need to list "task" here.
    """


DEFAULT_GOVERNANCE_CONFIG = GovernanceConfig()


def merge_config(partial: dict[str, Any] | None = None) -> GovernanceConfig:
    """Merge a partial config dict over the defaults and return a `GovernanceConfig`."""
    if not partial:
        return GovernanceConfig()

    hitl_partial = partial.get("hitl") or {}
    if isinstance(hitl_partial, HITLConfig):
        hitl = hitl_partial
    else:
        # Accept poll_interval_ms or poll_interval_s; same for max_wait
        hitl = HITLConfig(
            enabled=hitl_partial.get("enabled", DEFAULT_HITL_CONFIG.enabled),
            poll_interval_ms=hitl_partial.get(
                "poll_interval_ms", DEFAULT_HITL_CONFIG.poll_interval_ms
            ),
            skip_tool_types=set(hitl_partial.get("skip_tool_types", [])),
        )

    def _to_set(v: Any) -> set[str]:
        if isinstance(v, set):
            return v
        if isinstance(v, (list, tuple)):
            return set(v)
        return set()

    # api_timeout: accept both seconds (float ≤ 600) and milliseconds (int > 600)
    raw_timeout = partial.get("api_timeout", DEFAULT_GOVERNANCE_CONFIG.api_timeout)
    api_timeout = float(raw_timeout) if raw_timeout <= 600 else float(raw_timeout) / 1000.0

    raw_tool_type_map = partial.get("tool_type_map")
    tool_type_map: dict[str, str] = raw_tool_type_map if isinstance(raw_tool_type_map, dict) else {}

    return GovernanceConfig(
        on_api_error=partial.get("on_api_error", DEFAULT_GOVERNANCE_CONFIG.on_api_error),
        api_timeout=api_timeout,
        send_chain_start_event=partial.get("send_chain_start_event", True),
        send_chain_end_event=partial.get("send_chain_end_event", True),
        send_tool_start_event=partial.get("send_tool_start_event", True),
        send_tool_end_event=partial.get("send_tool_end_event", True),
        send_llm_start_event=partial.get("send_llm_start_event", True),
        send_llm_end_event=partial.get("send_llm_end_event", True),
        skip_chain_types=_to_set(partial.get("skip_chain_types")),
        skip_tool_types=_to_set(partial.get("skip_tool_types")),
        hitl=hitl,
        session_id=partial.get("session_id"),
        multi_agent_session_id=partial.get("multi_agent_session_id"),
        agent_name=partial.get("agent_name"),
        task_queue=partial.get("task_queue", "langgraph"),
        use_native_interrupt=partial.get("use_native_interrupt", False),
        use_core_instrumentation=partial.get("use_core_instrumentation", True),
        strict_activity_context=partial.get("strict_activity_context", False),
        root_node_names=_to_set(partial.get("root_node_names")),
        tool_type_map=tool_type_map,
    )


# ═══════════════════════════════════════════════════════════════════
# Global Config Singleton
# ═══════════════════════════════════════════════════════════════════


@dataclass
class _GlobalConfigState:
    api_url: str = ""
    api_key: str = ""
    governance_timeout: float = 30.0  # seconds
    agent_did: str | None = None
    agent_private_key: str | None = None

    def configure(
        self,
        api_url: str,
        api_key: str,
        governance_timeout: float = 30.0,
        agent_identity: AgentIdentityConfig | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.governance_timeout = governance_timeout
        self.agent_did = agent_identity.did if agent_identity else None
        self.agent_private_key = agent_identity.private_key if agent_identity else None

    def __repr__(self) -> str:
        if self.api_key and len(self.api_key) > 8:
            masked = f"obx_****{self.api_key[-4:]}"
        elif self.api_key:
            masked = "****"
        else:
            masked = ""
        return (
            f"_GlobalConfigState(api_url={self.api_url!r}, "
            f"api_key={masked!r}, "
            f"governance_timeout={self.governance_timeout}, "
            f"agent_did={self.agent_did!r})"
        )

    def is_configured(self) -> bool:
        return bool(self.api_url and self.api_key)


_global_config = _GlobalConfigState()


def get_global_config() -> _GlobalConfigState:
    """Return the global SDK configuration singleton."""
    return _global_config


# ═══════════════════════════════════════════════════════════════════
# Server-side API key validation (sync, using urllib — no httpx at module level)
# ═══════════════════════════════════════════════════════════════════


def _validate_api_key_with_server(
    api_url: str,
    api_key: str,
    timeout: float,
    agent_identity: AgentIdentityConfig | None = None,
) -> None:
    """Validate API key by calling /api/v1/auth/validate endpoint (synchronous).

    Uses urllib to avoid importing httpx at module level, matching the
    openbox-temporal-sdk-python pattern for sandbox compatibility.
    """
    from urllib.error import HTTPError, URLError  # lazy
    from urllib.request import Request, urlopen  # lazy

    from openbox_langgraph.client import build_auth_headers  # lazy

    try:
        req = Request(
            f"{api_url}/api/v1/auth/validate",
            headers=build_auth_headers(
                api_key,
                method="GET",
                pathname="/api/v1/auth/validate",
                body=b"",
                agent_identity=agent_identity,
            ),
            method="GET",
        )
        with urlopen(req, timeout=timeout) as response:
            if response.getcode() != 200:
                msg = "Invalid API key. Check your API key at dashboard.openbox.ai"
                raise OpenBoxAuthError(msg)
            _get_logger().info("OpenBox API key validated successfully")

    except HTTPError as e:
        if e.code in (401, 403):
            msg = "Invalid API key. Check your API key at dashboard.openbox.ai"
            raise OpenBoxAuthError(msg) from e
        msg = f"Cannot reach OpenBox Core at {api_url}: HTTP {e.code}"
        raise OpenBoxNetworkError(msg) from e
    except URLError as e:
        msg = f"Cannot reach OpenBox Core at {api_url}: {e.reason}"
        raise OpenBoxNetworkError(msg) from e
    except (OpenBoxAuthError, OpenBoxNetworkError):
        raise
    except Exception as e:
        msg = f"Cannot reach OpenBox Core at {api_url}: {e}"
        raise OpenBoxNetworkError(msg) from e


# ═══════════════════════════════════════════════════════════════════
# initialize() — synchronous, matching openbox-temporal-sdk-python
# ═══════════════════════════════════════════════════════════════════


def initialize(
    api_url: str,
    api_key: str,
    governance_timeout: float = 30.0,
    validate: bool = True,
    agent_did: str | None = None,
    agent_private_key: str | None = None,
) -> None:
    """Initialize the OpenBox LangGraph SDK with credentials.

    Synchronous — safe to call at module level or from any context.
    Validates the API key format and optionally pings the server.

    Args:
        api_url: Base URL of your OpenBox Core instance.
        api_key: API key in `obx_live_*` or `obx_test_*` format.
        governance_timeout: HTTP timeout in **seconds** for governance calls (default 30.0).
        validate: If True, validates the API key against the server on startup.
        agent_did: Optional OpenBox agent DID. Falls back to `OPENBOX_AGENT_DID`.
        agent_private_key: Optional raw Ed25519 private key seed. Falls back to
            `OPENBOX_AGENT_PRIVATE_KEY`.
    """
    import os

    validate_url_security(api_url)

    if not validate_api_key_format(api_key):
        # Show only the structural prefix, never actual secret chars
        parts = api_key.split("_", 2)
        prefix_hint = "_".join(parts[:2]) + "_..." if len(parts) >= 2 else "???"
        msg = (
            f"Invalid API key format. Expected 'obx_live_*' or 'obx_test_*', "
            f"got prefix: '{prefix_hint}'"
        )
        raise OpenBoxAuthError(msg)

    try:
        agent_identity = parse_optional_agent_identity_config(
            did=agent_did or os.environ.get("OPENBOX_AGENT_DID"),
            private_key=agent_private_key or os.environ.get("OPENBOX_AGENT_PRIVATE_KEY"),
        )
    except OpenBoxConfigError:
        raise

    if validate:
        _validate_api_key_with_server(
            api_url.rstrip("/"),
            api_key,
            governance_timeout,
            agent_identity,
        )

    _global_config.configure(
        api_url.rstrip("/"),
        api_key,
        governance_timeout,
        agent_identity,
    )

    _get_logger().info(f"OpenBox LangGraph SDK initialized with API URL: {api_url}")
