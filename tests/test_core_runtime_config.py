"""Tests for ``create_core_runtime`` — the opt-in base-SDK runtime builder.

Verifies layered config resolution + validation, DID identity preservation, and
the private-store isolation guarantee. None of this touches the default
execution path (legacy client + hooks); it only exercises the new opt-in seam.
"""

from __future__ import annotations

import base64

import pytest

pytest.importorskip("openbox_core")

from openbox_core.context import default_context_store
from openbox_core.errors import OpenBoxInsecureURLError
from openbox_core.runtime import OpenBoxRuntime

from openbox_langgraph.config import GovernanceConfig
from openbox_langgraph.core_runtime import create_core_runtime

# Non-real test key material (32-byte seed), same shape as test_did_client_signing.
_PRIVATE_KEY = base64.b64encode(bytes(range(32))).decode("ascii")
_DID = "did:aip:550e8400-e29b-41d4-a716-446655440000"
_URL = "https://core.openbox.ai"
_KEY = "obx_test_abc"


def _runtime(
    config: GovernanceConfig | None = None,
    *,
    api_url: str | None = _URL,
    api_key: str | None = _KEY,
    governance_timeout: float | None = 30.0,
    agent_did: str | None = None,
    agent_private_key: str | None = None,
) -> OpenBoxRuntime:
    return create_core_runtime(
        config or GovernanceConfig(),
        api_url=api_url,
        api_key=api_key,
        governance_timeout=governance_timeout,
        agent_did=agent_did,
        agent_private_key=agent_private_key,
    )


def test_builds_runtime_with_private_store() -> None:
    """Each runtime owns a private store, never the process-global default."""
    rt = _runtime()
    rt2 = _runtime()
    try:
        assert rt.context_store is not default_context_store()
        assert rt.context_store is not rt2.context_store
    finally:
        rt.close()
        rt2.close()


def test_preserves_did_identity() -> None:
    """A signed LangGraph config yields a runtime whose client carries identity."""
    rt = _runtime(agent_did=_DID, agent_private_key=_PRIVATE_KEY)
    try:
        assert rt.config.agent_did == _DID
        assert rt.config.load_identity() is not None
    finally:
        rt.close()


def test_unsigned_config_has_no_identity() -> None:
    rt = _runtime()
    try:
        assert rt.config.load_identity() is None
    finally:
        rt.close()


def test_validate_rejects_insecure_non_localhost_url() -> None:
    with pytest.raises(OpenBoxInsecureURLError):
        _runtime(api_url="http://evil.example.com")


def test_on_api_error_threaded_from_config() -> None:
    rt = _runtime(GovernanceConfig(on_api_error="fail_closed"))
    try:
        assert rt.config.on_api_error == "fail_closed"
    finally:
        rt.close()


def test_env_prefix_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """explicit > OPENBOX_LANGGRAPH_* > OPENBOX_* for api_url."""
    monkeypatch.setenv("OPENBOX_API_URL", "https://global.example.com")
    monkeypatch.setenv("OPENBOX_LANGGRAPH_API_URL", "https://prefixed.example.com")

    explicit = _runtime(api_url="https://explicit.example.com")
    prefixed = _runtime(api_url=None)  # falls through to the env layers
    try:
        assert explicit.config.api_url == "https://explicit.example.com"
        assert prefixed.config.api_url == "https://prefixed.example.com"
    finally:
        explicit.close()
        prefixed.close()
