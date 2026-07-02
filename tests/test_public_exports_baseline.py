"""Public-API surface guard.

Freezes the exact ``openbox_langgraph.__all__`` export list. Any change to the
public surface must be a *deliberate* edit to ``EXPECTED_EXPORTS`` below:

* Adding a name here without adding it to ``__all__`` (or vice versa) fails the
  test — this catches accidental public-API additions during internal rewiring.
* Removing a previously-exported name fails the test — this catches accidental
  breaking changes to importers pinned on the current surface.

Every exported name is also asserted to be a real, importable attribute so the
snapshot can never drift from an alias that no longer resolves.
"""

from __future__ import annotations

import pytest

import openbox_langgraph

# Frozen snapshot of the public surface. Update ONLY when an export is added or
# removed on purpose (and call it out in the changelog).
EXPECTED_EXPORTS = frozenset(
    {
        "__version__",
        "DEFAULT_HITL_CONFIG",
        "AgentIdentityConfig",
        "ApprovalExpiredError",
        "ApprovalRejectedError",
        "ApprovalResponse",
        "ApprovalTimeoutError",
        "GovernanceBlockedError",
        "GovernanceClient",
        "GovernanceConfig",
        "GovernanceHaltError",
        "GovernanceVerdictResponse",
        "GuardrailsReason",
        "GuardrailsResult",
        "GuardrailsValidationError",
        "HITLConfig",
        "LangChainGovernanceEvent",
        "LangGraphStreamEvent",
        "OpenBoxAuthError",
        "OpenBoxConfigError",
        "OpenBoxError",
        "OpenBoxInsecureURLError",
        "OpenBoxLangGraphHandler",
        "OpenBoxLangGraphHandlerOptions",
        "OpenBoxNetworkError",
        "Verdict",
        "VerdictContext",
        "WorkflowEventType",
        "WorkflowSpanBuffer",
        "WorkflowSpanProcessor",
        "build_agent_identity_canonical_request",
        "build_auth_headers",
        "create_agent_identity_headers",
        "create_openbox_graph_handler",
        "create_span",
        "enforce_verdict",
        "get_global_config",
        "highest_priority_verdict",
        "initialize",
        "is_hitl_applicable",
        "lang_graph_event_to_context",
        "merge_config",
        "parse_approval_response",
        "parse_governance_response",
        "parse_optional_agent_identity_config",
        "poll_until_decision",
        "rfc3339_now",
        "safe_serialize",
        "setup_opentelemetry_for_governance",
        "to_server_event_type",
        "traced",
        "validate_agent_identity_config",
        "verdict_from_string",
        "verdict_priority",
        "verdict_requires_approval",
        "verdict_should_stop",
    }
)


def test_all_matches_frozen_snapshot() -> None:
    actual = set(openbox_langgraph.__all__)
    missing = EXPECTED_EXPORTS - actual
    added = actual - EXPECTED_EXPORTS
    assert not missing, f"public exports removed (breaking): {sorted(missing)}"
    assert not added, f"public exports added without snapshot update: {sorted(added)}"


def test_all_entries_are_importable() -> None:
    unresolved = [
        name for name in openbox_langgraph.__all__ if not hasattr(openbox_langgraph, name)
    ]
    assert not unresolved, f"names in __all__ that do not resolve: {unresolved}"


def test_all_has_no_duplicates() -> None:
    names = list(openbox_langgraph.__all__)
    assert len(names) == len(set(names)), "duplicate names in __all__"
    # Public surface is 56 (55 baseline exports + __version__); a change here is
    # a deliberate surface change.
    assert len(EXPECTED_EXPORTS) == 56


def test_legacy_otel_setup_export_is_a_raising_shim() -> None:
    """`setup_opentelemetry_for_governance` stays exported (surface stability)
    but is now a deprecated shim: legacy in-repo hook governance was removed,
    so calling it raises rather than silently installing nothing."""
    from openbox_langgraph.errors import OpenBoxConfigError

    with pytest.raises(OpenBoxConfigError):
        openbox_langgraph.setup_opentelemetry_for_governance()
