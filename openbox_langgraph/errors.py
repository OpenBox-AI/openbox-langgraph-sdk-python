"""OpenBox LangGraph SDK — Custom exception classes."""

from __future__ import annotations


class OpenBoxError(Exception):
    """Base class for all OpenBox SDK errors."""


class OpenBoxAuthError(OpenBoxError):
    """Raised when the API key is invalid or unauthorized."""


class OpenBoxNetworkError(OpenBoxError):
    """Raised when the OpenBox Core API is unreachable or returns an error."""


class OpenBoxInsecureURLError(OpenBoxError):
    """Raised when an insecure HTTP URL is used for a non-localhost endpoint."""


class GovernanceBlockedError(OpenBoxError):
    """Raised when governance returns a BLOCK or HALT verdict.

    Supports two calling conventions:

    Hook-level (3 positional args):
        GovernanceBlockedError(verdict_str, reason, identifier)
        e.g. GovernanceBlockedError("block", "Blocked by policy", "https://api.example.com")

    SDK-level (keyword args):
        GovernanceBlockedError(reason, policy_id=..., risk_score=...)
    """

    def __init__(
        self,
        verdict_or_reason: str,
        reason_or_policy_id: str | None = None,
        identifier_or_risk_score: str | float | None = None,
        policy_id: str | None = None,
        risk_score: float | None = None,
    ) -> None:
        from openbox_langgraph.types import Verdict  # lazy to avoid circular

        _HOOK_VERDICTS = ("block", "halt", "require_approval", "stop")
        is_hook_call = verdict_or_reason in _HOOK_VERDICTS

        if is_hook_call:
            # GovernanceBlockedError(verdict, reason, identifier)
            self.verdict = Verdict.from_string(verdict_or_reason).value
            reason = reason_or_policy_id or verdict_or_reason
            self.identifier = (
                str(identifier_or_risk_score) if isinstance(identifier_or_risk_score, str) else ""
            )
            self.policy_id = policy_id
            self.risk_score = risk_score
            super().__init__(reason)
        else:
            # GovernanceBlockedError(reason, policy_id?, risk_score?)
            self.verdict = "block"
            self.identifier = ""
            self.policy_id = (
                reason_or_policy_id if isinstance(reason_or_policy_id, str) else policy_id
            )
            self.risk_score = (
                identifier_or_risk_score
                if isinstance(identifier_or_risk_score, (int, float))
                else risk_score
            )
            super().__init__(verdict_or_reason)


class GovernanceHaltError(OpenBoxError):
    """Raised when governance returns a HALT verdict (workflow-level stop)."""

    verdict = "halt"

    def __init__(
        self,
        reason: str,
        *,
        identifier: str = "",
        policy_id: str | None = None,
        risk_score: float | None = None,
    ) -> None:
        super().__init__(reason)
        self.identifier = identifier
        self.policy_id = policy_id
        self.risk_score = risk_score


class GuardrailsValidationError(OpenBoxError):
    """Raised when guardrails validation fails."""

    def __init__(self, reasons: list[str]) -> None:
        msg = f"Guardrails validation failed: {'; '.join(reasons)}"
        super().__init__(msg)
        self.reasons = reasons


class ApprovalExpiredError(OpenBoxError):
    """Raised when the HITL approval window has expired."""


class ApprovalRejectedError(OpenBoxError):
    """Raised when the HITL approval is explicitly rejected."""


class ApprovalTimeoutError(OpenBoxError):
    """Raised when HITL polling exceeds max_wait_ms."""

    def __init__(self, max_wait_ms: int | float) -> None:
        super().__init__(f"HITL approval timed out after {max_wait_ms}ms")
        self.max_wait_ms = max_wait_ms
