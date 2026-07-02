"""Approval parsing is action-first and strict — unknown/empty never auto-approves.

Two deliberate behavior changes from the SDK's prior inline parsing (both
regression-gated here):

* **Action wins over verdict** when both are present.
* **Strict vocabulary**: an empty/unrecognized decision maps to pending
  (``REQUIRE_APPROVAL`` — keep polling), never a lenient auto-ALLOW. A malformed
  or truncated approval body must never resolve to "approved".
"""

from __future__ import annotations

import pytest

from openbox_langgraph.errors import ApprovalExpiredError
from openbox_langgraph.hitl import HITLPollParams, poll_until_decision
from openbox_langgraph.types import ApprovalResponse, HITLConfig, Verdict, parse_approval_response


class _ExpiredAllowClient:
    """Stub whose poll always returns an expired BUT allow-shaped decision."""

    async def poll_approval(self, params: object) -> ApprovalResponse:
        return ApprovalResponse(verdict=Verdict.ALLOW, expired=True)


def test_action_wins_over_verdict() -> None:
    assert parse_approval_response({"action": "allow", "verdict": "block"}).verdict == Verdict.ALLOW
    assert parse_approval_response({"action": "block", "verdict": "allow"}).verdict == Verdict.BLOCK


def test_verdict_used_when_no_action() -> None:
    assert parse_approval_response({"verdict": "allow"}).verdict == Verdict.ALLOW
    assert parse_approval_response({"verdict": "block"}).verdict == Verdict.BLOCK


def test_empty_action_does_not_shadow_verdict() -> None:
    # A whitespace/empty action is ABSENT — it must fall through to the verdict.
    assert parse_approval_response({"action": "   ", "verdict": "allow"}).verdict == Verdict.ALLOW


def test_missing_both_is_pending_not_allow() -> None:
    # The critical fix: an empty body must NOT resolve to approved.
    assert parse_approval_response({}).verdict == Verdict.REQUIRE_APPROVAL


def test_unknown_decision_is_pending_not_allow() -> None:
    # "approve"/"reject" are NOT in the strict vocabulary → pending, not ALLOW.
    assert parse_approval_response({"action": "approve"}).verdict == Verdict.REQUIRE_APPROVAL
    assert parse_approval_response({"action": "garbage"}).verdict == Verdict.REQUIRE_APPROVAL


def test_expired_flag_preserved() -> None:
    assert parse_approval_response({"verdict": "allow", "expired": True}).expired is True


@pytest.mark.asyncio
async def test_expired_allow_shaped_still_raises_expired(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expired-first: an expired approval must raise even when allow-shaped.

    Locks the ordering — the poll loop checks ``expired`` before ``ALLOW`` — so
    an approval that lapsed but happens to carry an allow verdict can never be
    treated as approved.
    """

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("openbox_langgraph.hitl.asyncio.sleep", _no_sleep)
    with pytest.raises(ApprovalExpiredError):
        await poll_until_decision(
            _ExpiredAllowClient(),  # type: ignore[arg-type]
            HITLPollParams(workflow_id="w", run_id="r", activity_id="a", activity_type="tool_call"),
            HITLConfig(),
        )
