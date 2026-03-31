"""OpenBox LangGraph SDK — Human-in-the-Loop (HITL) approval polling."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from openbox_langgraph.client import ApprovalPollParams, GovernanceClient
from openbox_langgraph.errors import (
    ApprovalExpiredError,
    ApprovalRejectedError,
    GovernanceBlockedError,
)
from openbox_langgraph.types import HITLConfig, Verdict, verdict_should_stop


@dataclass
class HITLPollParams:
    """Parameters for the HITL polling loop."""

    workflow_id: str
    run_id: str
    activity_id: str
    activity_type: str


async def poll_until_decision(
    client: GovernanceClient,
    params: HITLPollParams,
    config: HITLConfig,
) -> None:
    """Block until governance approves or rejects.

    Polls OpenBox Core indefinitely until a terminal verdict is received.
    The server controls approval expiration — the SDK does not impose a deadline.

    Resolves (returns None) when approval is granted (ALLOW verdict).
    Raises on rejection, expiry, or HALT/BLOCK verdict.

    Args:
        client: The governance HTTP client.
        params: Identifiers for the pending approval.
        config: HITL polling configuration (poll_interval_ms).

    Raises:
        ApprovalExpiredError: When the approval window has expired (server-side).
        ApprovalRejectedError: When approval is explicitly rejected.
        GovernanceBlockedError: When verdict is BLOCK.
    """
    while True:
        await asyncio.sleep(config.poll_interval_ms / 1000.0)

        response = await client.poll_approval(
            ApprovalPollParams(
                workflow_id=params.workflow_id,
                run_id=params.run_id,
                activity_id=params.activity_id,
            )
        )

        if response is None:
            # API unreachable — keep polling (fail-open for HITL)
            continue

        if response.expired:
            msg = (
                f"Approval expired for {params.activity_type} "
                f"(activity_id={params.activity_id})"
            )
            raise ApprovalExpiredError(msg)

        verdict = response.verdict
        reason = response.reason

        if verdict == Verdict.ALLOW:
            return

        # HALT or any stop verdict → rejected
        if verdict_should_stop(verdict):
            msg = reason or f"Approval rejected for {params.activity_type}"
            raise ApprovalRejectedError(msg)

        # BLOCK specifically
        if verdict == Verdict.BLOCK:
            msg = reason or f"Approval rejected for {params.activity_type}"
            raise GovernanceBlockedError("block", msg, params.activity_id)

        # Still pending (REQUIRE_APPROVAL / CONSTRAIN) — keep polling
