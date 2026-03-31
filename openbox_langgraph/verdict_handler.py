"""OpenBox LangGraph SDK — Verdict enforcement."""

from __future__ import annotations

import re
import warnings
from typing import Literal

from openbox_langgraph.errors import (
    GovernanceBlockedError,
    GovernanceHaltError,
    GuardrailsValidationError,
)
from openbox_langgraph.types import (
    GovernanceVerdictResponse,
    Verdict,
    verdict_requires_approval,
)

# ═══════════════════════════════════════════════════════════════════
# VerdictContext
# ═══════════════════════════════════════════════════════════════════

VerdictContext = Literal[
    "chain_start",
    "chain_end",
    "tool_start",
    "tool_end",
    "llm_start",
    "llm_end",
    "agent_action",
    "agent_finish",
    "graph_node_start",
    "graph_node_end",
    "graph_root_start",
    "graph_root_end",
    "other",
]


def lang_graph_event_to_context(event_type: str, *, is_root: bool = False) -> VerdictContext:
    """Map a LangGraph v2 stream event type to a `VerdictContext`.

    Args:
        event_type: Raw LangGraph event string (e.g. `on_tool_start`).
        is_root: Whether this event belongs to the outermost graph invocation.
    """
    mapping: dict[str, VerdictContext] = {
        "on_chain_start": "graph_root_start" if is_root else "graph_node_start",
        "on_chain_end": "graph_root_end" if is_root else "graph_node_end",
        "on_chat_model_start": "llm_start",
        "on_chat_model_end": "llm_end",
        "on_tool_start": "tool_start",
        "on_tool_end": "tool_end",
        "on_retriever_start": "tool_start",
        "on_retriever_end": "tool_end",
    }
    return mapping.get(event_type, "other")


def is_hitl_applicable(context: VerdictContext) -> bool:
    """Return True if HITL polling applies to this verdict context.

    HITL applies to start events (before execution) and tool/LLM end events,
    because Behavior Rules on ActivityCompleted can return REQUIRE_APPROVAL.
    """
    return context in (
        "tool_start",
        "tool_end",
        "llm_start",
        "llm_end",
        "graph_node_start",
        "agent_action",
    )


# ═══════════════════════════════════════════════════════════════════
# Guardrail reason helpers (mirrors TS SDK)
# ═══════════════════════════════════════════════════════════════════

def _clean_guardrail_reason(reason: str) -> str:
    """Strip ReAct scratchpad contamination from a guardrail reason string.

    Guardrail services may echo the full prompt/trace including agent scratchpad.
    We only want the human-readable reason header and the quoted offending text.
    """
    # 1) Strip ReAct "Question:" line (includes session context)
    reason = re.sub(r"\n?-\s*Question:\s*\[Session context\][^\n]*\n?", "", reason)
    # 2) Strip agent scratchpad (Thought:, Action:, etc.)
    for marker in ("\n\nThought:", "\n\nThought", "\nThought:", "\nThought"):
        idx = reason.find(marker)
        if idx >= 0:
            return reason[:idx].rstrip()
    return reason.rstrip()


def _get_guardrail_failure_reasons(reasons: list | None) -> list[str]:
    """Return a single cleaned reason string for input guardrail failures, mirroring TS SDK:
    - Take only the first reason (Core already returns a primary one)
    - Clean off any agent scratchpad sections
    """
    first = next((r.reason for r in (reasons or []) if r.reason), None)
    if not first:
        return ["Guardrails validation failed"]
    return [_clean_guardrail_reason(first)]


def _get_guardrail_output_failure_reasons(reasons: list | None) -> list[str]:
    """Return all reason strings for output guardrail failures, mirroring Temporal SDK:
    - Join all reasons with '; ' (output has no ReAct scratchpad contamination)
    """
    strings = [r.reason for r in (reasons or []) if r.reason]
    return ["; ".join(strings)] if strings else ["Guardrails output validation failed"]


# ═══════════════════════════════════════════════════════════════════
# enforce_verdict
# ═══════════════════════════════════════════════════════════════════

class VerdictEnforcementResult:
    """Result of `enforce_verdict` — indicates what the caller should do next."""

    def __init__(self, *, requires_hitl: bool = False, blocked: bool = False) -> None:
        self.requires_hitl = requires_hitl
        self.blocked = blocked


_OBSERVATION_ONLY_CONTEXTS: frozenset[VerdictContext] = frozenset({
    "chain_end",
    "graph_root_end",
    # graph_node_end is NOT observation-only: ToolCompleted/LLMCompleted verdicts
    # from Behavior Rules must be enforced, matching Temporal SDK's ActivityCompleted handling.
    "agent_finish",
    "other",
})


def enforce_verdict(
    response: GovernanceVerdictResponse,
    context: VerdictContext,
) -> VerdictEnforcementResult:
    """Enforce the governance verdict by raising an error or signalling HITL.

    Observation-only contexts (chain/root/node end, agent_finish) are never
    enforced — they are for telemetry only.

    Args:
        response: The verdict response from OpenBox Core.
        context: The verdict context that determines enforcement behaviour.

    Returns:
        A `VerdictEnforcementResult` indicating whether HITL should start.

    Raises:
        GuardrailsValidationError: When guardrails validation has failed.
        GovernanceHaltError: When the verdict is HALT.
        GovernanceBlockedError: When the verdict is BLOCK.
        GovernanceBlockedError: When REQUIRE_APPROVAL arrives at a non-HITL context.
    """
    if context in _OBSERVATION_ONLY_CONTEXTS:
        return VerdictEnforcementResult()

    verdict = response.verdict
    reason = response.reason
    policy_id = response.policy_id
    risk_score = response.risk_score

    # 1. HALT — highest priority, mirrors Temporal: checked before guardrails
    if verdict == Verdict.HALT:
        msg = reason or "Workflow halted by governance policy"
        raise GovernanceHaltError(msg, policy_id=policy_id, risk_score=risk_score)

    # 2. BLOCK — checked before guardrails (Temporal order)
    if verdict == Verdict.BLOCK:
        msg = reason or "Action blocked by governance policy"
        raise GovernanceBlockedError(msg, policy_id, risk_score)

    # 3. Guardrails validation failure — block when verdict is ALLOW/CONSTRAIN but guardrails failed
    if response.guardrails_result and not response.guardrails_result.validation_passed:
        gr = response.guardrails_result
        if gr.input_type == "activity_output":
            # Output guardrail block: all reasons joined (Temporal pattern, no scratchpad)
            reasons = _get_guardrail_output_failure_reasons(gr.reasons)
        else:
            # Input guardrail block: first reason, scratchpad-cleaned (TS SDK pattern)
            reasons = _get_guardrail_failure_reasons(gr.reasons)
        raise GuardrailsValidationError(reasons)

    # 4. REQUIRE_APPROVAL
    if verdict_requires_approval(verdict):
        if is_hitl_applicable(context):
            return VerdictEnforcementResult(requires_hitl=True)
        msg = reason or "Action requires approval but cannot be paused at this stage"
        raise GovernanceBlockedError(msg, policy_id, risk_score)

    # 5. CONSTRAIN — warn and continue
    if verdict == Verdict.CONSTRAIN and reason:
        suffix = f" (policy: {policy_id})" if policy_id else ""
        warnings.warn(f"[OpenBox] Governance constraint: {reason}{suffix}", stacklevel=2)
        return VerdictEnforcementResult()

    # 6. ALLOW — no action
    return VerdictEnforcementResult()
