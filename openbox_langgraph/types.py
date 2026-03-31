"""OpenBox LangGraph SDK — Core types mirroring sdk-langgraph/src/types.ts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

# ═══════════════════════════════════════════════════════════════════
# Verdict
# ═══════════════════════════════════════════════════════════════════

class Verdict(StrEnum):
    """5-tier graduated response. Priority: HALT > BLOCK > REQUIRE_APPROVAL > CONSTRAIN > ALLOW"""

    ALLOW = "allow"
    CONSTRAIN = "constrain"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK = "block"
    HALT = "halt"

    @classmethod
    def from_string(cls, value: str | None) -> Verdict:
        """Parse verdict string with v1.0 compat aliases (continue→ALLOW, stop→HALT)."""
        if value is None:
            return cls.ALLOW
        normalized = value.lower().replace("-", "_")
        if normalized == "continue":
            return cls.ALLOW
        if normalized == "stop":
            return cls.HALT
        if normalized in ("require_approval", "request_approval"):
            return cls.REQUIRE_APPROVAL
        try:
            return cls(normalized)
        except ValueError:
            return cls.ALLOW

    @property
    def priority(self) -> int:
        """Priority for aggregation: HALT=4, BLOCK=3, REQUIRE_APPROVAL=2, CONSTRAIN=1, ALLOW=0.

        Matches OpenBox Core's Verdict enum (0-indexed, governance.go).
        """
        return {
            Verdict.ALLOW: 0, Verdict.CONSTRAIN: 1, Verdict.REQUIRE_APPROVAL: 2,
            Verdict.BLOCK: 3, Verdict.HALT: 4,
        }[self]

    @classmethod
    def highest_priority(cls, verdicts: list[Verdict]) -> Verdict:
        """Get highest priority verdict from list. Returns ALLOW if empty."""
        return max(verdicts, key=lambda v: v.priority) if verdicts else cls.ALLOW

    def should_stop(self) -> bool:
        """True if BLOCK or HALT."""
        return self in (Verdict.BLOCK, Verdict.HALT)

    def requires_approval(self) -> bool:
        """True if REQUIRE_APPROVAL."""
        return self == Verdict.REQUIRE_APPROVAL


# Module-level aliases for backward compat and convenience
def verdict_from_string(value: str | None) -> Verdict:
    """Parse a verdict string — delegates to Verdict.from_string()."""
    return Verdict.from_string(value)


def verdict_priority(v: Verdict) -> int:
    """Return the numeric priority of a verdict (higher = more restrictive)."""
    return v.priority


def highest_priority_verdict(verdicts: list[Verdict]) -> Verdict:
    """Return the highest-priority (most restrictive) verdict from a list."""
    return Verdict.highest_priority(verdicts)


def verdict_should_stop(v: Verdict) -> bool:
    """Return True if the verdict requires stopping execution immediately."""
    return v.should_stop()


def verdict_requires_approval(v: Verdict) -> bool:
    """Return True if the verdict requires human approval before continuing."""
    return v.requires_approval()


# ═══════════════════════════════════════════════════════════════════
# Workflow Span Buffer (used by WorkflowSpanProcessor)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class WorkflowSpanBuffer:
    """Buffer for workflow governance state (used by WorkflowSpanProcessor)."""

    workflow_id: str
    run_id: str = ""
    workflow_type: str = ""
    verdict: Verdict | None = None
    verdict_reason: str | None = None


# ═══════════════════════════════════════════════════════════════════
# LangGraph v2 stream event
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# WorkflowEventType (server-side event labels)
# ═══════════════════════════════════════════════════════════════════

class WorkflowEventType(StrEnum):
    """Workflow lifecycle events for governance (matches OpenBox Core wire format)."""

    WORKFLOW_STARTED = "WorkflowStarted"
    WORKFLOW_COMPLETED = "WorkflowCompleted"
    WORKFLOW_FAILED = "WorkflowFailed"
    SIGNAL_RECEIVED = "SignalReceived"
    ACTIVITY_STARTED = "ActivityStarted"
    ACTIVITY_COMPLETED = "ActivityCompleted"


LangGraphEventType = Literal[
    "on_chain_start",
    "on_chain_end",
    "on_chain_stream",
    "on_chat_model_start",
    "on_chat_model_end",
    "on_chat_model_stream",
    "on_tool_start",
    "on_tool_end",
    "on_retriever_start",
    "on_retriever_end",
]

LangChainEventType = Literal[
    "ChainStarted",
    "ChainCompleted",
    "ChainFailed",
    "ToolStarted",
    "ToolCompleted",
    "ToolFailed",
    "LLMStarted",
    "LLMCompleted",
    "LLMFailed",
    "AgentAction",
    "AgentFinish",
    "RetrieverStarted",
    "RetrieverCompleted",
    "RetrieverFailed",
]

ServerEventType = Literal[
    "WorkflowStarted",
    "WorkflowCompleted",
    "WorkflowFailed",
    "SignalReceived",
    "ActivityStarted",
    "ActivityCompleted",
]


@dataclass
class LangGraphStreamEvent:
    """Raw LangGraph v2 streaming event from `.astream_events(version='v2')`."""

    event: str
    name: str
    run_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    parent_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LangGraphStreamEvent:
        """Construct from a raw dict emitted by LangGraph's event stream."""
        return cls(
            event=d.get("event", ""),
            name=d.get("name", ""),
            run_id=d.get("run_id", ""),
            metadata=d.get("metadata") or {},
            data=d.get("data") or {},
            tags=d.get("tags") or [],
            parent_ids=d.get("parent_ids") or [],
        )


def lang_graph_event_to_server_type(event_type: str) -> ServerEventType | None:
    """Map a LangGraph v2 event type to the server-accepted Temporal equivalent."""
    mapping: dict[str, ServerEventType] = {
        "on_chain_start": "WorkflowStarted",
        "on_chain_end": "WorkflowCompleted",
        "on_chat_model_start": "ActivityStarted",
        "on_chat_model_end": "ActivityCompleted",
        "on_tool_start": "ActivityStarted",
        "on_tool_end": "ActivityCompleted",
        "on_retriever_start": "ActivityStarted",
        "on_retriever_end": "ActivityCompleted",
    }
    return mapping.get(event_type)


def to_server_event_type(t: str) -> ServerEventType:
    """Map a LangChain SDK-internal event type to the server-accepted Temporal equivalent."""
    mapping: dict[str, ServerEventType] = {
        "WorkflowStarted": "WorkflowStarted",
        "WorkflowCompleted": "WorkflowCompleted",
        "WorkflowFailed": "WorkflowFailed",
        "SignalReceived": "SignalReceived",
        "ChainStarted": "WorkflowStarted",
        "ChainCompleted": "WorkflowCompleted",
        "ChainFailed": "WorkflowFailed",
        "ToolStarted": "ActivityStarted",
        "LLMStarted": "ActivityStarted",
        "AgentAction": "ActivityStarted",
        "RetrieverStarted": "ActivityStarted",
        "ToolCompleted": "ActivityCompleted",
        "ToolFailed": "ActivityCompleted",
        "LLMCompleted": "ActivityCompleted",
        "LLMFailed": "ActivityCompleted",
        "AgentFinish": "ActivityCompleted",
        "RetrieverCompleted": "ActivityCompleted",
        "RetrieverFailed": "ActivityCompleted",
    }
    return mapping.get(t, "ActivityCompleted")


# ═══════════════════════════════════════════════════════════════════
# Governance Event Payload (sent to OpenBox Core)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SpanData:
    """HTTP span captured during an activity execution."""

    span_id: str
    name: str
    trace_id: str | None = None
    parent_span_id: str | None = None
    kind: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    duration_ns: int | None = None
    attributes: dict[str, Any] | None = None
    status: dict[str, str] | None = None
    request_body: str | None = None
    response_body: str | None = None
    request_headers: dict[str, str] | None = None
    response_headers: dict[str, str] | None = None


@dataclass
class LangChainGovernanceEvent:
    """Governance event payload sent to OpenBox Core `/api/v1/governance/evaluate`."""

    source: Literal["workflow-telemetry"]
    event_type: str
    workflow_id: str
    run_id: str
    workflow_type: str
    task_queue: str
    timestamp: str

    # LangGraph routing
    langgraph_node: str | None = None
    langgraph_step: int | None = None
    subagent_name: str | None = None

    # Activity fields
    activity_id: str | None = None
    activity_type: str | None = None
    activity_input: list[Any] | None = None
    activity_output: Any = None
    workflow_output: Any = None
    spans: list[Any] | None = None
    span_count: int | None = None
    status: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    duration_ms: float | None = None
    error: dict[str, Any] | None = None

    # Hook trigger flag — True when event is triggered by a new span (e.g., outbound HTTP call)
    hook_trigger: bool = False

    # LLM/tool extensions
    llm_model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    has_tool_calls: bool | None = None
    finish_reason: str | None = None
    prompt: str | None = None
    completion: str | None = None
    tool_name: str | None = None
    tool_type: str | None = None
    tool_input: Any = None
    parent_run_id: str | None = None
    session_id: str | None = None
    attempt: int | None = None

    # Signal fields (SignalReceived events)
    signal_name: str | None = None
    signal_args: list[Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for the HTTP request body, omitting None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ═══════════════════════════════════════════════════════════════════
# Governance Response (from OpenBox Core)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GuardrailsReason:
    """A single guardrails violation reason."""

    type: str
    field: str
    reason: str


@dataclass
class GuardrailsResult:
    """Guardrails evaluation result embedded in a governance verdict."""

    input_type: Literal["activity_input", "activity_output"]
    redacted_input: Any
    validation_passed: bool
    reasons: list[GuardrailsReason] = field(default_factory=list)
    raw_logs: dict[str, Any] | None = None


@dataclass
class GovernanceVerdictResponse:
    """Response from governance API evaluation."""

    verdict: Verdict
    reason: str | None = None
    # v1.0 fields (kept for compatibility)
    policy_id: str | None = None
    risk_score: float = 0.0
    metadata: dict[str, Any] | None = None
    governance_event_id: str | None = None
    guardrails_result: GuardrailsResult | None = None
    # v1.1 fields
    approval_id: str | None = None
    approval_expiration_time: str | None = None
    trust_tier: str | None = None
    alignment_score: float | None = None
    behavioral_violations: list[str] | None = None
    constraints: list[Any] | None = None

    @property
    def action(self) -> str:
        """Backward compat: return action string from verdict."""
        if self.verdict == Verdict.ALLOW:
            return "continue"
        if self.verdict == Verdict.HALT:
            return "stop"
        if self.verdict == Verdict.REQUIRE_APPROVAL:
            return "require-approval"
        return self.verdict.value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GovernanceVerdictResponse:
        """Parse governance response from JSON dict (v1.0 and v1.1 compatible)."""
        guardrails_result: GuardrailsResult | None = None
        if gr := data.get("guardrails_result"):
            reasons = [
                GuardrailsReason(
                    type=r.get("type", ""),
                    field=r.get("field", ""),
                    reason=r.get("reason", ""),
                )
                for r in (gr.get("reasons") or [])
            ]
            guardrails_result = GuardrailsResult(
                input_type=gr.get("input_type", "activity_input"),
                redacted_input=gr.get("redacted_input"),
                validation_passed=gr.get("validation_passed", True) is not False,
                reasons=reasons,
                raw_logs=gr.get("raw_logs"),
            )

        verdict = Verdict.from_string(data.get("verdict") or data.get("action", "continue"))

        return cls(
            verdict=verdict,
            reason=data.get("reason"),
            policy_id=data.get("policy_id"),
            risk_score=data.get("risk_score", 0.0),
            metadata=data.get("metadata"),
            governance_event_id=data.get("governance_event_id"),
            guardrails_result=guardrails_result,
            approval_id=data.get("approval_id"),
            approval_expiration_time=data.get("approval_expiration_time"),
            trust_tier=data.get("trust_tier"),
            alignment_score=data.get("alignment_score"),
            behavioral_violations=data.get("behavioral_violations"),
            constraints=data.get("constraints"),
        )


def parse_governance_response(data: dict[str, Any]) -> GovernanceVerdictResponse:
    """Parse a raw dict from OpenBox Core into a `GovernanceVerdictResponse`."""
    return GovernanceVerdictResponse.from_dict(data)


# ═══════════════════════════════════════════════════════════════════
# Approval Response
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ApprovalResponse:
    """HITL approval poll response from `/api/v1/governance/approval`."""

    verdict: Verdict
    reason: str | None = None
    approval_expiration_time: str | None = None
    expired: bool = False


def parse_approval_response(data: dict[str, Any]) -> ApprovalResponse:
    """Parse a raw dict from the approval endpoint into an `ApprovalResponse`."""
    return ApprovalResponse(
        verdict=Verdict.from_string(data.get("verdict") or data.get("action")),
        reason=data.get("reason"),
        approval_expiration_time=data.get("approval_expiration_time"),
        expired=bool(data.get("expired", False)),
    )


# ═══════════════════════════════════════════════════════════════════
# HITL Config
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HITLConfig:
    """Human-in-the-loop polling configuration."""

    enabled: bool = True
    poll_interval_ms: int = 5_000
    skip_tool_types: set[str] = field(default_factory=set)


DEFAULT_HITL_CONFIG = HITLConfig()


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════

def rfc3339_now() -> str:
    """Return the current UTC time as an RFC3339 string."""
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def safe_serialize(value: Any) -> Any:
    """Safely serialize a value for inclusion in a governance event payload.

    Handles LangChain message objects and other non-serializable types by
    converting them to their string/dict representation.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: safe_serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_serialize(v) for v in value]
    # LangChain message objects expose a .dict() or .model_dump() method
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return value.dict()
        except Exception:
            pass
    return str(value)
