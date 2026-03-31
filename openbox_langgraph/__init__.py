"""
OpenBox LangGraph SDK — Python port of @openbox/langgraph-sdk.

Provides OpenBox governance and observability for any compiled LangGraph graph.

Example:
    >>> from openbox_langgraph import create_openbox_graph_handler
    >>> governed = await create_openbox_graph_handler(
    ...     graph=my_compiled_graph,
    ...     api_url="https://...",
    ...     api_key="obx_live_...",
    ...     agent_name="MyAgent",
    ... )
    >>> result = await governed.ainvoke(
    ...     {"messages": [{"role": "user", "content": "Hello"}]},
    ...     config={"configurable": {"thread_id": "session-abc"}},
    ... )
"""

from openbox_langgraph.client import GovernanceClient, build_auth_headers
from openbox_langgraph.config import (
    GovernanceConfig,
    get_global_config,
    initialize,
    merge_config,
)
from openbox_langgraph.errors import (
    ApprovalExpiredError,
    ApprovalRejectedError,
    ApprovalTimeoutError,
    GovernanceBlockedError,
    GovernanceHaltError,
    GuardrailsValidationError,
    OpenBoxAuthError,
    OpenBoxError,
    OpenBoxInsecureURLError,
    OpenBoxNetworkError,
)
from openbox_langgraph.hitl import poll_until_decision
from openbox_langgraph.langgraph_handler import (
    OpenBoxLangGraphHandler,
    OpenBoxLangGraphHandlerOptions,
    create_openbox_graph_handler,
)

# OTel exports (required dependency)
from openbox_langgraph.otel_setup import setup_opentelemetry_for_governance
from openbox_langgraph.span_processor import WorkflowSpanProcessor
from openbox_langgraph.tracing import create_span, traced
from openbox_langgraph.types import (
    DEFAULT_HITL_CONFIG,
    ApprovalResponse,
    GovernanceVerdictResponse,
    GuardrailsReason,
    GuardrailsResult,
    HITLConfig,
    LangChainGovernanceEvent,
    LangGraphStreamEvent,
    Verdict,
    WorkflowEventType,
    WorkflowSpanBuffer,
    highest_priority_verdict,
    parse_approval_response,
    parse_governance_response,
    rfc3339_now,
    safe_serialize,
    to_server_event_type,
    verdict_from_string,
    verdict_priority,
    verdict_requires_approval,
    verdict_should_stop,
)
from openbox_langgraph.verdict_handler import (
    VerdictContext,
    enforce_verdict,
    is_hitl_applicable,
    lang_graph_event_to_context,
)

__all__ = [
    "DEFAULT_HITL_CONFIG",
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
    "build_auth_headers",
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
    "poll_until_decision",
    "rfc3339_now",
    "safe_serialize",
    "setup_opentelemetry_for_governance",
    "to_server_event_type",
    "traced",
    "verdict_from_string",
    "verdict_priority",
    "verdict_requires_approval",
    "verdict_should_stop",
]
