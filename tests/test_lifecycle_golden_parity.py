"""Level-2 wire parity for the gate-routed lifecycle path.

Two guarantees, both end-to-end through the REAL base `GovernanceGate` (only the
outbound HTTP transport is mocked, to capture the exact bytes):

1. **No silent drops.** Every lifecycle event the handler emits must survive the
   gate's strict validation and actually POST. If a mapped envelope failed
   ``validate_lifecycle`` the gate would raise ``ContractError`` → ``_gate_evaluate``
   returns ``None`` → the event is dropped and Core never sees it (ungoverned).
   This test fails loudly if that happens — across the single-LLM, tool-call,
   two-LLM, subagent, and error-close scenarios, which together exercise every
   lifecycle event type the SDK emits (Signal/WorkflowStarted/LLMStarted/
   LLMCompleted/Tool*/subagent Tool*/root-close WorkflowCompleted/error-close).
2. **Byte parity minus enumerated compat-noise.** The gate wire body must equal
   the pre-migration wire body (``_wire_body``) minus exactly ``hook_trigger`` /
   ``spans`` / ``span_count`` — the three deltas the base envelope drops by
   construction.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from openbox_core.client import EvaluationClient
from openbox_core.config import OpenBoxConfig
from openbox_core.context import ContextStore
from openbox_core.runtime import OpenBoxRuntime

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.langgraph_handler import (
    OpenBoxLangGraphHandler,
    OpenBoxLangGraphHandlerOptions,
)
from openbox_langgraph.types import GovernanceVerdictResponse, LangChainGovernanceEvent, Verdict
from tests.golden.fake_agent_graphs import (
    build_error_graph,
    build_single_llm_graph,
    build_tool_call_graph,
    build_two_llm_call_graph,
)
from tests.golden.graph_capture_harness import _wire_body
from tests.golden.id_normalization import normalize_json

_URL = "https://core.openbox.ai"
_KEY = "obx_test_abc"
_ALLOWED_DELTA_KEYS = frozenset({"hook_trigger", "spans", "span_count"})

_Resolver = Callable[[object], str | None] | None


class _EventCapture(GovernanceClient):
    """Injected client that records the real events the handler emits."""

    def __init__(self) -> None:
        super().__init__(api_url=_URL, api_key=_KEY)
        self.events: list[LangChainGovernanceEvent] = []

    async def evaluate_event(
        self, event: LangChainGovernanceEvent
    ) -> GovernanceVerdictResponse | None:
        self.events.append(event)
        return GovernanceVerdictResponse(verdict=Verdict.ALLOW)


async def _capture_events(
    graph: object, thread_id: str, resolver: _Resolver = None
) -> list[LangChainGovernanceEvent]:
    client = _EventCapture()
    handler = OpenBoxLangGraphHandler(
        graph, OpenBoxLangGraphHandlerOptions(client=client, resolve_subagent_name=resolver)
    )
    try:
        await handler.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception:
        # The error-close scenario raises after emitting its events; every event
        # captured up to the raise is still parity-checked below.
        pass
    return client.events


def _gate_client() -> tuple[GovernanceClient, list[dict[str, object]]]:
    """A GovernanceClient wired to a real gate whose transport records bodies."""
    posted: list[dict[str, object]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        posted.append(json.loads(request.content))
        return httpx.Response(200, json={"verdict": "allow"})

    config = OpenBoxConfig.resolve(api_url=_URL, api_key=_KEY, validate=True)
    base_client = EvaluationClient(_URL, _KEY, async_transport=httpx.MockTransport(_handler))
    runtime = OpenBoxRuntime(config, client=base_client, context_store=ContextStore())
    # Fresh client per event keeps the de-dup store from suppressing a replay.
    return GovernanceClient(api_url=_URL, api_key=_KEY, gate=runtime.gate), posted


def _strip_deltas(body: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in body.items() if k not in _ALLOWED_DELTA_KEYS}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("build", "thread_id", "resolver"),
    [
        (lambda: build_single_llm_graph([AIMessage(content="hi")]), "parity-single", None),
        (build_tool_call_graph, "parity-tool", None),
        (build_two_llm_call_graph, "parity-two-llm", None),
        (build_tool_call_graph, "parity-subagent", lambda _e: "researcher"),
        (build_error_graph, "parity-error", None),
    ],
)
async def test_gate_wire_parity_and_no_drops(
    build: Callable[[], object], thread_id: str, resolver: _Resolver
) -> None:
    events = await _capture_events(build(), thread_id, resolver)
    assert events, "the graph emitted no governance events"

    for event in events:
        client, posted = _gate_client()
        before = len(posted)
        await client.evaluate_event(event)
        await client.close()

        assert len(posted) == before + 1, (
            f"{event.event_type} (activity_id={event.activity_id}) was DROPPED by the gate "
            f"(no POST) — a strict-validation ContractError would silently disable governance"
        )
        gate_body = normalize_json(posted[-1])
        legacy_body = normalize_json(_strip_deltas(_wire_body(event)))
        assert gate_body == legacy_body, (
            f"wire drift for {event.event_type} beyond the allowed compat-noise deltas:\n"
            f"  gate:   {gate_body}\n  legacy: {legacy_body}"
        )


def test_empty_task_queue_defaults_to_langgraph() -> None:
    """A falsy task_queue must serialize as "langgraph" (legacy parity), not "" —
    otherwise it would be a 4th, undocumented wire delta."""
    from openbox_langgraph.core_events import to_envelope

    event = LangChainGovernanceEvent(
        source="workflow-telemetry",
        event_type="ToolStarted",
        workflow_id="w",
        run_id="r",
        workflow_type="A",
        task_queue="",
        timestamp="2026-07-02T00:00:00Z",
        activity_id="a1",
        activity_type="tool_call",
    )
    assert to_envelope(event).to_payload_dict()["task_queue"] == "langgraph"
