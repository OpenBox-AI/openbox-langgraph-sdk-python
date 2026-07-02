"""Layer 3 helper: drive a real compiled LangGraph graph through the real
`OpenBoxLangGraphHandler` and record the ordered governance events + full
serialized bodies it emits.

Separated from capture_harness.py (Layer 1/2 wire-body + header capture from
hand-built events) and fake_agent_graphs.py (graph construction) to keep each
module focused and under the project's file-size guideline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.langgraph_handler import (
    OpenBoxLangGraphHandler,
    OpenBoxLangGraphHandlerOptions,
)
from openbox_langgraph.types import (
    GovernanceVerdictResponse,
    LangChainGovernanceEvent,
    Verdict,
    to_server_event_type,
)


def _wire_body(event: LangChainGovernanceEvent) -> dict[str, Any]:
    """Rebuild the EXACT payload dict `GovernanceClient.evaluate_event` sends.

    Mirrors client.py's evaluate_event 3-line transform (server event_type
    mapping + task_queue/source injection) verbatim, so a captured body is
    byte-for-byte what actually went on the wire — not just the pre-mapping
    `event.to_dict()`. Kept here (not imported from client.py) because that
    transform is a private implementation detail inlined in evaluate_event,
    not a reusable function.
    """
    payload = event.to_dict()
    payload["event_type"] = to_server_event_type(event.event_type)
    payload["task_queue"] = event.task_queue or "langgraph"
    payload["source"] = "workflow-telemetry"
    return payload


@dataclass
class OrderedCapture:
    """Records every real `evaluate_event` call the handler makes, in order.

    `entries` keeps the lightweight (event_type, activity_id) view used by the
    Layer 3 ordering fixtures — `event_type` here is the SDK-INTERNAL label
    (e.g. "LLMStarted", "ChainCompleted") so distinct lifecycle events stay
    distinguishable (several internal labels collapse to the same server
    label, e.g. both LLMStarted and ToolStarted map to "ActivityStarted").
    `bodies` keeps the actual WIRE body (`_wire_body`: server-mapped
    event_type + task_queue/source) at the same index — the real bytes the
    handler would have sent — used by Layer 1 to pin the handler's own event
    construction rather than a hand-authored stand-in. Lookups below take the
    SDK-internal label (unambiguous) and return the corresponding wire body.
    """

    entries: list[dict[str, str | None]] = field(default_factory=list)
    bodies: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event: LangChainGovernanceEvent) -> None:
        self.entries.append({"event_type": event.event_type, "activity_id": event.activity_id})
        self.bodies.append(_wire_body(event))

    def first_body_for_internal_type(self, internal_event_type: str) -> dict[str, Any]:
        """Return the wire body whose SOURCE event had this SDK-internal type.

        e.g. `first_body_for_internal_type("ChainCompleted")` returns the wire
        body for the root close event even though its wire `event_type` field
        reads "WorkflowCompleted" after server mapping — disambiguated via the
        internal label recorded alongside it in `entries`, not the wire label.

        Raises if none was captured — a golden fixture generator asking for
        an event type the run never produced is a wiring bug, not something
        to silently skip.
        """
        for entry, body in zip(self.entries, self.bodies, strict=True):
            if entry["event_type"] == internal_event_type:
                return body
        msg = (
            f"no captured event with internal event_type={internal_event_type!r} "
            f"(captured: {self.entries})"
        )
        raise AssertionError(msg)

    def first_body_where(self, predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        """Return the first captured WIRE body matching `predicate`.

        Use for disambiguating within a single internal event_type by another
        field (e.g. tool_name or subagent_name), since `predicate` sees the
        wire body — whose `event_type` is already server-mapped.
        """
        for body in self.bodies:
            if predicate(body):
                return body
        msg = f"no captured body matched predicate (captured: {self.entries})"
        raise AssertionError(msg)


# A verdict override lets a scenario force a specific response for a specific
# event — e.g. HALT on the LLMStarted pre-screen to trigger the handler's
# error-close WorkflowCompleted path (langgraph_handler.py _pre_screen_input).
VerdictOverride = Callable[[LangChainGovernanceEvent], GovernanceVerdictResponse | None]


class RecordingGovernanceClient(GovernanceClient):
    """`GovernanceClient` subclass that records every event before verdicting it.

    Used as the handler's injected `client=` seam (`OpenBoxLangGraphHandlerOptions.client`)
    so no real network calls occur and every governance send the handler makes is captured
    in order, without needing to fake the OTel/hook-transport plumbing. Defaults to ALLOW;
    pass `verdict_override` to force a different verdict for specific events (e.g. to
    exercise the handler's own error-close / enforcement-failure code paths for real).
    """

    def __init__(
        self, capture: OrderedCapture, *, verdict_override: VerdictOverride | None = None
    ) -> None:
        super().__init__(api_url="https://core.openbox.ai", api_key="obx_test_abc")
        self._capture = capture
        self._verdict_override = verdict_override

    async def evaluate_event(
        self, event: LangChainGovernanceEvent
    ) -> GovernanceVerdictResponse | None:
        self._capture.record(event)
        if self._verdict_override is not None:
            overridden = self._verdict_override(event)
            if overridden is not None:
                return overridden
        return GovernanceVerdictResponse(verdict=Verdict.ALLOW)


async def run_ordered_capture(
    compiled_graph: Any, *, thread_id: str, resolve_subagent_name: Any = None
) -> list[dict[str, str | None]]:
    """Wrap `compiled_graph` in the real handler and return its ordered event capture.

    `resolve_subagent_name`, when given, is forwarded to
    `OpenBoxLangGraphHandlerOptions` unchanged (see subagent ToolStarted capture).
    """
    capture, _client = await run_captured_ainvoke(
        compiled_graph, thread_id=thread_id, resolve_subagent_name=resolve_subagent_name
    )
    return capture.entries


async def run_captured_ainvoke(
    compiled_graph: Any,
    *,
    thread_id: str,
    resolve_subagent_name: Any = None,
    verdict_override: VerdictOverride | None = None,
    expect_raise: type[BaseException] | None = None,
) -> tuple[OrderedCapture, RecordingGovernanceClient]:
    """Run the real handler.ainvoke() and return the full (ordering + body) capture.

    Args:
        verdict_override: Forwarded to `RecordingGovernanceClient` — lets a
            scenario force a non-ALLOW verdict on a specific event to exercise
            a real error/enforcement code path in the handler.
        expect_raise: If given, `ainvoke` is expected to raise this exception
            type (e.g. GovernanceHaltError from an error-close scenario) —
            the exception is caught and swallowed so the caller can inspect
            what was captured up to and including the error-close event.
    """
    capture = OrderedCapture()
    client = RecordingGovernanceClient(capture, verdict_override=verdict_override)
    handler = OpenBoxLangGraphHandler(
        compiled_graph,
        OpenBoxLangGraphHandlerOptions(
            client=client,
            resolve_subagent_name=resolve_subagent_name,
        ),
    )
    coro = handler.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": thread_id}},
    )
    if expect_raise is not None:
        try:
            await coro
        except expect_raise:
            pass
        else:
            msg = f"expected {expect_raise.__name__} to be raised, but ainvoke succeeded"
            raise AssertionError(msg)
    else:
        await coro
    return capture, client
