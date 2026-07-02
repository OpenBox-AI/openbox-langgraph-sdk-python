"""Map a `LangChainGovernanceEvent` onto the base SDK's `EventEnvelope`.

Table-driven translation used ONLY by the gate-routed path in `client.py`
(`GovernanceClient.evaluate_event` when a `gate` is wired). The legacy httpx
transport path never touches this module — it keeps serializing
`LangChainGovernanceEvent.to_dict()` exactly as before.

Wire-parity contract: `to_envelope(event).to_payload_dict()` MUST equal
`event.to_dict()` (plus the `event_type`/`task_queue`/`source` fixups
`evaluate_event` normally applies inline) MINUS exactly three compatibility
deltas the base SDK omits by construction:

    1. `hook_trigger: false` — the base envelope only emits the key when True.
    2. `spans: []` / `span_count: 0` — never produced for lifecycle envelopes.
    3. (transport-level, not this module's concern) the `User-Agent` header.

Every other field must survive unchanged — activity_id/activity_type ride on
the envelope's dedicated slots; every remaining populated field goes into
`payload` via `extra` so Core sees the exact same key/value it does today.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from openbox_core.contracts.events import EventEnvelope, EventType, activity_completed

from openbox_langgraph.types import LangChainGovernanceEvent, to_server_event_type

__all__ = ["to_envelope"]

# Envelope fields carried on `EventEnvelope`'s own dedicated slots — never
# duplicated into `payload` (they would either be redundant with the slot or,
# for `event_type`/`hook_trigger`/`spans`/`span_count`, actively wrong to
# forward: those four are exactly the fields this migration phase must not
# let leak from the legacy dataclass onto the wire unchanged).
_ENVELOPE_OWNED_FIELDS = frozenset(
    {
        "source",
        "event_type",
        "activity_id",
        "activity_type",
        "timestamp",
        "hook_trigger",
        "spans",
        "span_count",
    }
)

# `to_server_event_type` wire-maps every internal label to one of these five
# lifecycle types; `Handoff` has no LangGraph-internal source label and is
# never produced by this SDK today.
_WIRE_TO_CORE_EVENT_TYPE: dict[str, EventType] = {
    "WorkflowStarted": EventType.WORKFLOW_STARTED,
    "WorkflowCompleted": EventType.WORKFLOW_COMPLETED,
    "WorkflowFailed": EventType.WORKFLOW_FAILED,
    "SignalReceived": EventType.SIGNAL_RECEIVED,
    "ActivityStarted": EventType.ACTIVITY_STARTED,
    "ActivityCompleted": EventType.ACTIVITY_COMPLETED,
}

# SDK-internal event_type labels whose wire type is ActivityCompleted AND
# which represent a plain activity close — routed through the base SDK's
# `activity_completed()` factory rather than the generic envelope construction
# below. LLMCompleted in particular MUST use this path: `activity_completed()`
# builds a plain lifecycle ActivityCompleted (no hook_trigger, no spans) and
# can never accidentally become a `hook()` envelope, which is reserved for
# span-bearing evaluations and would fail the strict gate's
# `ACTIVITY_COMPLETED_WITH_SPANS` / `HOOK_TRIGGER_FALSE` checks (event_rules.py)
# or — worse — silently change Core's interpretation of the event.
_ACTIVITY_COMPLETED_LABELS = frozenset(
    {"LLMCompleted", "ToolCompleted", "ToolFailed", "AgentFinish", "RetrieverCompleted",
     "RetrieverFailed"}
)


def to_envelope(event: LangChainGovernanceEvent) -> EventEnvelope:
    """Build a base `EventEnvelope` that wire-serializes to `event`'s body.

    `to_payload_dict()` on the result equals `event.to_dict()` (with the same
    `event_type`/`task_queue`/`source` normalization `evaluate_event` already
    applies) minus the three allowed compatibility deltas documented above.

    Raises:
        ValueError: if `event.event_type` does not map to a known wire type
            (defensive — `to_server_event_type` always returns a valid label
            today, so this should be unreachable in practice).
    """
    wire_label = to_server_event_type(event.event_type)
    core_type = _WIRE_TO_CORE_EVENT_TYPE.get(wire_label)
    if core_type is None:  # pragma: no cover — defensive, see docstring
        msg = f"no base EventType mapping for wire event_type {wire_label!r}"
        raise ValueError(msg)

    extra = dict(_extra_payload(event))
    # Match legacy evaluate_event: a falsy task_queue defaults to "langgraph",
    # so an explicit empty task_queue never becomes an unintended wire delta.
    extra["task_queue"] = event.task_queue or "langgraph"

    if wire_label in ("ActivityCompleted",) and event.event_type in _ACTIVITY_COMPLETED_LABELS:
        # Plain ActivityCompleted via the dedicated factory — guarantees
        # hook_trigger=False and spans=() by construction, never a hook().
        return activity_completed(
            workflow_id=event.workflow_id,
            run_id=event.run_id,
            workflow_type=event.workflow_type,
            activity_id=event.activity_id or "",
            activity_type=event.activity_type or "",
            task_queue=event.task_queue,
            timestamp=event.timestamp,
            extra=extra,
        )

    return EventEnvelope(
        event_type=core_type,
        payload=extra,
        activity_id=event.activity_id,
        activity_type=event.activity_type,
        timestamp=event.timestamp,
    )


def _extra_payload(event: LangChainGovernanceEvent) -> Mapping[str, Any]:
    """Every populated field NOT owned by the envelope's dedicated slots.

    Mirrors `LangChainGovernanceEvent.to_dict()`'s own None-filtering exactly
    — `EventEnvelope.to_payload_dict()`/the gate's `_finalize_payload` use
    `exclude_none=False` (spans need explicit `null` survival), so a None
    slipped in here would emit as a literal `null` key Core never saw before.
    `workflow_id`/`run_id`/`workflow_type`/`task_queue` are required lifecycle
    fields (never envelope-owned) so they always land in payload, matching
    the flat top-level shape `to_dict()` already produces for them.
    """
    return {
        f.name: value
        for f in fields(event)
        if f.name not in _ENVELOPE_OWNED_FIELDS
        and (value := getattr(event, f.name)) is not None
    }
