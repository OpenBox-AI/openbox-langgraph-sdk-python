"""Hand-built serialization pins for event types the real handler cannot emit.

Every other Layer 1 wire body is captured from a real `handler.ainvoke()` run
(see real_emitted_event_fixtures.py). These two are the deliberate, verified
exceptions — real emission was attempted and found infeasible for each:

- WorkflowFailed: grep across openbox_langgraph/*.py finds ZERO construction
  sites for event_type="WorkflowFailed" or "ChainFailed" (its SDK-internal
  source label). It exists only as an enum value and a to_server_event_type
  mapping target — nothing in the handler ever builds one. Hand-built here so
  the client's serialization of that shape is still pinned somewhere, should
  a future refactor add the missing construction site.
- ChainStarted (root): `_process_event` only sends a root ChainStarted when
  `send_chain_start_event=False` (langgraph_handler.py ~858), but that exact
  same config flag ALSO gates `_pre_screen_input`'s WorkflowStarted send —
  when it's False, `workflow_started_sent=False`, which is the only way
  `_process_event`'s ChainStarted skip-guard passes... except line 857-858
  ALSO returns early when `not self._config.send_chain_start_event`. The two
  guards are mutually exclusive on the same flag: no configuration reaches
  the code that would actually send a root ChainStarted. It is real dead
  code, not just hard to reach in a unit test — hence hand-built here rather
  than skipped.

Both fixtures are written via `write_fixture_pair(..., handbuilt_pin=True)`
so their filenames unambiguously carry the `.handbuilt_serialization_pin`
marker.
"""

from __future__ import annotations

from openbox_langgraph.types import LangChainGovernanceEvent

from .capture_harness import (
    base_event_kwargs,
    capture_single_event,
    new_run_ids,
    write_fixture_pair,
)


async def capture_handbuilt_pin_bodies() -> None:
    """Capture the two event types no real unit run can produce."""
    wf, run = new_run_ids()
    base = base_event_kwargs(workflow_id=wf, run_id=run)

    chain_started = LangChainGovernanceEvent(
        event_type="ChainStarted",
        activity_id="chain-run-1",
        activity_type="should_continue",
        activity_input=[{"messages": []}],
        **base,
    )
    write_fixture_pair(
        "chain_started",
        (await capture_single_event(chain_started)).json_body,
        handbuilt_pin=True,
    )

    workflow_failed = LangChainGovernanceEvent(
        event_type="WorkflowFailed",
        activity_id=f"{run}-wf",
        activity_type="GoldenBaselineAgent",
        status="failed",
        error={"message": "unrecoverable workflow error"},
        **base,
    )
    write_fixture_pair(
        "workflow_failed",
        (await capture_single_event(workflow_failed)).json_body,
        handbuilt_pin=True,
    )
