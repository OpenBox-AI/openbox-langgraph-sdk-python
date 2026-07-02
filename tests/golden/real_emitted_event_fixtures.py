"""Layer 1 wire-body goldens captured from the REAL handler, not hand-built.

A hand-built `LangChainGovernanceEvent` only pins the client's serialization
of a dict a test wrote by hand — it says nothing about whether the HANDLER
still builds that dict the same way. Since a refactor changes exactly that
construction, every event type a real unit run can produce is captured here
by driving `OpenBoxLangGraphHandler.ainvoke` for real (fake chat model only,
no network) and pulling the matching body out of what it actually sent.

Four scenarios cover every real-emittable lifecycle event:
- baseline: single-turn conversation, no tools — Signal/WorkflowStarted/
  LLMStarted/LLMCompleted/ChainCompleted.
- tool: a tool-call turn, no subagent resolver — plain ToolStarted/ToolCompleted.
- subagent: the SAME tool-call graph, WITH `resolve_subagent_name` set —
  subagent-labelled ToolStarted/ToolCompleted. Run separately from "tool"
  because a resolver that always returns a name would make every tool call
  in one run subagent-labelled, leaving no plain variant to capture.
- error: a verdict override forces HALT on the LLMStarted pre-screen, which
  triggers the handler's own error-close WorkflowCompleted(status="failed")
  path (langgraph_handler.py _pre_screen_input, ~line 573-587).

WorkflowFailed has no real construction site anywhere in the package (grep
confirms it) — it is hand-built and clearly labelled as a serialization pin
in layer1_handbuilt_pins.py. ChainStarted (root) is unreachable for the same
reason: sending it requires send_chain_start_event=False, but that same flag
also suppresses _process_event's own ChainStarted send — so it stays a
hand-built pin too. Both exceptions are the ones this module's docstring
allows: real emission was attempted and is properly infeasible.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from openbox_langgraph.errors import GovernanceHaltError
from openbox_langgraph.types import GovernanceVerdictResponse, LangChainGovernanceEvent, Verdict

from .capture_harness import write_fixture_pair
from .fake_agent_graphs import build_error_graph, build_single_llm_graph, build_tool_call_graph
from .graph_capture_harness import run_captured_ainvoke


async def capture_real_emitted_baseline_bodies() -> None:
    """Single-turn run: Signal, WorkflowStarted, LLMStarted, LLMCompleted, ChainCompleted.

    Note: a healthy run never sends a literal SDK-internal "WorkflowCompleted"
    event — the root close is "ChainCompleted", which wire-maps to the server
    label "WorkflowCompleted" (to_server_event_type). The literal SDK-internal
    "WorkflowCompleted" only exists on the error-close path (see
    capture_real_emitted_error_close_body) — that is a genuinely different
    event, not a duplicate of this one, so both are captured under distinct
    fixture names.
    """
    graph = build_single_llm_graph([AIMessage(content="hello from fake model")])
    capture, _client = await run_captured_ainvoke(graph, thread_id="golden-baseline-body-run")

    write_fixture_pair("signal_received", capture.first_body_for_internal_type("SignalReceived"))
    write_fixture_pair(
        "workflow_started", capture.first_body_for_internal_type("WorkflowStarted")
    )
    write_fixture_pair("llm_started", capture.first_body_for_internal_type("LLMStarted"))
    write_fixture_pair("llm_completed", capture.first_body_for_internal_type("LLMCompleted"))
    write_fixture_pair("chain_completed", capture.first_body_for_internal_type("ChainCompleted"))


async def capture_real_emitted_tool_bodies() -> None:
    """Plain tool-call run, no subagent resolver: ordinary ToolStarted/ToolCompleted."""
    graph = build_tool_call_graph()
    capture, _client = await run_captured_ainvoke(graph, thread_id="golden-tool-body-run")

    write_fixture_pair(
        "tool_started",
        capture.first_body_where(
            lambda b: b.get("event_type") == "ActivityStarted" and b.get("tool_name") == "echo_tool"
        ),
    )
    write_fixture_pair(
        "tool_completed",
        capture.first_body_where(
            lambda b: b.get("event_type") == "ActivityCompleted"
            and b.get("tool_name") == "echo_tool"
        ),
    )


async def capture_real_emitted_subagent_tool_bodies() -> None:
    """Same tool-call graph, WITH a subagent resolver: subagent-labelled Tool bodies."""

    def resolve_subagent(event: object) -> str | None:
        return "writer" if getattr(event, "name", None) == "echo_tool" else None

    graph = build_tool_call_graph()
    capture, _client = await run_captured_ainvoke(
        graph, thread_id="golden-subagent-body-run", resolve_subagent_name=resolve_subagent
    )

    write_fixture_pair(
        "subagent_tool_started",
        capture.first_body_where(
            lambda b: b.get("subagent_name") == "writer"
            and b.get("event_type") == "ActivityStarted"
        ),
    )
    write_fixture_pair(
        "subagent_tool_completed",
        capture.first_body_where(
            lambda b: b.get("subagent_name") == "writer"
            and b.get("event_type") == "ActivityCompleted"
        ),
    )


async def capture_real_emitted_error_close_body() -> None:
    """Error run: force HALT on the LLMStarted pre-screen -> error-close WorkflowCompleted."""

    def halt_on_pre_screen(event: LangChainGovernanceEvent) -> GovernanceVerdictResponse | None:
        if event.activity_id and event.activity_id.endswith("-pre"):
            return GovernanceVerdictResponse(
                verdict=Verdict.HALT, reason="golden baseline error-close trigger"
            )
        return None

    graph = build_error_graph()
    capture, _client = await run_captured_ainvoke(
        graph,
        thread_id="golden-error-close-body-run",
        verdict_override=halt_on_pre_screen,
        expect_raise=GovernanceHaltError,
    )

    write_fixture_pair(
        "workflow_completed_error_close",
        capture.first_body_where(
            lambda b: b.get("event_type") == "WorkflowCompleted" and b.get("status") == "failed"
        ),
    )
