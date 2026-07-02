"""One-shot generator for the wire-payload golden fixtures under tests/golden/.

Run with:  uv run --extra dev python3 -m tests.golden.generate_fixtures

This script drives the REAL `GovernanceClient` /
`OpenBoxLangGraphHandler` code paths (mock HTTP transport only, plus a
client-injection seam for the full-graph scenarios — see capture_harness.py
/ graph_capture_harness.py) and writes out the exact wire bodies, headers,
and event orderings observed today. The fixtures it produces are the
baseline oracle that tests/test_golden_baseline.py asserts against and that
a future refactor's parity tests will diff against.

Idempotent by construction: every volatile field (timestamps, nonces,
signatures, generated ids) is normalized to a stable placeholder before the
`.normalized.json` / `ordering_*.json` fixtures are written, so running this
twice in a row produces a byte-identical result (verify with
`git diff --stat tests/golden/*.normalized.json tests/golden/ordering_*.json`
after a second run — expect empty output). `.raw.json` fixtures are NOT
normalized and will differ between runs by design — they exist to show a
real example, not to be diffed.

Not itself a pytest test — invoke directly when fixtures need regenerating
(e.g. after an intentional, reviewed wire-format change).
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage

from openbox_langgraph.types import LangChainGovernanceEvent

from .capture_harness import (
    GOLDEN_DIR,
    base_event_kwargs,
    build_recording_client,
    new_run_ids,
    write_fixture_pair,
    write_ordering_fixture,
)
from .fake_agent_graphs import (
    build_single_llm_graph,
    build_tool_call_graph,
    build_two_llm_call_graph,
)
from .graph_capture_harness import run_ordered_capture
from .layer1_handbuilt_pins import capture_handbuilt_pin_bodies
from .real_emitted_event_fixtures import (
    capture_real_emitted_baseline_bodies,
    capture_real_emitted_error_close_body,
    capture_real_emitted_subagent_tool_bodies,
    capture_real_emitted_tool_bodies,
)


async def _capture_layer2_headers() -> None:
    """Header goldens: unsigned vs signed client, key names only (values normalized)."""
    wf, run = new_run_ids()
    base = base_event_kwargs(workflow_id=wf, run_id=run)
    probe_event = LangChainGovernanceEvent(
        event_type="WorkflowStarted",
        activity_id=f"{run}-wf",
        activity_type="GoldenBaselineAgent",
        **base,
    )

    unsigned_client, unsigned_recorder = build_recording_client(signed=False)
    try:
        await unsigned_client.evaluate_event(probe_event)
    finally:
        await unsigned_client.close()
    unsigned_headers = dict(unsigned_recorder.requests[0].headers)

    signed_client, signed_recorder = build_recording_client(signed=True)
    try:
        await signed_client.evaluate_event(probe_event)
    finally:
        await signed_client.close()
    signed_headers = dict(signed_recorder.requests[0].headers)

    write_fixture_pair(
        "headers_unsigned",
        {"present_headers": sorted(unsigned_headers.keys()), "raw_headers": unsigned_headers},
    )
    write_fixture_pair(
        "headers_signed",
        {"present_headers": sorted(signed_headers.keys()), "raw_headers": signed_headers},
    )


async def _capture_layer3_ordering() -> None:
    """Full-graph ordering goldens via the real handler.ainvoke()."""
    two_llm_graph = build_two_llm_call_graph()
    two_llm_turn = await run_ordered_capture(two_llm_graph, thread_id="golden-two-llm-turn")
    write_ordering_fixture("ordering_two_llm_calls_turn", two_llm_turn)

    def resolve_subagent(event: object) -> str | None:
        return "writer" if getattr(event, "name", None) == "echo_tool" else None

    tool_graph = build_tool_call_graph()
    tool_turn = await run_ordered_capture(
        tool_graph, thread_id="golden-tool-turn", resolve_subagent_name=resolve_subagent
    )
    write_ordering_fixture("ordering_tool_call_turn", tool_turn)


async def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    # Layer 1: wire bodies. Real-emitted first (the vast majority), then the
    # verified-infeasible hand-built pins.
    await capture_real_emitted_baseline_bodies()
    await capture_real_emitted_tool_bodies()
    await capture_real_emitted_subagent_tool_bodies()
    await capture_real_emitted_error_close_body()
    await capture_handbuilt_pin_bodies()

    # Layer 2: headers.
    await _capture_layer2_headers()

    # Layer 3: full-graph event ordering. Runs the same shape of scenario as
    # the baseline body capture, on a fresh thread_id, kept independent and
    # readable rather than reusing that capture's internal state.
    single_turn_graph = build_single_llm_graph([AIMessage(content="hello from fake model")])
    single_turn = await run_ordered_capture(single_turn_graph, thread_id="golden-single-turn")
    write_ordering_fixture("ordering_single_llm_turn", single_turn)

    await _capture_layer3_ordering()

    print(f"Wrote golden fixtures to {GOLDEN_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
