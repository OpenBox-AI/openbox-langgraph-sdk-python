"""F2 seam: an injected ``client=`` is used verbatim; no gate is interposed.

Users (and both demos) inject a `GovernanceClient` subclass via
`OpenBoxLangGraphHandlerOptions.client`. The lifecycle gate-routing must live
INSIDE `GovernanceClient.evaluate_event`, never at a handler call site — so an
injected client whose own `evaluate_event` is overridden still intercepts every
governance send, and no core runtime/gate is built to shadow it.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from openbox_langgraph.langgraph_handler import (
    OpenBoxLangGraphHandler,
    OpenBoxLangGraphHandlerOptions,
)
from tests.golden.fake_agent_graphs import build_single_llm_graph
from tests.golden.graph_capture_harness import OrderedCapture, RecordingGovernanceClient


def test_injected_client_used_and_no_runtime_built() -> None:
    capture = OrderedCapture()
    client = RecordingGovernanceClient(capture)
    handler = OpenBoxLangGraphHandler(
        build_single_llm_graph([AIMessage(content="hi")]),
        OpenBoxLangGraphHandlerOptions(client=client),
    )
    assert handler._client is client
    # No gate is built when a client is injected — the injected override is the
    # sole governance path (else gate-routing would silently bypass it).
    assert handler._core_runtime is None


@pytest.mark.asyncio
async def test_injected_client_intercepts_every_send() -> None:
    capture = OrderedCapture()
    client = RecordingGovernanceClient(capture)
    handler = OpenBoxLangGraphHandler(
        build_single_llm_graph([AIMessage(content="hi")]),
        OpenBoxLangGraphHandlerOptions(client=client),
    )
    await handler.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "seam-run"}},
    )
    assert capture.entries, "the injected client recorded none of the handler's sends"
    assert handler._core_runtime is None


def test_default_client_path_builds_runtime_and_wires_gate() -> None:
    """No injected client → the handler builds its own core runtime and routes
    the default GovernanceClient through that runtime's gate. Guards the eager
    build: valid global config must construct cleanly (not raise)."""
    from openbox_langgraph.config import initialize

    initialize(api_url="http://localhost:8080", api_key="obx_test_abc", validate=False)
    handler = OpenBoxLangGraphHandler(
        build_single_llm_graph([AIMessage(content="hi")]),
        OpenBoxLangGraphHandlerOptions(),
    )
    assert handler._core_runtime is not None
    assert handler._client._gate is handler._core_runtime.gate
