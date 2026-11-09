"""One agent session per matter or tenant (PROD-1109).

The adapter takes the run's scope from its input (or config, or the thread's
saved state) and puts it on every governance event of the run, so core keeps
one exposure history per matter instead of one per run.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

import httpx
import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.langgraph_handler import (
    OpenBoxLangGraphHandler,
    OpenBoxLangGraphHandlerOptions,
)
from openbox_langgraph.run_scope import scope_namespace_for, scope_value

exposure = pytest.importorskip(
    "openbox_core.exposure", reason="openbox-sdk-python without exposure fields"
)


def test_scope_value_and_namespace() -> None:
    assert scope_namespace_for("matter_id") == "matter"
    assert scope_namespace_for("tenant") == "tenant"
    assert scope_namespace_for("_id") == "_id"
    assert scope_value(" 0417 ", "matter") == "matter:0417"
    assert scope_value(417, "matter") == "matter:417"
    assert scope_value("tenant:acme", "matter") == "tenant:acme"
    for empty in (None, "", "  ", True, {"x": 1}):
        assert scope_value(empty, "matter") is None


class _Docs(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [Document("memo", metadata={"source": "dms://doc/1", "client": "coke"})]


class _State(TypedDict, total=False):
    matter_id: str
    query: str
    docs: list[str]


def _graph(checkpointer: Any = None) -> Any:
    retriever = _Docs()

    async def retrieve(state: _State, config: RunnableConfig) -> _State:
        found = await retriever.ainvoke(state.get("query") or "q", config=config)
        return {"docs": [d.page_content for d in found]}

    g = StateGraph(_State)
    g.add_node("retrieve", retrieve)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", END)
    return g.compile(checkpointer=checkpointer)


def _client(sent: list[dict]) -> GovernanceClient:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"verdict": "allow"})

    client = GovernanceClient(api_url="https://core.test", api_key="obx_test_abc")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client._sync_client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def _handler(sent: list[dict], checkpointer: Any = None, **opts: Any) -> OpenBoxLangGraphHandler:
    return OpenBoxLangGraphHandler(
        _graph(checkpointer),
        OpenBoxLangGraphHandlerOptions(
            client=_client(sent), label_map={"client": "client"}, **opts
        ),
    )


def _cfg(thread: str, **configurable: Any) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread, **configurable}}


def _scopes(sent: list[dict]) -> set[Any]:
    assert sent, "no governance events were sent"
    return {body.get("scope") for body in sent}


async def test_every_event_of_a_run_carries_the_matter_from_the_input() -> None:
    sent: list[dict] = []
    await _handler(sent, scope_key="matter_id").ainvoke(
        {"matter_id": "0417", "query": "q"}, config=_cfg("t1")
    )
    assert _scopes(sent) == {"matter:0417"}
    assert any(b.get("action_class") == "retrieve" for b in sent)
    # Nothing leaks past the run.
    assert "scope" not in exposure.current_exposure_fields()


async def test_two_matters_are_two_sessions() -> None:
    sent: list[dict] = []
    handler = _handler(sent, scope_key="matter_id")
    await handler.ainvoke({"matter_id": "0417"}, config=_cfg("t1"))
    first = len(sent)
    await handler.ainvoke({"matter_id": "0999"}, config=_cfg("t2"))
    assert _scopes(sent[:first]) == {"matter:0417"}
    assert _scopes(sent[first:]) == {"matter:0999"}


async def test_scope_from_config_and_custom_namespace() -> None:
    sent: list[dict] = []
    await _handler(sent, scope_key="tenant_id", scope_namespace="tenant").ainvoke(
        {"query": "q"}, config=_cfg("t1", tenant_id="acme")
    )
    assert _scopes(sent) == {"tenant:acme"}


async def test_a_follow_up_turn_keeps_the_thread_matter() -> None:
    sent: list[dict] = []
    handler = _handler(sent, checkpointer=MemorySaver(), scope_key="matter_id")
    await handler.ainvoke({"matter_id": "0417", "query": "first"}, config=_cfg("t1"))
    first = len(sent)
    # The second turn does not repeat the matter; the saved state has it.
    await handler.ainvoke({"query": "second"}, config=_cfg("t1"))
    assert _scopes(sent[first:]) == {"matter:0417"}


async def test_a_scope_the_caller_set_wins() -> None:
    sent: list[dict] = []
    with exposure.exposure_scope("matter:override"):
        await _handler(sent, scope_key="matter_id").ainvoke(
            {"matter_id": "0417"}, config=_cfg("t1")
        )
        assert exposure.current_exposure_fields()["scope"] == "matter:override"
    assert _scopes(sent) == {"matter:override"}


async def test_custom_resolver() -> None:
    sent: list[dict] = []
    await _handler(
        sent, scope_resolver=lambda inp, cfg: f"engagement:{inp['eng']}"
    ).ainvoke({"eng": "e1"}, config=_cfg("t1"))
    assert _scopes(sent) == {"engagement:e1"}


async def test_streaming_is_scoped_and_does_not_leak() -> None:
    sent: list[dict] = []
    handler = _handler(sent, scope_key="matter_id")
    async for _ in handler.astream_governed({"matter_id": "0417"}, config=_cfg("t1")):
        pass
    assert _scopes(sent) == {"matter:0417"}
    assert "scope" not in exposure.current_exposure_fields()


async def test_without_a_scope_key_runs_are_unscoped() -> None:
    sent: list[dict] = []
    await _handler(sent).ainvoke({"matter_id": "0417"}, config=_cfg("t1"))
    assert _scopes(sent) == {None}
