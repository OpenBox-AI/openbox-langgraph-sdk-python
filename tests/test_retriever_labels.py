"""Labels from retrieved documents go through Core's admission gate (PROD-1109).

A retriever's documents are labelled from their metadata and sent to Core as
the objects a retrieve returned. A refusal raises out of the retriever, so the
documents never reach the next node.
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
from langgraph.graph import END, START, StateGraph

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.errors import GovernanceBlockedError
from openbox_langgraph.langgraph_handler import (
    OpenBoxLangGraphHandler,
    OpenBoxLangGraphHandlerOptions,
)
from openbox_langgraph.retriever_labels import (
    LabelCoverage,
    labels_from_metadata,
    resources_from_documents,
)

exposure = pytest.importorskip(
    "openbox_core.exposure", reason="openbox-sdk-python without exposure fields"
)

LABEL_MAP = {"client": "client", "matter_id": "matter"}


def test_labels_from_metadata() -> None:
    meta = {"client": "coke", "matter_id": ["0417", "0417", " ", "a b", "x*"], "other": "y"}
    assert labels_from_metadata(meta, LABEL_MAP) == ["client:coke", "matter:0417"]
    assert labels_from_metadata({"client": True}, LABEL_MAP) == []
    assert labels_from_metadata({"client": 42}, LABEL_MAP) == ["client:42"]


def test_resources_merge_chunks_and_fall_back_for_uri() -> None:
    docs = [
        Document("a", metadata={"source": "dms://doc/1", "client": "coke"}),
        Document("b", metadata={"source": "dms://doc/1", "matter_id": "0417"}),
        Document("c", id="vec-9", metadata={}),
        Document("d", metadata={}),
    ]
    assert resources_from_documents(docs, retriever="kb", label_map=LABEL_MAP) == [
        ("dms://doc/1", ["client:coke", "matter:0417"]),
        ("vec-9", []),
        ("kb#3", []),
    ]


def test_coverage_counts_unknown_share() -> None:
    c = LabelCoverage(log_every=0)
    c.record("kb", 3, 1)
    c.record("kb", 1, 3)
    assert c.snapshot() == {"kb": {"labelled": 4, "unlabelled": 4, "unknown_share": 0.5}}


# ── Through a real graph ─────────────────────────────────────────


class _Docs(BaseRetriever):
    docs: list[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return list(self.docs)


class _State(TypedDict, total=False):
    query: str
    docs: list[str]
    reached_model: bool


def _graph(retriever: BaseRetriever, *, use_async: bool) -> Any:
    async def retrieve_async(state: _State, config: RunnableConfig) -> _State:
        found = await retriever.ainvoke(state["query"], config=config)
        return {"docs": [d.page_content for d in found]}

    def retrieve_sync(state: _State, config: RunnableConfig) -> _State:
        found = retriever.invoke(state["query"], config=config)
        return {"docs": [d.page_content for d in found]}

    def model(state: _State) -> _State:
        return {"reached_model": True}

    g = StateGraph(_State)
    g.add_node("retrieve", retrieve_async if use_async else retrieve_sync)
    g.add_node("model", model)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "model")
    g.add_edge("model", END)
    return g.compile()


def _client(sent: list[dict]) -> GovernanceClient:
    """Core double: refuses an admission that returns anything labelled client:pepsi."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        labels = {label for r in body.get("resources") or [] for label in r.get("labels") or []}
        if body.get("event_type") == "ActivityCompleted" and "client:pepsi" in labels:
            refusal = {"verdict": "block", "reason": "Action blocked by policy"}
            return httpx.Response(200, json=refusal)
        return httpx.Response(200, json={"verdict": "allow"})

    client = GovernanceClient(api_url="https://core.test", api_key="obx_test_abc")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client._sync_client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def _admissions(sent: list[dict]) -> list[dict]:
    return [b for b in sent if b.get("action_class") == "retrieve"]


COKE = [
    Document("coke memo", metadata={"source": "dms://doc/1", "client": "coke"}),
    Document("untagged", metadata={"source": "dms://doc/2"}),
]
PEPSI = [Document("pepsi memo", metadata={"source": "dms://doc/3", "client": "pepsi"})]


@pytest.mark.parametrize("use_async", [True, False], ids=["async-node", "sync-node"])
async def test_allowed_documents_are_labelled_and_reach_the_model(use_async: bool) -> None:
    sent: list[dict] = []
    handler = OpenBoxLangGraphHandler(
        _graph(_Docs(docs=COKE), use_async=use_async),
        OpenBoxLangGraphHandlerOptions(client=_client(sent), label_map=LABEL_MAP),
    )
    with exposure.exposure_scope("matter:0417"):
        state = await handler.ainvoke({"query": "q"}, config={"configurable": {"thread_id": "t1"}})

    assert state["docs"] == ["coke memo", "untagged"]
    assert state["reached_model"] is True
    (admission,) = _admissions(sent)
    assert admission["event_type"] == "ActivityCompleted"
    assert admission["scope"] == "matter:0417"
    assert admission["activity_type"] == "_Docs"
    assert admission["resources"] == [
        {"uri": "dms://doc/1", "labels": ["client:coke"], "label_source": "source_system"},
        {"uri": "dms://doc/2", "label_source": "source_system"},
    ]
    assert handler.label_coverage.snapshot()["_Docs"]["unknown_share"] == 0.5


@pytest.mark.parametrize("use_async", [True, False], ids=["async-node", "sync-node"])
async def test_refused_documents_never_reach_the_next_node(use_async: bool) -> None:
    sent: list[dict] = []
    handler = OpenBoxLangGraphHandler(
        _graph(_Docs(docs=PEPSI), use_async=use_async),
        OpenBoxLangGraphHandlerOptions(client=_client(sent), label_map=LABEL_MAP),
    )
    with pytest.raises(GovernanceBlockedError) as err:
        await handler.ainvoke({"query": "q"}, config={"configurable": {"thread_id": "t2"}})
    # The same refusal as any block: no label, no document, no barrier named.
    assert str(err.value) == "Action blocked by policy"
    assert "pepsi" not in str(err.value)
    assert len(_admissions(sent)) == 1
    # Nothing from the model node was ever sent.
    assert not any(b.get("activity_type") == "model" for b in sent)


async def test_without_a_label_map_retrievers_are_not_sent() -> None:
    sent: list[dict] = []
    handler = OpenBoxLangGraphHandler(
        _graph(_Docs(docs=PEPSI), use_async=True),
        OpenBoxLangGraphHandlerOptions(client=_client(sent)),
    )
    state = await handler.ainvoke({"query": "q"}, config={"configurable": {"thread_id": "t3"}})
    assert state["reached_model"] is True
    assert _admissions(sent) == []
