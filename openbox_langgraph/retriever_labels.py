"""Labels from retrieved documents, checked before they reach the graph (PROD-1109).

Exposure barriers need to know which client or matter every object an agent
reads belongs to. Retrieved documents usually say so already, in their
metadata (``client``, ``matter_id`` and so on). This module turns that
metadata into labels and sends the documents to Core as the objects a
retrieve returned, so Core's admission gate checks them before the agent
sees them.

``RetrieverAdmissionCallback`` is a LangChain callback with
``raise_error=True``. LangChain runs ``on_retriever_end`` before a retriever
returns, so when Core refuses, the error propagates out of the retriever and
the documents never reach the tool, the model or the graph state. The
refusal is the same one any block raises: it does not say which document or
label caused it.

A document with no mapped label is still sent, unlabelled. Core treats it as
Unknown: shadow mode records it, enforce mode refuses it. ``LabelCoverage``
counts how many documents each retriever labels, which is the label coverage
the design asks to measure on real traffic.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from openbox_langgraph.errors import GovernanceBlockedError
from openbox_langgraph.types import LangChainGovernanceEvent, rfc3339_now
from openbox_langgraph.verdict_handler import enforce_verdict

if TYPE_CHECKING:
    from openbox_langgraph.client import GovernanceClient

try:
    from openbox_core import exposure as _exposure  # type: ignore[attr-defined,unused-ignore]
except ImportError:  # openbox-sdk-python before exposure fields
    _exposure = None  # type: ignore[assignment,unused-ignore]

logger = logging.getLogger(__name__)

#: Metadata keys tried, in order, for a document's URI.
DEFAULT_URI_KEYS: tuple[str, ...] = ("source", "uri", "url", "id")

#: The refusal every block carries when Core gives no reason (PROD-1104).
UNIFORM_REFUSAL = "Action blocked by policy"


def exposure_available() -> bool:
    """Whether the installed openbox-sdk-python can carry exposure fields."""
    return _exposure is not None


def labels_from_metadata(metadata: Mapping[str, Any], label_map: Mapping[str, str]) -> list[str]:
    """Labels for one document: ``{"client": "client"}`` maps ``metadata["client"] = "coke"``
    to ``client:coke``.

    A list value gives one label per item. Values that are empty, booleans, or
    contain whitespace or ``*`` are skipped, since they are not labels Core can
    match; the document then counts as unlabelled for that key.
    """
    out: list[str] = []
    for key, namespace in label_map.items():
        raw = metadata.get(key)
        values = raw if isinstance(raw, (list, tuple, set, frozenset)) else [raw]
        for value in values:
            if value is None or isinstance(value, bool):
                continue
            text = str(value).strip()
            if not text or "*" in text or any(c.isspace() for c in text):
                continue
            label = f"{namespace}:{text}"
            if label not in out:
                out.append(label)
    return out


def document_uri(document: Any, index: int, retriever: str, uri_keys: Sequence[str]) -> str:
    """A stable URI for a document: the first metadata key present, then the
    document's id, then ``<retriever>#<index>``."""
    metadata = getattr(document, "metadata", None) or {}
    for key in uri_keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    doc_id = getattr(document, "id", None)
    if doc_id is not None and str(doc_id).strip():
        return str(doc_id).strip()
    return f"{retriever}#{index}"


def resources_from_documents(
    documents: Iterable[Any],
    *,
    retriever: str,
    label_map: Mapping[str, str],
    uri_keys: Sequence[str] = DEFAULT_URI_KEYS,
) -> list[tuple[str, list[str]]]:
    """``(uri, labels)`` per distinct document, in order. Chunks of one source
    document share a URI and their labels are merged."""
    merged: dict[str, list[str]] = {}
    for index, document in enumerate(documents):
        uri = document_uri(document, index, retriever, uri_keys)
        labels = labels_from_metadata(getattr(document, "metadata", None) or {}, label_map)
        have = merged.setdefault(uri, [])
        for label in labels:
            if label not in have:
                have.append(label)
    return list(merged.items())


class LabelCoverage:
    """Labelled and unlabelled document counts per retriever, thread safe.

    Logs a summary every ``log_every`` documents (0 turns logging off);
    ``snapshot()`` returns the counts for a caller that reports them itself.
    """

    def __init__(self, log_every: int = 100) -> None:
        self._log_every = log_every
        self._lock = threading.Lock()
        self._counts: dict[str, list[int]] = {}
        self._since_log = 0

    def record(self, retriever: str, labelled: int, unlabelled: int) -> None:
        with self._lock:
            counts = self._counts.setdefault(retriever, [0, 0])
            counts[0] += labelled
            counts[1] += unlabelled
            self._since_log += labelled + unlabelled
            due = self._log_every > 0 and self._since_log >= self._log_every
            if due:
                self._since_log = 0
        if due:
            for name, row in self.snapshot().items():
                logger.info(
                    "OpenBox label coverage: %s labelled %d, unlabelled %d (unknown share %.0f%%)",
                    name,
                    row["labelled"],
                    row["unlabelled"],
                    row["unknown_share"] * 100,
                )

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            out: dict[str, dict[str, float]] = {}
            for name, (labelled, unlabelled) in self._counts.items():
                total = labelled + unlabelled
                out[name] = {
                    "labelled": labelled,
                    "unlabelled": unlabelled,
                    "unknown_share": (unlabelled / total) if total else 0.0,
                }
            return out


class RetrieverAdmissionCallback(BaseCallbackHandler):
    """Sends what a retriever returned to Core's admission gate and raises on a refusal.

    Sync on purpose: LangChain runs a sync handler in an executor under an
    async run and still re-raises, so one handler covers sync and async
    retrievers with exactly one evaluate per retrieval.
    """

    raise_error = True

    def __init__(
        self,
        *,
        client: GovernanceClient,
        label_map: Mapping[str, str],
        coverage: LabelCoverage,
        workflow_id: str,
        run_id: str,
        workflow_type: str,
        task_queue: str,
        session_id: str | None,
        uri_keys: Sequence[str] = DEFAULT_URI_KEYS,
    ) -> None:
        super().__init__()
        self._client = client
        self._label_map = dict(label_map)
        self._coverage = coverage
        self._uri_keys = tuple(uri_keys)
        self._workflow_id = workflow_id
        self._run_id = run_id
        self._workflow_type = workflow_type
        self._task_queue = task_queue
        self._session_id = session_id
        self._names: dict[UUID, str] = {}
        self._lock = threading.Lock()

    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name") or "retriever"
        with self._lock:
            self._names[run_id] = str(name)

    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            self._names.pop(run_id, None)

    def on_retriever_end(self, documents: Sequence[Any], *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            name = self._names.pop(run_id, "retriever")
        if _exposure is None:
            return
        found = resources_from_documents(
            documents, retriever=name, label_map=self._label_map, uri_keys=self._uri_keys
        )
        if not found:
            return
        labelled = sum(1 for _, labels in found if labels)
        self._coverage.record(name, labelled, len(found) - labelled)

        resources = [_exposure.Resource(uri, labels, "source_system") for uri, labels in found]
        event = LangChainGovernanceEvent(
            source="workflow-telemetry",
            event_type="ToolCompleted",
            workflow_id=self._workflow_id,
            run_id=self._run_id,
            workflow_type=self._workflow_type,
            task_queue=self._task_queue,
            timestamp=rfc3339_now(),
            session_id=self._session_id,
            activity_id=f"{run_id}-admission",
            activity_type=name,
            status="completed",
        )
        with _exposure.governed_action("retrieve", resources=resources):
            response = self._client.evaluate_event_sync(event)
        if response is None:
            return
        result = enforce_verdict(response, "tool_end")
        if result.requires_hitl:
            # A retriever cannot be paused for an approval from inside its own
            # callback; refuse rather than let unapproved documents through.
            raise GovernanceBlockedError(
                response.reason or UNIFORM_REFUSAL, response.policy_id, response.risk_score
            )
