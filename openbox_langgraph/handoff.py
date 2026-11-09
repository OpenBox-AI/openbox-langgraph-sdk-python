"""Task-only handoffs between two governed agents (PROD-1109, explainer §8).

A handoff from one governed agent to another is where an exposure barrier
applies between agents. Subagents running inside one handler are the same
OpenBox agent and scope, so Core already treats them as one session; this
module is for an agent handing work to a *different* agent.

Two halves:

* **Parent:** :func:`handoff_tool` builds a tool whose only argument is the
  task. The parent's message history and state never travel; the child gets
  the task and nothing else. Core still counts the parent's whole taint as
  handed on (the spec's conservative rule), but the content that actually
  reaches the child is only what the task says.
* **Child:** :meth:`OpenBoxLangGraphHandler.areceive_handoff` records the
  handoff with Core first (Core checks it and copies the parent's exposure
  into the child), runs the graph on the task alone in the parent's scope,
  then asks Core whether the result may return to the parent
  (``action_class: handoff_return``) before it leaves. If not, the parent gets
  the same refusal any block carries and keeps working (PROD-1105).

The envelope the two exchange is plain JSON, so ``send`` can be an HTTP call,
a queue, or a direct call to the child's handler.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from openbox_langgraph.retriever_labels import UNIFORM_REFUSAL

try:
    from openbox_core import exposure as _exposure  # type: ignore[attr-defined,unused-ignore]
except ImportError:  # openbox-sdk-python before exposure fields
    _exposure = None  # type: ignore[assignment,unused-ignore]

try:  # PROD-1088: declare a tool call's action class on its start event
    from openbox_langchain.exposure import (  # type: ignore[import-not-found,unused-ignore]
        ToolExposure,
        exposure_tool,
    )
except ImportError:  # openbox-langchain before tool declarations
    ToolExposure = None  # type: ignore[assignment,misc,unused-ignore]
    exposure_tool = None  # type: ignore[assignment,unused-ignore]

logger = logging.getLogger(__name__)

#: Envelope key for the handoff's governance fields.
HANDOFF_KEY = "openbox_handoff"


def handoff_scope(scope: str | None, multi_agent_session_id: str) -> str:
    """The scope both sides of a handoff use: the declared one, else the
    multi-agent session (Core's own fallback)."""
    s = (scope or "").strip()
    return s or f"mas:{multi_agent_session_id}"


@dataclass(frozen=True)
class HandoffEnvelope:
    """What a parent sends a child: the task, and who is handing it over."""

    task: str
    multi_agent_session_id: str
    #: The parent's OpenBox DID, for the child's ``Handoff`` event.
    from_agent_did: str
    #: The parent's OpenBox agent id, for the ``handoff_return`` participant.
    from_agent_id: str
    scope: str | None = None

    def to_dict(self) -> dict[str, Any]:
        gov: dict[str, Any] = {
            "multi_agent_session_id": self.multi_agent_session_id,
            "from_agent_did": self.from_agent_did,
            "from_agent_id": self.from_agent_id,
        }
        if self.scope:
            gov["scope"] = self.scope
        return {"task": self.task, HANDOFF_KEY: gov}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HandoffEnvelope:
        gov = data.get(HANDOFF_KEY)
        task = data.get("task")
        if not isinstance(gov, Mapping) or not isinstance(task, str):
            raise ValueError("not an OpenBox handoff envelope")
        missing = [
            k for k in ("multi_agent_session_id", "from_agent_did", "from_agent_id")
            if not str(gov.get(k) or "").strip()
        ]
        if missing:
            raise ValueError(f"handoff envelope is missing {', '.join(missing)}")
        return cls(
            task=task,
            multi_agent_session_id=str(gov["multi_agent_session_id"]).strip(),
            from_agent_did=str(gov["from_agent_did"]).strip(),
            from_agent_id=str(gov["from_agent_id"]).strip(),
            scope=(str(gov.get("scope") or "").strip() or None),
        )


@dataclass(frozen=True)
class HandoffResult:
    """What a child sends back: its result, or a refusal that says nothing more."""

    result: str | None = None
    refused: bool = False

    def to_dict(self) -> dict[str, Any]:
        if self.refused:
            return {"refused": True, "reason": UNIFORM_REFUSAL}
        return {"result": self.result or ""}

    @classmethod
    def from_dict(cls, data: Any) -> HandoffResult:
        if not isinstance(data, Mapping) or data.get("refused"):
            return cls(refused=True)
        return cls(result=str(data.get("result") or ""))


class _TaskOnly(BaseModel):
    task: str = Field(description="What the other agent should do, in full. Nothing else is sent.")


Send = Callable[[dict[str, Any]], "Awaitable[Any] | Any"]


def _current_scope() -> str | None:
    if _exposure is None:
        return None
    scope = _exposure.current_exposure_fields().get("scope")
    return str(scope) if scope else None


def handoff_tool(
    *,
    name: str,
    description: str,
    target_agent_id: str,
    send: Send,
    from_agent_did: str,
    from_agent_id: str,
    multi_agent_session_id: str | Callable[[], str] | None = None,
) -> BaseTool:
    """A tool that hands a task, and only the task, to another governed agent.

    ``send(envelope)`` delivers the envelope to the child (sync or async) and
    returns the child's reply (:meth:`HandoffResult.to_dict`). The tool
    returns the child's result, or the uniform refusal when Core refused the
    handoff or its return. ``multi_agent_session_id`` ties the two agents'
    events together; by default each handoff gets a fresh one.
    """

    def envelope(task: str) -> dict[str, Any]:
        if callable(multi_agent_session_id):
            mas = multi_agent_session_id()
        else:
            mas = multi_agent_session_id or uuid.uuid4().hex
        return HandoffEnvelope(
            task=task,
            multi_agent_session_id=mas,
            from_agent_did=from_agent_did,
            from_agent_id=from_agent_id,
            scope=_current_scope(),
        ).to_dict()

    def answer(reply: Any) -> str:
        result = HandoffResult.from_dict(reply)
        return UNIFORM_REFUSAL if result.refused else (result.result or "")

    async def arun(task: str) -> str:
        reply = send(envelope(task))
        if inspect.isawaitable(reply):
            reply = await reply
        return answer(reply)

    def run(task: str) -> str:
        reply = send(envelope(task))
        if inspect.isawaitable(reply):
            close = getattr(reply, "close", None)
            if close:
                close()
            raise TypeError(f"{name}: send is async; call the agent with ainvoke")
        return answer(reply)

    tool = StructuredTool.from_function(
        func=run,
        coroutine=arun,
        name=name,
        description=description,
        args_schema=_TaskOnly,
    )

    if exposure_tool is not None and ToolExposure is not None and _exposure is not None:
        participant = _exposure.Participant

        def participants(_args: dict[str, Any]) -> list[Any]:
            scope = _current_scope()
            child = f"{target_agent_id}/{scope}" if scope else target_agent_id
            return [participant("agent_session", child)]

        exposure_tool(
            tool,
            ToolExposure(
                action_class="handoff",
                participants=participants,
                payload=lambda args: str(args.get("task") or ""),
            ),
        )
    else:
        logger.debug(
            "OpenBox: %s is task-only but its start event is not declared as a handoff; "
            "needs openbox-langchain with tool declarations (PROD-1088)",
            name,
        )
    return tool
