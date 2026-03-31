"""OpenBox LangGraph SDK — OpenBoxLangGraphHandler.

Wraps any compiled LangGraph graph and processes the v2 event stream to apply
OpenBox governance at every node, tool, and LLM invocation.

For framework-specific integrations (e.g. DeepAgents) use the dedicated
`openbox-deepagent` package which extends this handler.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.config import GovernanceConfig, get_global_config, merge_config
from openbox_langgraph.errors import (
    ApprovalExpiredError,
    ApprovalRejectedError,
    ApprovalTimeoutError,
    GovernanceBlockedError,
    GovernanceHaltError,
    GuardrailsValidationError,
)
from openbox_langgraph.hitl import HITLPollParams, poll_until_decision
from openbox_langgraph.types import (
    GovernanceVerdictResponse,
    LangChainGovernanceEvent,
    LangGraphStreamEvent,
    rfc3339_now,
    safe_serialize,
)
from openbox_langgraph.verdict_handler import (
    enforce_verdict,
    lang_graph_event_to_context,
)

_logger = logging.getLogger(__name__)

_otel_tracer = otel_trace.get_tracer("openbox-langgraph")


def _extract_governance_blocked(exc: Exception) -> GovernanceBlockedError | None:
    """Walk exception chain to find a wrapped GovernanceBlockedError.

    LLM SDKs (OpenAI, Anthropic) wrap httpx errors. When an OTel hook raises
    GovernanceBlockedError inside httpx, the LLM SDK wraps it as APIConnectionError.
    This function unwraps the chain via __cause__ / __context__ to recover it.
    """
    cause: BaseException | None = exc
    seen: set[int] = set()
    while cause is not None:
        if id(cause) in seen:
            break
        seen.add(id(cause))
        if isinstance(cause, GovernanceBlockedError):
            return cause
        cause = getattr(cause, '__cause__', None) or getattr(cause, '__context__', None)
    return None


# ═══════════════════════════════════════════════════════════════════
# Guardrails callback handler — pre-LLM interception for PII redaction
# ═══════════════════════════════════════════════════════════════════

class _GuardrailsCallbackHandler(AsyncCallbackHandler):
    """LangChain callback handler that intercepts on_chat_model_start BEFORE the
    LLM call fires, sends a governance LLMStarted event, and mutates the messages
    in-place with redacted_input from Core.

    This mirrors the TypeScript SDK's handleChatModelStart with awaitHandlers=True.
    Injected into config['callbacks'] so LangGraph propagates it to every LLM node.
    """

    raise_error = True  # Surface GuardrailsValidationError / GovernanceHaltError

    def __init__(
        self,
        client: GovernanceClient,
        config: GovernanceConfig,
        workflow_id: str,
        run_id: str,
        thread_id: str,
        pre_screen_response: GovernanceVerdictResponse | None = None,
        pre_screen_activity_id: str | None = None,
        llm_activity_map: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._config = config
        self._workflow_id = workflow_id
        self._run_id = run_id
        self._thread_id = thread_id
        self._pre_screen_response = pre_screen_response
        self._pre_screen_activity_id = pre_screen_activity_id
        # Shared dict: LangChain callback UUID → activity_id to use for span hook.
        # Written here, read by _process_event when LLMCompleted fires.
        self._llm_activity_map: dict[str, str] = (
            llm_activity_map if llm_activity_map is not None else {}
        )

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not self._config.send_llm_start_event:
            return

        # Extract human/user turn text only — mirrors _extract_prompt_from_messages.
        # Subagent-internal LLM calls have only system/tool messages → empty prompt
        # → skip guard below prevents sending {"prompt": ""} to Core's guardrail.
        prompt_parts: list[str] = []
        for group in messages:
            for msg in group:
                role = getattr(msg, "type", None) or getattr(msg, "role", None) or ""
                if role not in ("human", "user", "generic"):
                    continue
                content = msg.content
                if isinstance(content, str):
                    prompt_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            prompt_parts.append(part.get("text", ""))
        prompt_text = "\n".join(prompt_parts)

        # Skip governance for LLM calls with no human-turn text (e.g. subagent
        # internal LLMs that only have system/tool messages). Sending an empty
        # prompt causes Core's guardrail to return a JSON parse error (block).
        if not prompt_text.strip():
            return

        model_name = (
            serialized.get("name")
            or (serialized.get("id") or [None])[-1]
            or "LLM"
        )
        event_run_id = str(run_id)

        gov = LangChainGovernanceEvent(
            source="workflow-telemetry",
            event_type="LLMStarted",
            workflow_id=self._workflow_id,
            run_id=self._run_id,
            workflow_type=self._config.agent_name or "LangGraphRun",
            task_queue=self._config.task_queue,
            timestamp=rfc3339_now(),
            session_id=self._config.session_id,
            activity_id=event_run_id,
            activity_type="llm_call",
            activity_input=[{"prompt": prompt_text}],
            llm_model=model_name,
            prompt=prompt_text,
        )

        if self._pre_screen_response is not None:
            # Reuse the pre-screen verdict for PII redaction — the pre-screen already
            # created an ActivityStarted row (activity_id=run_id+"-pre").  Record that
            # mapping so _process_event knows to attach the LLM span hook to THAT row
            # rather than creating a new one with the callback UUID.
            response = self._pre_screen_response
            self._pre_screen_response = None
            if self._pre_screen_activity_id:
                self._llm_activity_map[event_run_id] = self._pre_screen_activity_id
        else:
            # No pre-screen (second+ LLM call, or pre-screen disabled) — create a
            # new row with the callback UUID so the span hook has a row to attach to.
            response = await self._client.evaluate_event(gov)
            self._llm_activity_map[event_run_id] = event_run_id
        if response is None:
            return

        # NOTE: enforce_verdict / HITL are intentionally NOT called here.
        # LangGraph's graph runner catches callback exceptions even with raise_error=True
        # and logs them as warnings instead of propagating them to the caller.
        # Block/halt/guardrail enforcement is done in _pre_screen_input() which runs
        # directly in ainvoke/astream_governed before the stream starts.
        #
        # This callback handler's only job is PII redaction (in-place message mutation).

        # Apply PII redaction: mutate messages in-place before the LLM call fires
        gr = response.guardrails_result
        if gr and gr.input_type == "activity_input" and gr.redacted_input is not None:
            redacted = gr.redacted_input
            # Core returns [{"prompt": "..."}] — extract the prompt string
            if isinstance(redacted, list) and redacted:
                first = redacted[0]
                if isinstance(first, dict):
                    redacted_text = first.get("prompt")
                elif isinstance(first, str):
                    redacted_text = first
                else:
                    redacted_text = None
            elif isinstance(redacted, str):
                redacted_text = redacted
            else:
                redacted_text = None

            if redacted_text:
                # Replace the last human message in each message group
                for group in messages:
                    for j in range(len(group) - 1, -1, -1):
                        msg = group[j]
                        if msg.type in ("human", "generic"):
                            msg.content = redacted_text  # type: ignore[assignment]
                            break


# ═══════════════════════════════════════════════════════════════════
# Run buffer (tracks in-flight runs for duration/context)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class _RunBuffer:
    run_id: str
    run_type: str
    name: str
    thread_id: str
    start_time_ms: float = field(default_factory=lambda: time.monotonic() * 1000)
    start_time_ns: int = field(default_factory=time.time_ns)
    langgraph_node: str | None = None
    langgraph_step: int | None = None
    subagent_name: str | None = None
    llm_started: bool = False  # True only when LLMStarted was actually sent to Core
    otel_span: Any = None       # OTel span for context propagation across asyncio.Task
    otel_token: Any = None      # OTel context detach token


class _RunBufferManager:
    def __init__(self) -> None:
        self._buffers: dict[str, _RunBuffer] = {}

    def register(
        self,
        run_id: str,
        run_type: str,
        name: str,
        thread_id: str,
        langgraph_node: str | None = None,
        langgraph_step: int | None = None,
        subagent_name: str | None = None,
    ) -> None:
        self._buffers[run_id] = _RunBuffer(
            run_id=run_id,
            run_type=run_type,
            name=name,
            thread_id=thread_id,
            langgraph_node=langgraph_node,
            langgraph_step=langgraph_step,
            subagent_name=subagent_name,
        )

    def get(self, run_id: str) -> _RunBuffer | None:
        return self._buffers.get(run_id)

    def remove(self, run_id: str) -> None:
        self._buffers.pop(run_id, None)

    def duration_ms(self, run_id: str) -> float | None:
        buf = self._buffers.get(run_id)
        if buf is None:
            return None
        return time.monotonic() * 1000 - buf.start_time_ms


# ═══════════════════════════════════════════════════════════════════
# Root run tracker (identifies the outermost graph invocation)
# ═══════════════════════════════════════════════════════════════════

class _RootRunTracker:
    def __init__(self) -> None:
        self._root_run_id: str | None = None

    def is_root(self, run_id: str) -> bool:
        """Return True and register run_id as root if no root exists yet."""
        if self._root_run_id is None:
            self._root_run_id = run_id
            return True
        return self._root_run_id == run_id

    @property
    def root_run_id(self) -> str | None:
        return self._root_run_id

    def reset(self) -> None:
        self._root_run_id = None


# ═══════════════════════════════════════════════════════════════════
# Options
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OpenBoxLangGraphHandlerOptions:
    """Configuration options for `OpenBoxLangGraphHandler`."""

    client: GovernanceClient | None = None
    on_api_error: str = "fail_open"
    api_timeout: int = 30_000
    send_chain_start_event: bool = True
    send_chain_end_event: bool = True
    send_tool_start_event: bool = True
    send_tool_end_event: bool = True
    send_llm_start_event: bool = True
    send_llm_end_event: bool = True
    skip_chain_types: set[str] = field(default_factory=set)
    skip_tool_types: set[str] = field(default_factory=set)
    hitl: Any = None  # HITLConfig | dict | None
    session_id: str | None = None
    agent_name: str | None = None
    task_queue: str = "langgraph"
    use_native_interrupt: bool = False
    root_node_names: set[str] = field(default_factory=set)
    resolve_subagent_name: Callable[[LangGraphStreamEvent], str | None] | None = None
    """Optional hook for framework-specific subagent name detection.

    Called on every `on_chain_start` / `on_tool_start` event.
    Return the subagent name if this event is a subagent invocation, else None.
    DeepAgents integration sets this to detect `task` tool sub-graphs.
    """
    sqlalchemy_engine: Any = None
    """Optional SQLAlchemy Engine instance to instrument for DB governance.
    Required when the engine is created before the handler (e.g. SQLDatabase.from_uri()).
    """
    tool_type_map: dict[str, str] | None = None
    """Optional mapping of tool_name → tool_type for execution tree classification.

    Supported values: "http", "database", "builtin", "a2a", "custom".
    If a tool is not listed and subagent_name is set, defaults to "a2a".
    Otherwise defaults to "custom".

    Example::

        tool_type_map={"search_web": "http", "query_db": "database"}
    """


# ═══════════════════════════════════════════════════════════════════
# OpenBoxLangGraphHandler
# ═══════════════════════════════════════════════════════════════════

class OpenBoxLangGraphHandler:
    """Wraps a compiled LangGraph graph and applies OpenBox governance to its event stream.

    Usage:
        governed = await create_openbox_graph_handler(
            graph=my_compiled_graph,
            api_url=os.environ["OPENBOX_URL"],
            api_key=os.environ["OPENBOX_API_KEY"],
            agent_name="MyAgent",
        )
        result = await governed.ainvoke(
            {"messages": [{"role": "user", "content": "Hello"}]},
            config={"configurable": {"thread_id": "session-abc"}},
        )
    """

    def __init__(
        self,
        graph: Any,
        options: OpenBoxLangGraphHandlerOptions | None = None,
    ) -> None:
        opts = options or OpenBoxLangGraphHandlerOptions()
        self._graph = graph
        self._resolve_subagent_name = opts.resolve_subagent_name

        # Build GovernanceConfig from options
        self._config = merge_config({
            "on_api_error": opts.on_api_error,
            "api_timeout": opts.api_timeout,
            "send_chain_start_event": opts.send_chain_start_event,
            "send_chain_end_event": opts.send_chain_end_event,
            "send_tool_start_event": opts.send_tool_start_event,
            "send_tool_end_event": opts.send_tool_end_event,
            "send_llm_start_event": opts.send_llm_start_event,
            "send_llm_end_event": opts.send_llm_end_event,
            "skip_chain_types": opts.skip_chain_types,
            "skip_tool_types": opts.skip_tool_types,
            "hitl": opts.hitl,
            "session_id": opts.session_id,
            "agent_name": opts.agent_name,
            "task_queue": opts.task_queue,
            "use_native_interrupt": opts.use_native_interrupt,
            "root_node_names": opts.root_node_names,
            "tool_type_map": opts.tool_type_map or {},
        })

        if opts.client:
            self._client = opts.client
        else:
            gc = get_global_config()
            self._client = GovernanceClient(
                api_url=gc.api_url,
                api_key=gc.api_key,
                timeout=gc.governance_timeout,  # seconds
                on_api_error=self._config.on_api_error,
            )

        # Setup OTel HTTP governance hooks (required)
        gc = get_global_config()
        if gc and gc.api_url and gc.api_key:
            from openbox_langgraph.otel_setup import setup_opentelemetry_for_governance
            from openbox_langgraph.span_processor import WorkflowSpanProcessor
            self._span_processor = WorkflowSpanProcessor()
            setup_opentelemetry_for_governance(
                span_processor=self._span_processor,
                api_url=gc.api_url,
                api_key=gc.api_key,
                ignored_urls=[gc.api_url],
                api_timeout=gc.governance_timeout,
                on_api_error=self._config.on_api_error,
                sqlalchemy_engine=opts.sqlalchemy_engine,
            )
            _logger.debug("[OpenBox] OTel HTTP governance hooks enabled")
        else:
            self._span_processor = None

    # ─────────────────────────────────────────────────────────────
    # Pre-screen: enforce guardrails before stream starts
    # ─────────────────────────────────────────────────────────────

    async def _pre_screen_input(
        self,
        input: dict[str, Any],
        workflow_id: str,
        run_id: str,
        graph_input: dict[str, Any] | None = None,
    ) -> tuple[bool, GovernanceVerdictResponse | None]:
        """Send WorkflowStarted + LLMStarted governance events before the stream starts.

        Returns (workflow_started_sent, pre_screen_response):
        - workflow_started_sent: True if WorkflowStarted was sent (suppress duplicate
          from on_chain_start in _process_event).
        - pre_screen_response: the LLMStarted verdict response, passed to the callback
          handler so on_chat_model_start can reuse it for PII redaction without sending
          a second ActivityStarted event.

        Unlike the callback handler (which LangGraph's runner silently swallows),
        exceptions raised here propagate directly to the ainvoke/astream_governed
        caller — so GuardrailsValidationError, GovernanceHaltError, GovernanceBlockedError
        all reach the user's except block and halt the session correctly.
        """
        # ── 0. SignalReceived — fire before WorkflowStarted so the dashboard shows
        # the user prompt as the trigger that initiated the session.
        # Extract the last human message from the input as the signal payload.
        _sig_messages = input.get("messages") or []
        _user_prompt: str | None = None
        for _msg in reversed(_sig_messages):
            if isinstance(_msg, dict):
                if _msg.get("role") in ("user", "human"):
                    _user_prompt = _msg.get("content") or None
                    break
            elif hasattr(_msg, "type") and _msg.type in ("human", "generic"):
                _c = _msg.content
                _user_prompt = _c if isinstance(_c, str) else None
                break
        if _user_prompt:
            sig_event = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type="SignalReceived",
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                activity_id=f"{run_id}-sig",
                activity_type="user_prompt",
                signal_name="user_prompt",
                signal_args=[_user_prompt],
            )
            await self._client.evaluate_event(sig_event)

        # ── 1. WorkflowStarted — must precede any ActivityStarted so the dashboard
        # creates a session to attach the guardrail event to (mirrors both SDKs).
        # Gated on send_chain_start_event only, NOT on send_llm_start_event.
        if self._config.send_chain_start_event:
            wf_start = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type="WorkflowStarted",
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                activity_id=f"{run_id}-wf",
                activity_type=self._config.agent_name or "LangGraphRun",
                activity_input=[safe_serialize(input)],
            )
            await self._client.evaluate_event(wf_start)
            workflow_started_sent = True
        else:
            workflow_started_sent = False

        # ── 2. LLMStarted pre-screen — enforce guardrails on the user prompt
        if not self._config.send_llm_start_event:
            return workflow_started_sent, None

        messages = input.get("messages") or []
        prompt_parts: list[str] = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role in ("user", "human") and isinstance(content, str):
                    prompt_parts.append(content)
            elif hasattr(msg, "type") and msg.type in ("human", "generic"):
                c = msg.content
                if isinstance(c, str):
                    prompt_parts.append(c)
        if not prompt_parts:
            return workflow_started_sent, None
        prompt_text = "\n".join(prompt_parts)

        gov = LangChainGovernanceEvent(
            source="workflow-telemetry",
            event_type="LLMStarted",
            workflow_id=workflow_id,
            run_id=run_id,
            workflow_type=self._config.agent_name or "LangGraphRun",
            task_queue=self._config.task_queue,
            timestamp=rfc3339_now(),
            session_id=self._config.session_id,
            activity_id=f"{run_id}-pre",
            activity_type="llm_call",
            activity_input=[{"prompt": prompt_text}],
            prompt=prompt_text,
        )

        response = await self._client.evaluate_event(gov)
        if response is None:
            return workflow_started_sent, None

        # Enforce — exceptions propagate directly to the caller here.
        # If blocked/halted, close the WorkflowStarted session first so the
        # dashboard doesn't show an orphaned open session.
        enforcement_error: Exception | None = None
        try:
            result = enforce_verdict(response, "llm_start")
        except Exception as exc:
            enforcement_error = exc
            result = None  # type: ignore[assignment]

        if (
            enforcement_error is not None
            and workflow_started_sent
            and self._config.send_chain_end_event
        ):
            wf_end = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type="WorkflowCompleted",
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                activity_id=f"{run_id}-wf",
                activity_type=self._config.agent_name or "LangGraphRun",
                status="failed",
                error=str(enforcement_error),
            )
            await self._client.evaluate_event(wf_end)
            raise enforcement_error

        if result and result.requires_hitl:
            try:
                await poll_until_decision(
                    self._client,
                    HITLPollParams(
                        workflow_id=workflow_id,
                        run_id=run_id,
                        activity_id=f"{run_id}-pre",
                        activity_type="llm_call",
                    ),
                    self._config.hitl,
                )
            except (ApprovalRejectedError, ApprovalExpiredError, ApprovalTimeoutError) as e:
                raise GovernanceHaltError(str(e)) from e

        return workflow_started_sent, response

    # ─────────────────────────────────────────────────────────────
    # Public invoke / ainvoke
    # ─────────────────────────────────────────────────────────────

    async def ainvoke(
        self,
        input: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Invoke the governed graph and return the final state.

        Streams events via `astream_events` (governance applied inline) and
        returns the final graph output from the root `on_chain_end` event.
        Does NOT call `ainvoke` on the underlying graph a second time.

        Args:
            input: The initial graph state (e.g. `{"messages": [...]}`)
            config: LangGraph RunnableConfig — must include
                `{"configurable": {"thread_id": "..."}}` for session tracking.
        """
        thread_id = _extract_thread_id(config)
        # Generate fresh workflow_id + run_id per turn, matching Temporal SDK:
        #   workflow_id = stable logical session ID (unique per-turn)
        #   run_id      = unique execution attempt ID (distinct from workflow_id)
        # Core seals a workflow after WorkflowCompleted — reusing the same
        # workflow_id causes HALT: "fully attested and sealed".
        _turn = uuid.uuid4().hex
        workflow_id = f"{thread_id}-{_turn[:8]}"
        run_id = f"{thread_id}-run-{_turn[8:16]}"
        root_tracker = _RootRunTracker()
        buffer = _RunBufferManager()
        final_output: dict[str, Any] = {}

        # Pre-screen: enforce guardrails BEFORE stream starts so exceptions
        # propagate to the caller (LangGraph runner swallows callback exceptions).
        # Returns (workflow_started_sent, pre_screen_response) — response reused
        # by callback handler for PII redaction to avoid duplicate ActivityStarted.
        workflow_started_sent, pre_screen_response = await self._pre_screen_input(
            input, workflow_id, run_id
        )

        # Shared map: LangChain callback UUID → activity_id to use for the LLM span hook.
        # Written by _GuardrailsCallbackHandler.on_chat_model_start, read by _process_event.
        llm_activity_map: dict[str, str] = {}

        # Inject guardrails callback for PII redaction only (in-place message mutation).
        guardrails_cb = _GuardrailsCallbackHandler(
            client=self._client,
            config=self._config,
            workflow_id=workflow_id,
            run_id=run_id,
            thread_id=thread_id,
            pre_screen_response=pre_screen_response,
            pre_screen_activity_id=f"{run_id}-pre" if pre_screen_response is not None else None,
            llm_activity_map=llm_activity_map,
        )
        cfg = dict(config or {})
        cfg["callbacks"] = [*list(cfg.get("callbacks") or []), guardrails_cb]

        try:
            async for event in self._graph.astream_events(
                input, config=cfg, version="v2", **kwargs
            ):
                stream_event = LangGraphStreamEvent.from_dict(event)
                await self._process_event(
                    stream_event, thread_id, workflow_id, run_id, root_tracker, buffer,
                    workflow_started_sent=workflow_started_sent, llm_activity_map=llm_activity_map,
                )
                # Capture the root graph's final output from on_chain_end
                if (
                    stream_event.event == "on_chain_end"
                    and root_tracker.root_run_id == stream_event.run_id
                ):
                    output = stream_event.data.get("output")
                    if isinstance(output, dict):
                        final_output = output
        except GovernanceBlockedError as hook_err:
            if hook_err.verdict != "require_approval":
                raise
            _logger.info("[OpenBox] Hook REQUIRE_APPROVAL during ainvoke, polling")
            await poll_until_decision(
                self._client,
                HITLPollParams(workflow_id=workflow_id, run_id=run_id,
                               activity_id=f"{run_id}-hook", activity_type="hook"),
                self._config.hitl,
            )
            _logger.info("[OpenBox] Approval granted, retrying ainvoke")
            final_output = await self._graph.ainvoke(input, config=cfg, **kwargs)
        except Exception as exc:
            hook_err = _extract_governance_blocked(exc)
            if hook_err is None or hook_err.verdict != "require_approval":
                raise
            _logger.info("[OpenBox] Hook REQUIRE_APPROVAL (wrapped) during ainvoke, polling")
            await poll_until_decision(
                self._client,
                HITLPollParams(workflow_id=workflow_id, run_id=run_id,
                               activity_id=f"{run_id}-hook", activity_type="hook"),
                self._config.hitl,
            )
            _logger.info("[OpenBox] Approval granted, retrying ainvoke")
            final_output = await self._graph.ainvoke(input, config=cfg, **kwargs)

        return final_output

    async def astream_governed(
        self,
        input: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream governed graph updates, yielding each update chunk.

        Governance is applied inline as events are streamed. The caller
        receives graph state update chunks identically to `astream_events`.

        Args:
            input: The initial graph state.
            config: LangGraph RunnableConfig with `thread_id`.
        """
        thread_id = _extract_thread_id(config)
        _turn = uuid.uuid4().hex
        workflow_id = f"{thread_id}-{_turn[:8]}"
        run_id = f"{thread_id}-run-{_turn[8:16]}"
        root_tracker = _RootRunTracker()
        buffer = _RunBufferManager()

        workflow_started_sent, pre_screen_response = await self._pre_screen_input(
            input, workflow_id, run_id
        )

        llm_activity_map: dict[str, str] = {}
        guardrails_cb = _GuardrailsCallbackHandler(
            client=self._client,
            config=self._config,
            workflow_id=workflow_id,
            run_id=run_id,
            thread_id=thread_id,
            pre_screen_response=pre_screen_response,
            pre_screen_activity_id=f"{run_id}-pre" if pre_screen_response is not None else None,
            llm_activity_map=llm_activity_map,
        )
        cfg = dict(config or {})
        cfg["callbacks"] = [*list(cfg.get("callbacks") or []), guardrails_cb]

        _debug = os.environ.get("OPENBOX_DEBUG") == "1"
        async for event in self._graph.astream_events(
            input, config=cfg, version="v2", **kwargs
        ):
            stream_event = LangGraphStreamEvent.from_dict(event)
            if _debug and "_stream" not in stream_event.event:
                sys.stderr.write(
                    f"[OBX_EVENT] {stream_event.event:<25} name={stream_event.name!r:<35} "
                    f"node={stream_event.metadata.get('langgraph_node')!r}\n"
                )
            await self._process_event(
                stream_event, thread_id, workflow_id, run_id, root_tracker, buffer,
                workflow_started_sent=workflow_started_sent, llm_activity_map=llm_activity_map,
            )
            yield event

    async def astream(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Graph-compatible astream — delegates to astream_governed.

        Provided so `langgraph dev` and other LangGraph tooling that calls
        ``graph.astream(...)`` can use this handler as a drop-in replacement
        for a ``CompiledStateGraph``.
        """
        async for chunk in self.astream_governed(input, config=config, **kwargs):
            yield chunk

    async def astream_events(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
        *,
        version: str = "v2",
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Graph-compatible astream_events — runs governance and re-yields raw events.

        Provided so tooling that calls ``graph.astream_events(...)`` works
        transparently with the governed handler.
        """
        thread_id = _extract_thread_id(config)
        _turn = uuid.uuid4().hex
        workflow_id = f"{thread_id}-{_turn[:8]}"
        run_id = f"{thread_id}-run-{_turn[8:16]}"
        root_tracker = _RootRunTracker()
        buffer = _RunBufferManager()

        workflow_started_sent, pre_screen_response = await self._pre_screen_input(
            input, workflow_id, run_id
        )

        llm_activity_map: dict[str, str] = {}
        guardrails_cb = _GuardrailsCallbackHandler(
            client=self._client,
            config=self._config,
            workflow_id=workflow_id,
            run_id=run_id,
            thread_id=thread_id,
            pre_screen_response=pre_screen_response,
            pre_screen_activity_id=f"{run_id}-pre" if pre_screen_response is not None else None,
            llm_activity_map=llm_activity_map,
        )
        cfg = dict(config or {})
        cfg["callbacks"] = [*list(cfg.get("callbacks") or []), guardrails_cb]

        async for event in self._graph.astream_events(
            input, config=cfg, version=version, **kwargs
        ):
            stream_event = LangGraphStreamEvent.from_dict(event)
            await self._process_event(
                stream_event, thread_id, workflow_id, run_id, root_tracker, buffer,
                workflow_started_sent=workflow_started_sent, llm_activity_map=llm_activity_map,
            )
            yield event

    # ─────────────────────────────────────────────────────────────
    # Event processing
    # ─────────────────────────────────────────────────────────────

    async def _process_event(
        self,
        event: LangGraphStreamEvent,
        thread_id: str,
        workflow_id: str,
        run_id: str,
        root_tracker: _RootRunTracker,
        buffer: _RunBufferManager,
        *,
        workflow_started_sent: bool = False,
        llm_activity_map: dict[str, str] | None = None,
    ) -> None:
        """Process a single LangGraph stream event through governance."""
        gov_event, is_root, is_start, event_type_label = self._map_event(
            event, thread_id, workflow_id, run_id, root_tracker, buffer
        )
        if gov_event is None:
            return

        # ── Skip events disabled in config
        if is_start:
            if event_type_label == "ChainStarted" and not self._config.send_chain_start_event:
                return
            # _pre_screen_input already sent WorkflowStarted — skip duplicate from on_chain_start
            if event_type_label == "ChainStarted" and is_root and workflow_started_sent:
                return
            if event_type_label == "ToolStarted" and not self._config.send_tool_start_event:
                return
            if event_type_label == "LLMStarted" and not self._config.send_llm_start_event:
                return
            # _GuardrailsCallbackHandler owns LLMStarted — it fires pre-LLM with redaction.
            # Skip re-sending here to avoid duplicate governance events.
            if event_type_label == "LLMStarted":
                return
        else:
            if event_type_label == "ChainCompleted" and not self._config.send_chain_end_event:
                return
            if event_type_label == "ToolCompleted" and not self._config.send_tool_end_event:
                return
            # Skip LLMCompleted governance event — no ActivityCompleted sent for LLM
            # calls (mirrors Temporal SDK). Fire the LLM span hook instead, routed to
            # the correct existing row so no orphan rows are created.
            if event_type_label == "LLMCompleted":
                if self._config.send_llm_start_event and gov_event.activity_id:
                    # Resolve activity_id for the LLM row (pre-screen or callback-UUID)
                    llm_activity_id = (
                        (llm_activity_map or {}).get(gov_event.activity_id)
                        or gov_event.activity_id
                    )
                    llm_activity_type = gov_event.activity_type or "llm_call"

                    # Close the LLM row in Core with ActivityCompleted
                    completed_activity_id = f"{llm_activity_id}-c"
                    completed_event = LangChainGovernanceEvent(
                        source="workflow-telemetry",
                        event_type="LLMCompleted",
                        workflow_id=workflow_id,
                        run_id=run_id,
                        workflow_type=self._config.agent_name or "LangGraphRun",
                        task_queue=self._config.task_queue,
                        timestamp=rfc3339_now(),
                        session_id=self._config.session_id,
                        activity_id=completed_activity_id,
                        activity_type=llm_activity_type,
                        activity_output=gov_event.activity_output,
                        status="completed",
                        duration_ms=gov_event.duration_ms,
                        llm_model=gov_event.llm_model,
                        input_tokens=gov_event.input_tokens,
                        output_tokens=gov_event.output_tokens,
                        total_tokens=gov_event.total_tokens,
                        has_tool_calls=gov_event.has_tool_calls,
                        completion=gov_event.completion,
                        langgraph_node=gov_event.langgraph_node,
                        langgraph_step=gov_event.langgraph_step,
                    )

                    response = await self._client.evaluate_event(completed_event)
                    if response is not None:
                        context = lang_graph_event_to_context(event.event, is_root=is_root)
                        result = enforce_verdict(response, context)
                        if result.requires_hitl:
                            await poll_until_decision(
                                self._client,
                                HITLPollParams(
                                    workflow_id=workflow_id,
                                    run_id=run_id,
                                    activity_id=completed_activity_id,
                                    activity_type=llm_activity_type,
                                ),
                                self._config.hitl,
                            )
                return

        # ── Send to OpenBox Core
        response = await self._client.evaluate_event(gov_event)

        if response is None:
            return

        # ── Determine context and enforce verdict
        context = lang_graph_event_to_context(event.event, is_root=is_root)
        try:
            result = enforce_verdict(response, context)
        except (GovernanceBlockedError, GovernanceHaltError, GuardrailsValidationError):
            raise

        # ── HITL polling
        if result.requires_hitl:
            activity_id = gov_event.activity_id or event.run_id
            activity_type = gov_event.activity_type or event.name
            try:
                await poll_until_decision(
                    self._client,
                    HITLPollParams(
                        workflow_id=workflow_id,
                        run_id=run_id,
                        activity_id=activity_id,
                        activity_type=activity_type,
                    ),
                    self._config.hitl,
                )
            except (ApprovalRejectedError, ApprovalExpiredError, ApprovalTimeoutError) as e:
                raise GovernanceHaltError(str(e)) from e

    # ─────────────────────────────────────────────────────────────
    # Tool type classification
    # ─────────────────────────────────────────────────────────────

    def _resolve_tool_type(self, tool_name: str, subagent_name: str | None) -> str | None:
        """Resolve the semantic tool_type for a given tool.

        Priority:
        1. Explicit entry in tool_type_map
        2. "a2a" if subagent_name is set
        3. None for unknown tools (no classification prefix in the label)
        """
        if tool_name in self._config.tool_type_map:
            return self._config.tool_type_map[tool_name]
        if subagent_name:
            return "a2a"
        return None

    def _enrich_activity_input(
        self,
        base_input: list[Any] | None,
        tool_type: str | None,
        subagent_name: str | None,
    ) -> list[Any] | None:
        """Append an ``__openbox`` metadata entry to activity_input for Rego policy use.

        Core forwards ``activity_input`` as-is to ``input.activity_input`` in OPA.
        By appending a sentinel object, Rego policies can classify tools without
        any Core changes:

        .. code-block:: rego

            some item in input.activity_input
            meta := item["__openbox"]
            meta.subagent_name == "writer"

        Only appended when tool_type or subagent_name is set (skips for unclassified tools).
        """
        if tool_type is None and subagent_name is None:
            return base_input
        meta: dict[str, Any] = {}
        if tool_type is not None:
            meta["tool_type"] = tool_type
        if subagent_name is not None:
            meta["subagent_name"] = subagent_name
        result = list(base_input) if base_input else []
        result.append({"__openbox": meta})
        return result

    # ─────────────────────────────────────────────────────────────
    # Event mapping (LangGraph event → governance event)
    # ─────────────────────────────────────────────────────────────

    def _map_event(
        self,
        event: LangGraphStreamEvent,
        thread_id: str,
        workflow_id: str,
        run_id: str,
        root_tracker: _RootRunTracker,
        buffer: _RunBufferManager,
    ) -> tuple[LangChainGovernanceEvent | None, bool, bool, str]:
        """Map a LangGraph stream event to a governance event.

        Returns:
            A 4-tuple of (governance_event | None, is_root, is_start, event_type_label).
        """
        ev = event.event
        event_run_id = event.run_id   # LangGraph internal run UUID for this node/tool/llm
        name = event.name
        metadata = event.metadata
        data = event.data

        langgraph_node = metadata.get("langgraph_node")
        langgraph_step = metadata.get("langgraph_step")

        subagent_name = (
            self._resolve_subagent_name(event) if self._resolve_subagent_name else None
        )

        def base(
            event_type: str,
            *,
            is_start: bool,
            **extra: Any,
        ) -> tuple[LangChainGovernanceEvent, bool, bool, str]:
            is_root = root_tracker.root_run_id == event_run_id or (
                ev == "on_chain_start" and root_tracker.is_root(event_run_id)
            )
            gov = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type=event_type,
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                langgraph_node=langgraph_node,
                langgraph_step=langgraph_step,
                **extra,
            )
            return gov, is_root, is_start, event_type

        if ev == "on_chain_start":
            is_root = root_tracker.is_root(event_run_id)
            if is_root:
                buffer.register(
                    event_run_id, "graph", name, thread_id, langgraph_node, langgraph_step
                )
                gov = LangChainGovernanceEvent(
                    source="workflow-telemetry",
                    event_type="ChainStarted",
                    workflow_id=workflow_id,
                    run_id=run_id,
                    workflow_type=self._config.agent_name or name or "LangGraphRun",
                    task_queue=self._config.task_queue,
                    timestamp=rfc3339_now(),
                    session_id=self._config.session_id,
                    activity_id=event_run_id,
                    activity_type=name,
                    activity_input=(
                        [safe_serialize(data.get("input"))]
                        if data.get("input") is not None
                        else None
                    ),
                    langgraph_node=langgraph_node,
                    langgraph_step=langgraph_step,
                )
                return gov, True, True, "ChainStarted"
            # Non-root chain = subgraph node
            if name in self._config.skip_chain_types:
                return None, False, True, "ChainStarted"
            # Skip non-subagent chains — LangGraph fires BOTH on_chain_start
            # (BaseTool's Runnable layer) AND on_tool_start (Tool layer) for
            # the same tool invocation with different run_ids, creating
            # duplicate ActivityStarted events.  on_tool_start handles tools
            # with proper span hook context; only subagent chains need this.
            if not subagent_name:
                return None, False, True, "ChainStarted"
            buffer.register(
                event_run_id,
                "subgraph" if subagent_name else "chain",
                name,
                thread_id,
                langgraph_node,
                langgraph_step,
                subagent_name,
            )
            # Use ToolStarted so to_server_event_type maps to ActivityStarted,
            # NOT WorkflowStarted — sending WorkflowCompleted for a sub-chain
            # seals the session in Core and causes all subsequent requests to HALT.
            chain_tool_type = self._resolve_tool_type(name, subagent_name)
            chain_base_input = (
                [safe_serialize(data.get("input"))] if data.get("input") is not None else None
            )
            gov = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type="ToolStarted",
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                activity_id=event_run_id,
                activity_type=name,
                activity_input=self._enrich_activity_input(
                    chain_base_input, chain_tool_type, subagent_name
                ),
                tool_name=name,
                tool_type=chain_tool_type,
                langgraph_node=langgraph_node,
                langgraph_step=langgraph_step,
                subagent_name=subagent_name,
            )
            return gov, False, True, "ChainStarted"

        if ev == "on_chain_end":
            is_root = root_tracker.root_run_id == event_run_id
            dur = buffer.duration_ms(event_run_id)
            buffer.remove(event_run_id)
            output = data.get("output")
            serialized_output = (
                safe_serialize({"result": output})
                if isinstance(output, str)
                else safe_serialize(output)
            )
            if is_root:
                gov = LangChainGovernanceEvent(
                    source="workflow-telemetry",
                    event_type="ChainCompleted",
                    workflow_id=workflow_id,
                    run_id=run_id,
                    workflow_type=self._config.agent_name or name or "LangGraphRun",
                    task_queue=self._config.task_queue,
                    timestamp=rfc3339_now(),
                    session_id=self._config.session_id,
                    activity_id=event_run_id,
                    activity_type=name,
                    workflow_output=safe_serialize(output),
                    activity_output=serialized_output,
                    status="completed",
                    duration_ms=dur,
                    langgraph_node=langgraph_node,
                    langgraph_step=langgraph_step,
                )
                return gov, True, False, "ChainCompleted"
            if name in self._config.skip_chain_types:
                return None, False, False, "ChainCompleted"
            # Skip non-subagent chains (mirrors on_chain_start skip above)
            if not subagent_name:
                return None, False, False, "ChainCompleted"
            # Use ToolCompleted → ActivityCompleted (not WorkflowCompleted)
            chain_tool_type = self._resolve_tool_type(name, subagent_name)
            gov = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type="ToolCompleted",
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                activity_id=event_run_id,
                activity_type=name,
                activity_output=serialized_output,
                tool_name=name,
                tool_type=chain_tool_type,
                status="completed",
                duration_ms=dur,
                langgraph_node=langgraph_node,
                langgraph_step=langgraph_step,
                subagent_name=subagent_name,
            )
            return gov, False, False, "ChainCompleted"

        if ev == "on_tool_start":
            if name in self._config.skip_tool_types:
                return None, False, True, "ToolStarted"

            # Register activity context with SpanProcessor for hook-level governance
            # All tools (including subagents) get span-level governance
            if getattr(self, '_span_processor', None) is not None:
                activity_context = {
                    "source": "workflow-telemetry",
                    "event_type": "ActivityStarted",
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "workflow_type": self._config.agent_name or "LangGraphRun",
                    "task_queue": self._config.task_queue or "langgraph",
                    "activity_id": event_run_id,
                    "activity_type": name,
                }
                self._span_processor.set_activity_context(
                    workflow_id, event_run_id, activity_context
                )

            buffer.register(event_run_id, "tool", name, thread_id, langgraph_node, langgraph_step)
            buf = buffer.get(event_run_id)
            if buf is not None:
                buf.subagent_name = subagent_name

            # Create OTel span to propagate trace context across asyncio.Task boundaries.
            # Tool execution happens in a spawned Task with a new OTel context — this span
            # bridges the gap so httpx child spans inherit the correct trace_id.
            if getattr(self, '_span_processor', None) is not None:
                parent_ctx = otel_context.get_current()
                tool_span = _otel_tracer.start_span(
                    f"tool.{name}", context=parent_ctx, kind=otel_trace.SpanKind.INTERNAL,
                )
                token = otel_context.attach(otel_trace.set_span_in_context(tool_span))
                trace_id = tool_span.get_span_context().trace_id
                if trace_id:
                    self._span_processor.register_trace(trace_id, workflow_id, event_run_id)
                if buf is not None:
                    buf.otel_span = tool_span
                    buf.otel_token = token
            tool_input = _unwrap_tool_input(data.get("input"))
            tool_type = self._resolve_tool_type(name, subagent_name)
            # NOTE: No internal span here. In the Temporal SDK, @traced spans
            # fire DURING activity execution — after ActivityStarted is stored
            # in Core.  Firing here would race with the ToolStarted event below
            # (the hook span may arrive at Core before the parent event exists).
            # The "completed" internal span fires at on_tool_end instead.
            gov = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type="ToolStarted",
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                activity_id=event_run_id,
                activity_type=name,
                activity_input=self._enrich_activity_input(
                    [safe_serialize(tool_input)], tool_type, subagent_name
                ),
                tool_name=name,
                tool_type=tool_type,
                tool_input=safe_serialize(data.get("input")),
                subagent_name=subagent_name,
                langgraph_node=langgraph_node,
                langgraph_step=langgraph_step,
            )
            return gov, False, True, "ToolStarted"

        if ev == "on_tool_end":
            if name in self._config.skip_tool_types:
                return None, False, False, "ToolCompleted"
            dur = buffer.duration_ms(event_run_id)
            buf = buffer.get(event_run_id)

            # End OTel span created in on_tool_start and detach context
            if buf is not None and buf.otel_span is not None:
                if buf.otel_token is not None:
                    otel_context.detach(buf.otel_token)
                buf.otel_span.end()

            # Clear SpanProcessor activity context for all tools
            if getattr(self, '_span_processor', None) is not None:
                self._span_processor.clear_activity_context(workflow_id, event_run_id)

            buffer.remove(event_run_id)
            completed_activity_id = f"{event_run_id}-c"
            tool_output = data.get("output")
            serialized_output = (
                safe_serialize({"result": tool_output})
                if isinstance(tool_output, str)
                else safe_serialize(tool_output)
            )
            tool_type = self._resolve_tool_type(name, subagent_name)
            gov = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type="ToolCompleted",
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                activity_id=completed_activity_id,
                activity_type=name,
                activity_output=serialized_output,
                tool_name=name,
                tool_type=tool_type,
                subagent_name=subagent_name,
                status="completed",
                duration_ms=dur,
                langgraph_node=langgraph_node,
                langgraph_step=langgraph_step,
            )
            return gov, False, False, "ToolCompleted"

        if ev == "on_chat_model_start":
            buffer.register(event_run_id, "llm", name, thread_id, langgraph_node, langgraph_step)
            # Register activity context with OTel SpanProcessor for hook-level governance
            if getattr(self, '_span_processor', None) is not None:
                activity_context = {
                    "source": "workflow-telemetry",
                    "event_type": "ActivityStarted",
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "workflow_type": self._config.agent_name or "LangGraphRun",
                    "task_queue": self._config.task_queue or "langgraph",
                    "activity_id": event_run_id,
                    "activity_type": "llm_call",
                }
                self._span_processor.set_activity_context(
                    workflow_id, event_run_id, activity_context
                )

                # Create OTel span to propagate trace context across asyncio.Task boundaries
                parent_ctx = otel_context.get_current()
                llm_span = _otel_tracer.start_span(
                    "llm.call", context=parent_ctx, kind=otel_trace.SpanKind.INTERNAL,
                )
                token = otel_context.attach(otel_trace.set_span_in_context(llm_span))
                trace_id = llm_span.get_span_context().trace_id
                if trace_id:
                    self._span_processor.register_trace(trace_id, workflow_id, event_run_id)
                buf = buffer.get(event_run_id)
                if buf is not None:
                    buf.otel_span = llm_span
                    buf.otel_token = token
            messages = (data.get("input") or {}).get("messages", [])
            prompt_text = _extract_prompt_from_messages(messages)
            # Skip sending empty prompts — subagent-internal LLM calls have only
            # system/tool messages, no human turn. Core's guardrail JSON-parses the
            # prompt field and returns a parse error ("Expecting value ... char 0") → block.
            if not prompt_text.strip():
                return None, False, True, "LLMStarted"
            # Mark that LLMStarted will be sent — on_chat_model_end uses this to
            # decide whether to fire the LLM span hook.  Without this guard, span
            # hooks fire for internal subagent LLM calls that have no row in Core,
            # causing Core to create orphan empty rows (duplicate with no data).
            buf = buffer.get(event_run_id)
            if buf is not None:
                buf.llm_started = True
            model_name = _extract_model_name_from_event(event) or name
            gov = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type="LLMStarted",
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                activity_id=event_run_id,
                activity_type="llm_call",
                activity_input=[{"prompt": prompt_text}],
                llm_model=model_name,
                prompt=prompt_text,
                langgraph_node=langgraph_node,
                langgraph_step=langgraph_step,
            )
            return gov, False, True, "LLMStarted"

        if ev == "on_chat_model_end":
            dur = buffer.duration_ms(event_run_id)
            buf = buffer.get(event_run_id)
            llm_started = buf.llm_started if buf else False

            # End OTel span created in on_chat_model_start and detach context
            if buf is not None and buf.otel_span is not None:
                if buf.otel_token is not None:
                    otel_context.detach(buf.otel_token)
                buf.otel_span.end()

            buffer.remove(event_run_id)
            # Don't clear SpanProcessor activity context here — the tool (parent)
            # is still active. Clearing would break subagent LLM span attribution.
            # Context is cleared at on_tool_end instead.
            # Skip if LLMStarted was never sent (empty/no human-turn prompt).
            # Firing a hook_trigger span for a non-existent row creates an
            # orphan empty ActivityStarted row in Core.
            if not llm_started:
                return None, False, False, "LLMCompleted"
            llm_output = data.get("output") or {}
            input_tokens, output_tokens, total_tokens = _extract_token_usage(llm_output)
            completion_text = _extract_completion_text(llm_output)
            model_name = (
                _extract_model_name_from_output(llm_output)
                or _extract_model_name_from_event(event)
                or name
            )
            has_tool_calls = bool(_extract_tool_calls(llm_output))
            # NOTE: No span hook for LLM calls.  The user's hard rule:
            # "every activity started … not the LLM prompt should have a span call"
            # LLM events are explicitly excluded from the span requirement.
            # Additionally, the LLMStarted activity_id (from _pre_screen_input or
            # _GuardrailsCallbackHandler) doesn't reliably match event.run_id here,
            # so a span hook would create an orphan governance event (duplicate).
            gov = LangChainGovernanceEvent(
                source="workflow-telemetry",
                event_type="LLMCompleted",
                workflow_id=workflow_id,
                run_id=run_id,
                workflow_type=self._config.agent_name or "LangGraphRun",
                task_queue=self._config.task_queue,
                timestamp=rfc3339_now(),
                session_id=self._config.session_id,
                activity_id=event_run_id,
                activity_output=safe_serialize(llm_output),
                status="completed",
                duration_ms=dur,
                llm_model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                has_tool_calls=has_tool_calls,
                completion=completion_text,
                langgraph_node=langgraph_node,
                langgraph_step=langgraph_step,
            )
            return gov, False, False, "LLMCompleted"

        # Streaming chunks and other events — not governed
        return None, False, False, ""


# ═══════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════

def create_openbox_graph_handler(
    graph: Any,
    *,
    api_url: str,
    api_key: str,
    governance_timeout: float = 30.0,
    validate: bool = True,
    enable_telemetry: bool = True,
    sqlalchemy_engine: Any = None,
    **handler_kwargs: Any,
) -> OpenBoxLangGraphHandler:
    """Create a fully configured `OpenBoxLangGraphHandler` wrapping a compiled LangGraph graph.

    Calls the synchronous `initialize()` to validate credentials and set up global config,
    then returns a ready-to-use `OpenBoxLangGraphHandler`.

    Args:
        graph: A compiled LangGraph graph (e.g. `StateGraph.compile()`).
        api_url: Base URL of your OpenBox Core instance.
        api_key: API key in `obx_live_*` or `obx_test_*` format.
        governance_timeout: HTTP timeout in **seconds** for governance calls (default 30.0).
        validate: If True, validates the API key against the server on startup.
        enable_telemetry: Reserved for future HTTP-span telemetry patching.
        sqlalchemy_engine: Optional SQLAlchemy Engine instance to instrument for DB
            governance. Required when the engine is created before the handler.
        **handler_kwargs: Additional keyword arguments forwarded to
            `OpenBoxLangGraphHandlerOptions`.

    Returns:
        A configured `OpenBoxLangGraphHandler` ready to govern the graph.

    Example:
        >>> governed = create_openbox_graph_handler(
        ...     graph=my_graph,
        ...     api_url=os.environ["OPENBOX_URL"],
        ...     api_key=os.environ["OPENBOX_API_KEY"],
        ...     agent_name="MyAgent",
        ...     hitl={"enabled": True, "poll_interval_ms": 5000, "max_wait_ms": 300000},
        ... )
    """
    from openbox_langgraph.config import initialize
    initialize(
        api_url=api_url,
        api_key=api_key,
        governance_timeout=governance_timeout,
        validate=validate,
    )

    options = OpenBoxLangGraphHandlerOptions(
        api_timeout=governance_timeout,
        sqlalchemy_engine=sqlalchemy_engine,
        **{k: v for k, v in handler_kwargs.items() if hasattr(OpenBoxLangGraphHandlerOptions, k)},
    )
    return OpenBoxLangGraphHandler(graph, options)


# ═══════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════

def _extract_thread_id(config: dict[str, Any] | None) -> str:
    """Extract thread_id from a LangGraph RunnableConfig dict."""
    if not config:
        return "default"
    configurable = config.get("configurable") or {}
    return configurable.get("thread_id") or "default"


def _unwrap_tool_input(raw: Any) -> Any:
    """Unwrap potentially double-encoded JSON tool input."""
    import json

    if not isinstance(raw, str):
        return raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            if list(parsed.keys()) == ["input"] and isinstance(parsed["input"], str):
                return json.loads(parsed["input"])
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return raw


def _extract_prompt_from_messages(messages: Any) -> str:
    """Extract the last human/user message text from a LangChain messages structure."""
    if not isinstance(messages, (list, tuple)):
        return ""
    flat: list[Any] = []
    for item in messages:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    for msg in reversed(flat):
        if hasattr(msg, "content"):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get("content", "")
        else:
            continue
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            return " ".join(parts)
    return ""


def _extract_model_name_from_event(event: LangGraphStreamEvent) -> str | None:
    """Extract model name from event metadata."""
    return (
        event.metadata.get("ls_model_name")
        or event.metadata.get("model_name")
        or None
    )


def _extract_model_name_from_output(output: Any) -> str | None:
    """Extract model name from LLM output dict."""
    if not isinstance(output, dict):
        return None
    meta = output.get("response_metadata") or {}
    return meta.get("model_name") or meta.get("model") or output.get("model") or None


def _extract_token_usage(output: Any) -> tuple[int | None, int | None, int | None]:
    """Extract (input_tokens, output_tokens, total_tokens) from an LLM output dict."""
    if not isinstance(output, dict):
        return None, None, None
    usage = (
        output.get("usage_metadata") or output.get("response_metadata", {}).get("usage", {}) or {}
    )
    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
    total = usage.get("total_tokens") or (
        (input_tokens or 0) + (output_tokens or 0) if (input_tokens or output_tokens) else None
    )
    return input_tokens, output_tokens, total


def _extract_completion_text(output: Any) -> str | None:
    """Extract the assistant completion text from an LLM output dict."""
    if not isinstance(output, dict):
        return None
    # LangChain AIMessage structure
    content = output.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return " ".join(parts) if parts else None
    return None


def _extract_tool_calls(output: Any) -> list[Any]:
    """Return tool_calls list from an LLM output dict (empty list if none)."""
    if not isinstance(output, dict):
        return []
    tool_calls = output.get("tool_calls") or []
    if tool_calls:
        return tool_calls
    # LangChain AIMessage wraps tool_calls in additional_kwargs
    additional = output.get("additional_kwargs") or {}
    return additional.get("tool_calls") or []
