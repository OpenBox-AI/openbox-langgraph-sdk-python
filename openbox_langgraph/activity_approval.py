"""Wait at the governed operation, preserving its stack and activity identity."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field

from openbox_core.client import EvaluationClient
from openbox_core.contracts.context import ActivityContext
from openbox_core.contracts.results import ApprovalResult, EvaluationResult, Verdict

from openbox_langgraph.errors import (
    ApprovalExpiredError,
    ApprovalRejectedError,
    OpenBoxConfigError,
    _raise_core_error,
)
from openbox_langgraph.types import HITLConfig


@dataclass
class _ApprovalTurn:
    cancelled: threading.Event = field(default_factory=threading.Event)
    decisions: dict[
        str, tuple[EvaluationResult, ApprovalExpiredError | ApprovalRejectedError | None]
    ] = field(default_factory=dict)


class ActivityApprovalWaiter:
    """Share approval decisions between the sync and async tool callbacks.

    The callbacks can enforce the same stashed evaluation twice. Only that
    exact evaluation is reusable: a later hook verdict on the same activity
    must get its own decision. All state is discarded when the turn ends.
    """

    def __init__(self, client: EvaluationClient, config: HITLConfig) -> None:
        self._client = client
        self._config = config
        self._lock = threading.Lock()
        self._turns: dict[tuple[str, str], _ApprovalTurn] = {}

    def begin_turn(self, workflow_id: str, run_id: str) -> None:
        with self._lock:
            self._turns[(workflow_id, run_id)] = _ApprovalTurn()

    def end_turn(self, workflow_id: str) -> None:
        with self._lock:
            for key in list(self._turns):
                if key[0] == workflow_id:
                    self._turns.pop(key).cancelled.set()

    def _turn(self, ctx: ActivityContext) -> _ApprovalTurn:
        workflow_id, run_id, _ = self._identity(ctx)
        with self._lock:
            turn = self._turns.get((workflow_id, run_id))
        if turn is None:
            raise OpenBoxConfigError("Approval requires an active governed turn")
        return turn

    @staticmethod
    def _identity(ctx: ActivityContext) -> tuple[str, str, str]:
        if not ctx.workflow_id or not ctx.run_id or not ctx.activity_id:
            raise OpenBoxConfigError("Approval requires workflow, run, and activity IDs")
        return ctx.workflow_id, ctx.run_id, ctx.activity_id

    @staticmethod
    def _check_cancelled(turn: _ApprovalTurn) -> None:
        # Cancelling an asyncio task does not stop its executor thread. Wake
        # the sync waiter too, so a late approval cannot run an abandoned tool.
        if turn.cancelled.is_set():
            raise asyncio.CancelledError

    def _reuse_decision(
        self, turn: _ApprovalTurn, ctx: ActivityContext, result: EvaluationResult
    ) -> bool:
        self._check_cancelled(turn)
        with self._lock:
            decision = turn.decisions.get(self._identity(ctx)[2])
        if decision is None or decision[0] is not result:
            return False
        if decision[1] is not None:
            raise decision[1]
        return True

    def _record_response(
        self,
        turn: _ApprovalTurn,
        ctx: ActivityContext,
        result: EvaluationResult,
        response: ApprovalResult | None,
    ) -> bool:
        self._check_cancelled(turn)
        try:
            approved = self._approved(response, ctx)
        except (ApprovalExpiredError, ApprovalRejectedError) as exc:
            with self._lock:
                turn.decisions[self._identity(ctx)[2]] = (result, exc)
            raise
        if not approved:
            return False
        with self._lock:
            turn.decisions[self._identity(ctx)[2]] = (result, None)
        return True

    @staticmethod
    def _approved(response: ApprovalResult | None, ctx: ActivityContext) -> bool:
        if response is None:
            return False
        if response.expired:
            raise ApprovalExpiredError(f"Approval expired for {ctx.activity_type}")
        if response.verdict in (Verdict.BLOCK, Verdict.HALT):
            raise ApprovalRejectedError(
                response.reason or f"Approval rejected for {ctx.activity_type}"
            )
        return response.verdict is Verdict.ALLOW

    async def wait(self, result: EvaluationResult, ctx: ActivityContext) -> None:
        turn = self._turn(ctx)
        if self._reuse_decision(turn, ctx, result):
            return
        while True:
            self._check_cancelled(turn)
            try:
                response = await self._client.apoll_approval(*self._identity(ctx))
            except Exception as exc:
                _raise_core_error(exc)
            self._check_cancelled(turn)
            if self._record_response(turn, ctx, result, response):
                return
            await asyncio.sleep(self._config.poll_interval_ms / 1000.0)

    def wait_sync(self, result: EvaluationResult, ctx: ActivityContext) -> None:
        turn = self._turn(ctx)
        if self._reuse_decision(turn, ctx, result):
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # A sync tool runs in LangChain's executor thread.
        else:
            raise OpenBoxConfigError(
                "A synchronous operation requiring approval cannot wait on the event-loop "
                "thread. Use its async API or run it with asyncio.to_thread()."
            )
        while True:
            self._check_cancelled(turn)
            try:
                response = self._client.poll_approval(*self._identity(ctx))
            except Exception as exc:
                _raise_core_error(exc)
            self._check_cancelled(turn)
            if self._record_response(turn, ctx, result, response):
                return
            turn.cancelled.wait(self._config.poll_interval_ms / 1000.0)
