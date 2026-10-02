"""Decision reuse, identity isolation, and fail-closed approval boundaries."""

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest
from openbox_core.client import EvaluationClient
from openbox_core.contracts.context import ActivityContext
from openbox_core.contracts.results import ApprovalResult, EvaluationResult, Verdict
from openbox_core.errors import OpenBoxAuthError as CoreAuthError

from openbox_langgraph.activity_approval import ActivityApprovalWaiter
from openbox_langgraph.errors import ApprovalRejectedError, OpenBoxAuthError, OpenBoxConfigError
from openbox_langgraph.types import HITLConfig


def setup_waiter():
    client = Mock(spec=EvaluationClient)
    client.apoll_approval = AsyncMock(return_value=ApprovalResult(verdict=Verdict.ALLOW))
    ctx = ActivityContext(workflow_id="workflow", run_id="run", activity_id="tool")
    waiter = ActivityApprovalWaiter(client, HITLConfig(poll_interval_ms=1))
    waiter.begin_turn(ctx.workflow_id, ctx.run_id)
    return waiter, client, ctx


async def test_only_the_exact_evaluation_reuses_a_decision():
    waiter, client, ctx = setup_waiter()
    result = EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL)
    await waiter.wait(result, ctx)
    waiter.wait_sync(result, ctx)  # The other callback reads the same stash.
    client.apoll_approval.assert_awaited_once_with("workflow", "run", "tool")
    client.poll_approval.assert_not_called()

    # A later hook can ask for approval under the same tool ID. Its new
    # evaluation cannot inherit the earlier grant or mutate the original.
    later = EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL)
    client.apoll_approval.return_value = ApprovalResult(verdict=Verdict.BLOCK)
    with pytest.raises(ApprovalRejectedError):
        await waiter.wait(later, ctx)
    with pytest.raises(ApprovalRejectedError):
        waiter.wait_sync(later, ctx)
    assert client.apoll_approval.await_count == 2
    client.poll_approval.assert_not_called()
    assert result.verdict is Verdict.REQUIRE_APPROVAL


async def test_sync_wait_on_loop_fails_before_polling():
    waiter, client, ctx = setup_waiter()
    with pytest.raises(OpenBoxConfigError, match=r"asyncio\.to_thread"):
        waiter.wait_sync(EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL), ctx)
    client.poll_approval.assert_not_called()


@pytest.mark.parametrize("sync", [False, True])
async def test_auth_failure_is_native_and_never_approves(sync):
    waiter, client, ctx = setup_waiter()
    client.apoll_approval.side_effect = CoreAuthError("not authorized")
    client.poll_approval.side_effect = CoreAuthError("not authorized")
    result = EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL)
    with pytest.raises(OpenBoxAuthError):
        if sync:
            await asyncio.to_thread(waiter.wait_sync, result, ctx)
        else:
            await waiter.wait(result, ctx)


async def test_ending_one_turn_does_not_release_or_cancel_another():
    waiter, client, ctx = setup_waiter()
    other = replace(ctx, workflow_id="other-workflow", run_id="other-run")
    waiter.begin_turn(other.workflow_id, other.run_id)
    client.apoll_approval.return_value = ApprovalResult(verdict=Verdict.CONSTRAIN)
    result = EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL)
    first = asyncio.create_task(waiter.wait(result, ctx))
    second = asyncio.create_task(waiter.wait(result, other))
    try:
        await asyncio.sleep(0.01)
        assert not first.done() and not second.done()
        waiter.end_turn(ctx.workflow_id)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, 1)
        assert not second.done()
        client.apoll_approval.return_value = ApprovalResult(verdict=Verdict.ALLOW)
        await asyncio.wait_for(second, 1)
    finally:
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.parametrize("missing", ["workflow_id", "run_id", "activity_id"])
async def test_missing_identity_never_polls(missing):
    waiter, client, ctx = setup_waiter()
    with pytest.raises(OpenBoxConfigError, match="IDs"):
        await waiter.wait(
            EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL), replace(ctx, **{missing: None})
        )
    client.apoll_approval.assert_not_called()
