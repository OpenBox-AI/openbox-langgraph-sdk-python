"""`_gate_evaluate` translates base-SDK outcomes into this SDK's contract.

The single seam between `gate.aevaluate` (base `EvaluationResult` + `openbox_core`
exceptions) and the handler's expectation (`GovernanceVerdictResponse | None` +
`openbox_langgraph.errors`). Tested with fake gates so each outcome is forced
deterministically:

- client-synthesized fail-open fallback -> None (keeps the pre-screen-None ->
  callback re-evaluation -> PII-redaction path firing);
- ``GovernanceAPIError`` (network failure) -> ``OpenBoxNetworkError``;
- ``ContractError`` (a malformed envelope = an SDK-side mapping bug) -> None
  (fail-open telemetry-drop, independent of ``on_api_error`` — an SDK defect must
  not block a user's graph at the trust boundary);
- any OTHER error (e.g. a malformed Core 200 body the base parser can't decode)
  -> routed through the ``on_api_error`` policy like the legacy transport:
  fail_open -> None (never crash the graph), fail_closed -> OpenBoxNetworkError;
- a real BLOCK result -> an enforced verdict, never collapsed to None.
"""

from __future__ import annotations

import pytest
from openbox_core.contracts.results import EvaluationResult
from openbox_core.contracts.results import Verdict as CoreVerdict
from openbox_core.errors import ContractError, GovernanceAPIError
from openbox_core.errors import OpenBoxAuthError as CoreAuthError
from openbox_core.errors import OpenBoxConfigError as CoreConfigError
from openbox_core.errors import OpenBoxSigningError as CoreSigningError

from openbox_langgraph.client import GovernanceClient, _gate_evaluate
from openbox_langgraph.errors import (
    OpenBoxAuthError,
    OpenBoxConfigError,
    OpenBoxNetworkError,
    OpenBoxSigningError,
)
from openbox_langgraph.types import LangChainGovernanceEvent, Verdict


def _event() -> LangChainGovernanceEvent:
    return LangChainGovernanceEvent(
        source="workflow-telemetry",
        event_type="ToolStarted",
        workflow_id="wf-1",
        run_id="run-1",
        workflow_type="Agent",
        task_queue="langgraph",
        timestamp="2026-07-02T00:00:00Z",
        activity_id="act-1",
        activity_type="tool_call",
    )


class _Gate:
    """Minimal gate stand-in: returns a fixed result or raises a fixed error."""

    def __init__(
        self, *, result: EvaluationResult | None = None, exc: Exception | None = None
    ) -> None:
        self._result = result
        self._exc = exc

    async def aevaluate(self, _envelope: object) -> EvaluationResult:
        if self._exc is not None:
            raise self._exc
        assert self._result is not None
        return self._result


@pytest.mark.parametrize(
    "error, expected",
    [
        (CoreAuthError("rejected"), OpenBoxAuthError),
        (CoreSigningError("revoked", "workload_identity_revoked"), OpenBoxSigningError),
        (CoreConfigError("invalid bootstrap"), OpenBoxConfigError),
    ],
)
async def test_auth_and_configuration_errors_never_become_fallback_allow(error, expected):
    with pytest.raises(expected):
        await _gate_evaluate(_Gate(exc=error), _event(), "fail_open")


async def test_rejected_gate_event_is_not_deduplicated_into_allow() -> None:
    client = GovernanceClient(
        api_url="https://core.example.test",
        api_key="obx_test_key",
        gate=_Gate(exc=CoreAuthError("rejected")),
    )
    for _ in range(2):
        with pytest.raises(OpenBoxAuthError):
            await client.evaluate_event(_event())


@pytest.mark.asyncio
async def test_contract_error_drops_to_none_even_under_fail_closed() -> None:
    gate = _Gate(exc=ContractError("malformed envelope"))
    resp = await _gate_evaluate(gate, _event(), "fail_closed")  # type: ignore[arg-type]
    assert resp is None


@pytest.mark.asyncio
async def test_governance_api_error_becomes_langgraph_network_error() -> None:
    gate = _Gate(exc=GovernanceAPIError("core unreachable"))
    with pytest.raises(OpenBoxNetworkError):
        await _gate_evaluate(gate, _event(), "fail_closed")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_client_synthesized_fallback_collapses_to_none() -> None:
    gate = _Gate(result=EvaluationResult.fallback_allow("network fallback"))
    resp = await _gate_evaluate(gate, _event(), "fail_open")  # type: ignore[arg-type]
    assert resp is None


@pytest.mark.asyncio
async def test_unexpected_error_fail_open_returns_none() -> None:
    # A malformed Core 200 body raises a non-network error deep in the base
    # parser; under fail_open it must degrade to None, never crash the graph.
    gate = _Gate(exc=ValueError("malformed 200 body"))
    resp = await _gate_evaluate(gate, _event(), "fail_open")  # type: ignore[arg-type]
    assert resp is None


@pytest.mark.asyncio
async def test_unexpected_error_fail_closed_raises_network_error() -> None:
    gate = _Gate(exc=ValueError("malformed 200 body"))
    with pytest.raises(OpenBoxNetworkError):
        await _gate_evaluate(gate, _event(), "fail_closed")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_real_block_result_is_enforced() -> None:
    blocked = EvaluationResult(verdict=CoreVerdict.BLOCK, reason="policy", raw={"verdict": "block"})
    resp = await _gate_evaluate(_Gate(result=blocked), _event(), "fail_open")  # type: ignore[arg-type]
    assert resp is not None
    assert resp.verdict == Verdict.BLOCK
