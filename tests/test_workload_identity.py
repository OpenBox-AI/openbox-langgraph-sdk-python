"""IAM v3 integration through the published base client and real LangChain callbacks."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from langchain_core.messages import HumanMessage
from openbox_core.client import EvaluationClient
from openbox_core.context import activity_scope
from openbox_core.contracts.context import ActivityContext

from openbox_langgraph import (
    OpenBoxAuthError,
    OpenBoxConfigError,
    OpenBoxNetworkError,
    OpenBoxSigningError,
    create_openbox_graph_handler,
    get_global_config,
    initialize,
)
from openbox_langgraph.client import ApprovalPollParams, GovernanceClient
from openbox_langgraph.config import GovernanceConfig
from openbox_langgraph.core_runtime import create_core_runtime
from openbox_langgraph.types import LangChainGovernanceEvent, Verdict
from tests.golden.fake_agent_graphs import build_tool_call_graph

API_URL = "https://core.example.test"
API_KEY = "obx_test_workload"
ISSUER = "https://identity.example.test/realms/openbox"
TOKEN_URL = f"{ISSUER}/protocol/openid-connect/token"
BOOTSTRAP_PATH = "/api/v3/auth/bootstrap"
TOKEN_HEADER = "X-OpenBox-Workload-Token"


@pytest.fixture(scope="module")
def workload_key() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return (
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        .decode()
        .strip()
    )


class WorkloadWire:
    """Controlled Core/Keycloak responses; production SDK code builds every request."""

    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []
        self.clients: list[EvaluationClient] = []
        self.bootstrap_status = 200
        self.token_status = 200
        self.runtime_status = 200
        self.reject_payload: Callable[[dict[str, Any]], bool] | None = None
        self.runtime_body: dict[str, Any] = {"verdict": "allow"}
        self.approval_body: dict[str, Any] = {"action": "allow", "verdict": "block"}
        self.bootstrap_body: dict[str, Any] = {
            "bootstrap_version": 3,
            "contract_version": 3,
            "issuer": ISSUER,
            "token_endpoint": TOKEN_URL,
            "audience": "openbox-core",
            "client_id": "openbox-agent-test",
            "service_account_id": "22222222-2222-4222-8222-222222222222",
            "activation_version": "33333333-3333-4333-8333-333333333333",
            "identity_source": "openbox",
            "kid": "workload-key-1",
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if request.url.path == BOOTSTRAP_PATH:
            body = self.bootstrap_body
            if self.bootstrap_status == 409:
                body = {"reason_code": "workload_identity_unavailable"}
            return httpx.Response(self.bootstrap_status, json=body)
        if str(request.url) == TOKEN_URL:
            return httpx.Response(
                self.token_status,
                json={
                    "access_token": "test-workload-token",
                    "token_type": "Bearer",
                    "expires_in": 300,
                },
            )
        if request.url.path.endswith("/approval") and self.runtime_status == 200:
            return httpx.Response(200, json=self.approval_body)
        if self.reject_payload is not None and request.url.path.endswith("/evaluate"):
            if not self.reject_payload(json.loads(request.content)):
                return httpx.Response(200, json={"verdict": "allow"})
        return httpx.Response(self.runtime_status, json=self.runtime_body)


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> WorkloadWire:
    for prefix in ("OPENBOX", "OPENBOX_LANGGRAPH"):
        for suffix in (
            "WORKLOAD_PRIVATE_KEY",
            "AGENT_DID",
            "AGENT_PRIVATE_KEY",
            "AGENT_IDENTITY_METHOD",
            "OKTA_AGENT_PRIVATE_KEY",
        ):
            monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)
    server = WorkloadWire()
    transport = httpx.MockTransport(server)

    def make_client(*args: Any, **kwargs: Any) -> EvaluationClient:
        client = EvaluationClient(*args, **kwargs, transport=transport, async_transport=transport)
        server.clients.append(client)
        return client

    monkeypatch.setattr("openbox_core.client.EvaluationClient", make_client)
    monkeypatch.setattr("openbox_core.runtime.EvaluationClient", make_client)
    return server


def event(activity_id: str = "activity-1") -> LangChainGovernanceEvent:
    return LangChainGovernanceEvent(
        source="workflow-telemetry",
        event_type="ToolStarted",
        workflow_id="workflow-1",
        run_id="run-1",
        workflow_type="Agent",
        task_queue="langgraph",
        timestamp="2026-10-02T00:00:00Z",
        activity_id=activity_id,
        activity_type="tool_call",
    )


def test_startup_performs_v3_handshake_and_closes_client(
    wire: WorkloadWire,
    workload_key: str,
) -> None:
    initialize(API_URL, API_KEY, workload_private_key=workload_key)
    assert [str(call.url) for call in wire.calls] == [
        f"{API_URL}{BOOTSTRAP_PATH}",
        TOKEN_URL,
        f"{API_URL}/api/v3/auth/validate",
    ]
    bootstrap, token, validate = wire.calls
    assert bootstrap.headers["Authorization"] == f"Bearer {API_KEY}"
    assert TOKEN_HEADER not in bootstrap.headers
    assert "Authorization" not in token.headers
    form = parse_qs(token.content.decode())
    assert form["grant_type"] == ["client_credentials"]
    assert "client_assertion" in form
    assert validate.headers[TOKEN_HEADER] == "test-workload-token"
    assert validate.headers["X-OpenBox-SDK-Version"] == "openbox-langgraph-python-v1.2.0"
    assert workload_key not in repr(get_global_config())
    assert all(client._sync_client is None for client in wire.clients)


@pytest.mark.parametrize("source", ["explicit", "framework", "global"])
def test_key_precedence_and_validate_false(
    monkeypatch: pytest.MonkeyPatch,
    wire: WorkloadWire,
    workload_key: str,
    source: str,
) -> None:
    monkeypatch.setenv("OPENBOX_WORKLOAD_PRIVATE_KEY", workload_key)
    if source != "global":
        monkeypatch.setenv("OPENBOX_LANGGRAPH_WORKLOAD_PRIVATE_KEY", workload_key)
        monkeypatch.setenv("OPENBOX_WORKLOAD_PRIVATE_KEY", "invalid-lower-priority-key")
    if source == "explicit":
        monkeypatch.setenv("OPENBOX_LANGGRAPH_WORKLOAD_PRIVATE_KEY", "invalid-prefixed-key")
    initialize(
        API_URL,
        API_KEY,
        validate=False,
        workload_private_key=workload_key if source == "explicit" else None,
    )
    assert get_global_config().workload_private_key == workload_key
    assert wire.calls == []
    runtime = create_core_runtime(
        GovernanceConfig(),
        api_url=API_URL,
        api_key=API_KEY,
        workload_private_key=workload_key if source == "explicit" else None,
    )
    try:
        assert runtime.config.workload_private_key == workload_key
    finally:
        runtime.close()
    assert wire.calls == []


@pytest.mark.parametrize("invalid_key", ["", "not-a-pem", "private-key-must-not-appear"])
def test_invalid_key_fails_locally_even_without_server_validation(
    wire: WorkloadWire,
    invalid_key: str,
) -> None:
    with pytest.raises(OpenBoxConfigError, match="workload_private_key") as error:
        initialize(API_URL, API_KEY, validate=False, workload_private_key=invalid_key)
    if invalid_key:
        assert invalid_key not in str(error.value)
    assert wire.calls == []
    assert not get_global_config().is_configured()


def test_reinitialization_clears_workload_key(wire: WorkloadWire, workload_key: str) -> None:
    initialize(API_URL, API_KEY, validate=False, workload_private_key=workload_key)
    initialize(API_URL, API_KEY, validate=False)
    assert get_global_config().workload_private_key is None


def test_failed_startup_does_not_publish_configuration(
    wire: WorkloadWire,
    workload_key: str,
) -> None:
    wire.bootstrap_status = 401
    with pytest.raises(OpenBoxAuthError):
        initialize(API_URL, API_KEY, workload_private_key=workload_key)
    assert not get_global_config().is_configured()
    assert wire.clients[0]._sync_client is None


@pytest.mark.parametrize("identity_source", ["openbox", "okta", "entra"])
async def test_graph_callbacks_hooks_and_approval_share_workload_auth(
    wire: WorkloadWire,
    workload_key: str,
    identity_source: str,
) -> None:
    wire.bootstrap_body["identity_source"] = identity_source
    governed = create_openbox_graph_handler(
        build_tool_call_graph(),
        api_url=API_URL,
        api_key=API_KEY,
        workload_private_key=workload_key,
        validate=False,
    )
    runtime = governed._core_runtime
    assert runtime is not None
    assert governed._client._core_client is runtime.client
    try:
        result = await governed.ainvoke({"messages": [HumanMessage(content="hello")]})
        assert result["messages"][-1].content == "done"
        with activity_scope(
            ActivityContext(
                workflow_id="workflow-1",
                run_id="run-1",
                workflow_type="Agent",
                activity_id="hook-activity",
                activity_type="tool_call",
                task_queue="langgraph",
            ),
            store=runtime.context_store,
        ):

            def tool_response(request: httpx.Request) -> httpx.Response:
                assert TOKEN_HEADER not in request.headers
                return httpx.Response(200, text="tool result")

            with httpx.Client(transport=httpx.MockTransport(tool_response)) as tool_client:
                assert tool_client.get("https://tool.example.test/resource").status_code == 200

        await governed._client.validate_api_key()
        sync_result = governed._client.evaluate_event_sync(event())
        assert sync_result is not None and sync_result.verdict == Verdict.ALLOW
        assert await governed._client.evaluate_raw({"event_type": "ActivityCompleted"})
        approval = await governed._client.poll_approval(ApprovalPollParams("wf", "run", "act"))
        assert approval is not None and approval.verdict == Verdict.ALLOW

        calls = [call for call in wire.calls if call.url.path.endswith("/evaluate")]
        payloads = [json.loads(call.content) for call in calls]
        assert {"WorkflowStarted", "ActivityStarted", "ActivityCompleted"} <= {
            payload["event_type"] for payload in payloads
        }
        assert any(payload.get("activity_type") == "llm_call" for payload in payloads)
        assert any(payload.get("hook_trigger") for payload in payloads)
        runtime_calls = [
            call
            for call in wire.calls
            if call.url.host == "core.example.test" and call.url.path != BOOTSTRAP_PATH
        ]
        assert all(call.url.path.startswith("/api/v3/") for call in runtime_calls)
        assert all(call.headers[TOKEN_HEADER] == "test-workload-token" for call in runtime_calls)
        assert all(call.headers["Authorization"] == f"Bearer {API_KEY}" for call in runtime_calls)
        assert sum(call.url.path == BOOTSTRAP_PATH for call in wire.calls) == 1
        assert sum(str(call.url) == TOKEN_URL for call in wire.calls) == 1

        # Closing the facade must not close the runtime's borrowed client.
        await governed._client.close()
        await governed._client.validate_api_key()
        assert len(wire.clients) == 1
    finally:
        await runtime.aclose()
    assert runtime.client._sync_client is None and runtime.client._async_client is None


async def perform(client: GovernanceClient, operation: str) -> Any:
    if operation == "validate":
        return await client.validate_api_key()
    if operation == "sync":
        return client.evaluate_event_sync(event())
    if operation == "raw":
        return await client.evaluate_raw({"event_type": "ActivityStarted"})
    if operation == "approval":
        return await client.poll_approval(ApprovalPollParams("wf", "run", "act"))
    return await client.evaluate_event(event())


@pytest.mark.parametrize("entrypoint", ["ainvoke", "astream_governed", "astream_events"])
async def test_callback_authentication_failure_stops_graph(
    wire: WorkloadWire,
    workload_key: str,
    entrypoint: str,
) -> None:
    wire.runtime_status = 401
    wire.reject_payload = lambda payload: (
        payload.get("event_type") == "ActivityStarted"
        and payload.get("activity_type") != "llm_call"
        and not payload.get("hook_trigger")
    )
    governed = create_openbox_graph_handler(
        build_tool_call_graph(),
        api_url=API_URL,
        api_key=API_KEY,
        workload_private_key=workload_key,
        validate=False,
    )
    assert governed._core_runtime is not None
    try:
        with pytest.raises(OpenBoxAuthError):
            if entrypoint == "ainvoke":
                await governed.ainvoke({"messages": [HumanMessage(content="hello")]})
            else:
                async for _ in getattr(governed, entrypoint)(
                    {"messages": [HumanMessage(content="hello")]}
                ):
                    pass
    finally:
        await governed._core_runtime.aclose()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_hook_authentication_failure_stops_operation(
    wire: WorkloadWire,
    workload_key: str,
    asynchronous: bool,
) -> None:
    wire.runtime_status = 403
    wire.reject_payload = lambda payload: bool(payload.get("hook_trigger"))
    runtime = create_core_runtime(
        GovernanceConfig(), api_url=API_URL, api_key=API_KEY, workload_private_key=workload_key
    )
    effects: list[httpx.Request] = []

    def tool_response(request: httpx.Request) -> httpx.Response:
        effects.append(request)
        return httpx.Response(200)

    try:
        with (
            activity_scope(
                ActivityContext(
                    workflow_id="wf",
                    run_id="run",
                    activity_id="act",
                    activity_type="tool_call",
                ),
                store=runtime.context_store,
            ),
            pytest.raises(OpenBoxAuthError),
        ):
            transport = httpx.MockTransport(tool_response)
            if asynchronous:
                async with httpx.AsyncClient(transport=transport) as client:
                    await client.get("https://tool.example.test/resource")
            else:
                with httpx.Client(transport=transport) as client:
                    client.get("https://tool.example.test/resource")
        assert effects == []
    finally:
        await runtime.aclose()


@pytest.mark.parametrize("operation", ["validate", "async", "sync", "raw", "approval"])
@pytest.mark.parametrize("status", [401, 403])
async def test_rejected_identity_never_fails_open_or_stays_pending(
    wire: WorkloadWire,
    workload_key: str,
    operation: str,
    status: int,
) -> None:
    wire.runtime_status = status
    wire.runtime_body = {"reason_code": "workload_identity_revoked"}
    client = GovernanceClient(api_url=API_URL, api_key=API_KEY, workload_private_key=workload_key)
    try:
        with pytest.raises(OpenBoxSigningError) as error:
            await perform(client, operation)
        assert error.value.reason_code == "workload_identity_revoked"
        assert isinstance(error.value, OpenBoxAuthError)
        # An explicit retry of the same event must not turn the auth rejection
        # into a deduplicated None (implicit ALLOW).
        with pytest.raises(OpenBoxAuthError):
            await perform(client, operation)
        assert not any(call.url.path.startswith("/api/v1/") for call in wire.calls)
    finally:
        await client.close()
    assert wire.clients[0]._sync_client is None and wire.clients[0]._async_client is None


@pytest.mark.parametrize("operation", ["async", "approval"])
@pytest.mark.parametrize(
    "failure", ["bootstrap_auth", "bootstrap_outage", "metadata", "token_auth", "token_outage"]
)
async def test_bootstrap_and_token_failures_never_reach_governance(
    wire: WorkloadWire,
    workload_key: str,
    operation: str,
    failure: str,
) -> None:
    expected: type[Exception] = OpenBoxNetworkError
    if failure == "bootstrap_auth":
        wire.bootstrap_status = 403
        expected = OpenBoxAuthError
    elif failure == "bootstrap_outage":
        wire.bootstrap_status = 503
    elif failure == "metadata":
        wire.bootstrap_body["contract_version"] = 999
        expected = OpenBoxConfigError
    elif failure == "token_auth":
        wire.token_status = 401
        expected = OpenBoxAuthError
    else:
        wire.token_status = 503
    client = GovernanceClient(api_url=API_URL, api_key=API_KEY, workload_private_key=workload_key)
    try:
        with pytest.raises(expected):
            await perform(client, operation)
        assert all(
            call.url.path == BOOTSTRAP_PATH or str(call.url) == TOKEN_URL for call in wire.calls
        )
    finally:
        await client.close()


@pytest.mark.parametrize("status", [404, 409])
async def test_explicit_no_authority_keeps_base_legacy_compatibility(
    wire: WorkloadWire,
    workload_key: str,
    status: int,
) -> None:
    wire.bootstrap_status = status
    client = GovernanceClient(api_url=API_URL, api_key=API_KEY, workload_private_key=workload_key)
    try:
        await client.validate_api_key()
        assert wire.calls[-1].url.path == "/api/v1/auth/validate"
        assert TOKEN_HEADER not in wire.calls[-1].headers
        assert not any(str(call.url) == TOKEN_URL for call in wire.calls)
    finally:
        await client.close()


async def test_ordinary_transport_fallback_and_real_block_remain_distinct(
    wire: WorkloadWire,
    workload_key: str,
) -> None:
    client = GovernanceClient(api_url=API_URL, api_key=API_KEY, workload_private_key=workload_key)
    try:
        wire.runtime_status = 503
        assert await client.evaluate_event(event()) is None
        assert await client.evaluate_raw({"event_type": "ActivityStarted"}) is None
        wire.runtime_status = 200
        wire.runtime_body = {"verdict": "block", "reason": "policy", "fallback_used": True}
        result = await client.evaluate_event(event("activity-2"))
        assert result is not None and result.verdict == Verdict.BLOCK
        assert await client.evaluate_raw({"event_type": "ActivityStarted"}) == wire.runtime_body
    finally:
        await client.close()


async def test_token_refresh_is_owned_by_base_client(wire: WorkloadWire, workload_key: str) -> None:
    client = GovernanceClient(api_url=API_URL, api_key=API_KEY, workload_private_key=workload_key)
    try:
        await client.validate_api_key()
        base = wire.clients[0]
        base._workload_token = replace(
            base._workload_token, expires_at=datetime.now(UTC) - timedelta(seconds=1)
        )
        await client.poll_approval(ApprovalPollParams("wf", "run", "act"))
        assert sum(str(call.url) == TOKEN_URL for call in wire.calls) == 2
        assert wire.calls[-1].url.path == "/api/v3/governance/approval"
    finally:
        await client.close()
