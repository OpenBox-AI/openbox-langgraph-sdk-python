"""Core configuration and governance client tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from openbox_langgraph import config as config_module
from openbox_langgraph.client import ApprovalPollParams, GovernanceClient, build_auth_headers
from openbox_langgraph.errors import (
    OpenBoxAuthError,
    OpenBoxInsecureURLError,
    OpenBoxNetworkError,
)
from openbox_langgraph.types import LangChainGovernanceEvent, Verdict


class _AsyncClientStub:
    """Minimal async HTTP client stub for GovernanceClient tests."""

    def __init__(self, responses: list[httpx.Response] | None = None) -> None:
        self.responses = responses or []
        self.calls: list[dict[str, Any]] = []
        self.is_closed = False

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append({"method": "GET", "url": url, **kwargs})
        return self.responses.pop(0)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append({"method": "POST", "url": url, **kwargs})
        return self.responses.pop(0)

    async def aclose(self) -> None:
        self.is_closed = True


class _SyncClientStub:
    """Minimal sync HTTP client stub for GovernanceClient tests."""

    def __init__(self, responses: list[httpx.Response] | None = None) -> None:
        self.responses = responses or []
        self.calls: list[dict[str, Any]] = []
        self.is_closed = False

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append({"method": "POST", "url": url, **kwargs})
        return self.responses.pop(0)

    def close(self) -> None:
        self.is_closed = True


def _response(status_code: int, payload: dict[str, Any] | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://core.openbox.test")
    return httpx.Response(status_code, json=payload or {}, request=request)


def _event(activity_id: str = "activity-1") -> LangChainGovernanceEvent:
    return LangChainGovernanceEvent(
        source="workflow-telemetry",
        event_type="ToolStarted",
        workflow_id="workflow-1",
        run_id="run-1",
        workflow_type="AgentWorkflow",
        task_queue="langgraph",
        timestamp="2026-01-01T00:00:00.000Z",
        activity_id=activity_id,
        activity_type="search",
    )


def test_validate_api_key_format_accepts_expected_prefixes() -> None:
    assert config_module.validate_api_key_format("obx_live_abc123")
    assert config_module.validate_api_key_format("obx_test_abc_123")
    assert not config_module.validate_api_key_format("sk-test")


def test_validate_url_security_rejects_non_localhost_http() -> None:
    with pytest.raises(OpenBoxInsecureURLError):
        config_module.validate_url_security("http://api.openbox.test")

    config_module.validate_url_security("http://localhost:8080")
    config_module.validate_url_security("https://api.openbox.test")


def test_merge_config_normalizes_timeout_sets_and_hitl() -> None:
    merged = config_module.merge_config(
        {
            "api_timeout": 3_000,
            "skip_chain_types": ["retriever"],
            "skip_tool_types": ("http",),
            "root_node_names": {"root"},
            "hitl": {"enabled": False, "poll_interval_ms": 100},
            "tool_type_map": {"search": "http"},
        }
    )

    assert merged.api_timeout == 3.0
    assert merged.skip_chain_types == {"retriever"}
    assert merged.skip_tool_types == {"http"}
    assert merged.root_node_names == {"root"}
    assert not merged.hitl.enabled
    assert merged.hitl.poll_interval_ms == 100
    assert merged.tool_type_map == {"search": "http"}


def test_initialize_validates_and_stores_global_config(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, float]] = []

    def _validate(api_url: str, api_key: str, timeout: float) -> None:
        calls.append((api_url, api_key, timeout))

    monkeypatch.setattr(config_module, "_validate_api_key_with_server", _validate)

    config_module.initialize(
        "https://core.openbox.test/",
        "obx_test_valid",
        governance_timeout=12,
        validate=True,
    )

    global_config = config_module.get_global_config()
    assert global_config.api_url == "https://core.openbox.test"
    assert global_config.api_key == "obx_test_valid"
    assert global_config.governance_timeout == 12
    assert calls == [("https://core.openbox.test", "obx_test_valid", 12)]


def test_initialize_rejects_invalid_api_key_without_server_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_module,
        "_validate_api_key_with_server",
        lambda *_args: pytest.fail("server validation should not run"),
    )

    with pytest.raises(OpenBoxAuthError):
        config_module.initialize("https://core.openbox.test", "invalid", validate=True)


def test_build_auth_headers_masks_no_values_and_sets_sdk_headers() -> None:
    headers = build_auth_headers("obx_test_key")

    assert headers["Authorization"] == "Bearer obx_test_key"
    assert headers["Content-Type"] == "application/json"
    assert headers["User-Agent"].startswith("OpenBox-LangGraph-SDK/")
    assert headers["X-OpenBox-SDK-Version"]


async def test_validate_api_key_success_and_auth_failure() -> None:
    client = GovernanceClient(api_url="https://core.openbox.test", api_key="obx_test_key")
    client._client = _AsyncClientStub([_response(200)])
    await client.validate_api_key()

    failing = GovernanceClient(api_url="https://core.openbox.test", api_key="obx_test_key")
    failing._client = _AsyncClientStub([_response(403)])
    with pytest.raises(OpenBoxAuthError):
        await failing.validate_api_key()


async def test_evaluate_event_maps_payload_and_deduplicates() -> None:
    stub = _AsyncClientStub([_response(200, {"verdict": "allow"})])
    client = GovernanceClient(api_url="https://core.openbox.test", api_key="obx_test_key")
    client._client = stub

    result = await client.evaluate_event(_event())
    duplicate = await client.evaluate_event(_event())

    assert result is not None
    assert result.verdict == Verdict.ALLOW
    assert duplicate is None
    assert len(stub.calls) == 1
    assert stub.calls[0]["json"]["event_type"] == "ActivityStarted"
    assert stub.calls[0]["json"]["source"] == "workflow-telemetry"


async def test_evaluate_event_fail_closed_raises_network_error() -> None:
    client = GovernanceClient(
        api_url="https://core.openbox.test",
        api_key="obx_test_key",
        on_api_error="fail_closed",
    )
    client._client = _AsyncClientStub([_response(500)])

    with pytest.raises(OpenBoxNetworkError):
        await client.evaluate_event(_event())


def test_evaluate_event_sync_success_and_fail_open() -> None:
    client = GovernanceClient(api_url="https://core.openbox.test", api_key="obx_test_key")
    client._sync_client = _SyncClientStub([_response(200, {"verdict": "constrain"})])

    result = client.evaluate_event_sync(_event())

    assert result is not None
    assert result.verdict == Verdict.CONSTRAIN

    fail_open = GovernanceClient(api_url="https://core.openbox.test", api_key="obx_test_key")
    fail_open._sync_client = _SyncClientStub([_response(500)])
    assert fail_open.evaluate_event_sync(_event("activity-2")) is None


async def test_poll_approval_parses_expired_and_allows_network_fail_open() -> None:
    expired_at = (datetime.now(tz=UTC) - timedelta(minutes=1)).isoformat()
    client = GovernanceClient(api_url="https://core.openbox.test", api_key="obx_test_key")
    client._client = _AsyncClientStub(
        [
            _response(
                200,
                {
                    "verdict": "require_approval",
                    "approval_expiration_time": expired_at,
                },
            ),
            _response(500),
        ]
    )

    expired = await client.poll_approval(
        ApprovalPollParams("workflow-1", "run-1", "activity-1")
    )
    missing = await client.poll_approval(
        ApprovalPollParams("workflow-1", "run-1", "activity-1")
    )

    assert expired is not None
    assert expired.expired
    assert missing is None


async def test_evaluate_raw_and_close() -> None:
    async_stub = _AsyncClientStub([_response(200, {"ok": True})])
    sync_stub = _SyncClientStub()
    client = GovernanceClient(api_url="https://core.openbox.test", api_key="obx_test_key")
    client._client = async_stub
    client._sync_client = sync_stub

    result = await client.evaluate_raw({"event_type": "ActivityStarted"})
    await client.close()

    assert result == {"ok": True}
    assert async_stub.is_closed
    assert sync_stub.is_closed
