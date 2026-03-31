"""OpenBox LangGraph SDK — Governance HTTP Client."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC
from typing import Any

import httpx

from openbox_langgraph.errors import OpenBoxNetworkError
from openbox_langgraph.types import (
    ApprovalResponse,
    GovernanceVerdictResponse,
    LangChainGovernanceEvent,
    Verdict,
    parse_approval_response,
    to_server_event_type,
)

_SDK_VERSION = "0.1.0"


def build_auth_headers(api_key: str) -> dict[str, str]:
    """Build standard auth headers for governance API calls.

    Single source of truth — used by GovernanceClient and hook_governance.
    """
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": f"OpenBox-LangGraph-SDK/{_SDK_VERSION}",
        "X-OpenBox-SDK-Version": _SDK_VERSION,
    }


@dataclass
class ApprovalPollParams:
    """Parameters for an HITL approval poll request."""

    workflow_id: str
    run_id: str
    activity_id: str


class GovernanceClient:
    """Async HTTP client for the OpenBox Core governance API.

    Uses persistent httpx.AsyncClient instances (lazy-init) to avoid the
    overhead of creating a new TCP connection per governance call.
    """

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        timeout: float = 30.0,  # seconds
        on_api_error: str = "fail_open",
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout  # already in seconds
        self._on_api_error = on_api_error
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None
        self._cached_headers = build_auth_headers(api_key)
        # Deduplication: prevent sending the same (activity_id, event_type) twice
        # within the same workflow run. Keyed by (workflow_id, run_id) so it resets
        # automatically on each new ainvoke() call.
        self._dedup_run: tuple[str, str] | None = None
        self._dedup_sent: set[tuple[str, str]] = set()

    def _get_client(self) -> httpx.AsyncClient:
        """Return or create the persistent async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def _get_sync_client(self) -> httpx.Client:
        """Return or create the persistent sync HTTP client."""
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(timeout=self._timeout)
        return self._sync_client

    async def close(self) -> None:
        """Close the underlying HTTP clients."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()
        self._sync_client = None

    # ─────────────────────────────────────────────────────────────
    # Public methods
    # ─────────────────────────────────────────────────────────────

    async def validate_api_key(self) -> None:
        """Validate the API key against the server.

        Raises:
            OpenBoxAuthError: If the key is rejected (401/403).
            OpenBoxNetworkError: If the server is unreachable.
        """
        from openbox_langgraph.errors import OpenBoxAuthError

        try:
            client = self._get_client()
            response = await client.get(
                f"{self._api_url}/api/v1/auth/validate",
                headers=self._headers(),
            )
            if response.status_code in (401, 403):
                msg = "Invalid API key. Check your API key at dashboard.openbox.ai"
                raise OpenBoxAuthError(msg)
            if not response.is_success:
                msg = f"Cannot reach OpenBox Core at {self._api_url}: HTTP {response.status_code}"
                raise OpenBoxNetworkError(msg)
        except (OpenBoxAuthError, OpenBoxNetworkError):
            raise
        except Exception as e:
            msg = f"Cannot reach OpenBox Core at {self._api_url}: {e}"
            raise OpenBoxNetworkError(msg) from e

    def _is_duplicate(
        self, workflow_id: str, run_id: str, activity_id: str, event_type: str
    ) -> bool:
        """Return True if this (activity_id, event_type) was already sent in this run.

        Resets automatically when (workflow_id, run_id) changes — i.e. on each new
        ainvoke() call, which generates a fresh workflow_id + run_id pair.
        Hook events (evaluate_raw) are never checked here — multiple hooks per
        activity are expected and valid.
        """
        current_run = (workflow_id, run_id)
        if self._dedup_run != current_run:
            self._dedup_run = current_run
            self._dedup_sent = set()
        key = (activity_id, event_type)
        if key in self._dedup_sent:
            return True
        self._dedup_sent.add(key)
        return False

    async def evaluate_event(
        self, event: LangChainGovernanceEvent
    ) -> GovernanceVerdictResponse | None:
        """Send a governance event to OpenBox Core and return the verdict.

        Returns `None` on network failure when `on_api_error` is `fail_open`.
        Silently drops duplicate (activity_id, event_type) pairs within the same run.

        Args:
            event: The governance event payload to evaluate.

        Raises:
            OpenBoxNetworkError: On network failure when `on_api_error` is `fail_closed`.
        """
        server_event_type = to_server_event_type(event.event_type)
        if event.activity_id and self._is_duplicate(
            event.workflow_id, event.run_id, event.activity_id, server_event_type
        ):
            if os.environ.get("OPENBOX_DEBUG") == "1":
                print(
                    f"[OpenBox Debug] dedup: dropped duplicate {server_event_type}"
                    f" activity_id={event.activity_id}"
                )
            return None

        payload = event.to_dict()
        payload["event_type"] = server_event_type
        payload["task_queue"] = event.task_queue or "langgraph"
        payload["source"] = "workflow-telemetry"

        if os.environ.get("OPENBOX_DEBUG") == "1":
            import json
            print(
                f"[OpenBox Debug] governance request: {json.dumps(payload, indent=2, default=str)}"
            )

        try:
            client = self._get_client()
            response = await client.post(
                f"{self._api_url}/api/v1/governance/evaluate",
                headers=self._headers(),
                json=payload,
            )

            if not response.is_success:
                if self._on_api_error == "fail_closed":
                    msg = f"Governance API error: HTTP {response.status_code}"
                    raise OpenBoxNetworkError(msg)
                return None

            data = response.json()
            return GovernanceVerdictResponse.from_dict(data)

        except OpenBoxNetworkError:
            raise
        except Exception as e:
            if self._on_api_error == "fail_closed":
                msg = f"Governance API unreachable: {e}"
                raise OpenBoxNetworkError(msg) from e
            return None

    def evaluate_event_sync(
        self, event: LangChainGovernanceEvent
    ) -> GovernanceVerdictResponse | None:
        """Sync version of evaluate_event using httpx.Client.

        Used by sync middleware hooks (invoke/stream) to avoid asyncio.run()
        teardown killing the HTTP connection before Core finishes processing.
        """
        server_event_type = to_server_event_type(event.event_type)
        if event.activity_id and self._is_duplicate(
            event.workflow_id, event.run_id, event.activity_id, server_event_type
        ):
            return None

        payload = event.to_dict()
        payload["event_type"] = server_event_type
        payload["task_queue"] = event.task_queue or "langgraph"
        payload["source"] = "workflow-telemetry"

        if os.environ.get("OPENBOX_DEBUG") == "1":
            import json
            print(
                "[OpenBox Debug] sync governance request:"
                f" {json.dumps(payload, indent=2, default=str)}"
            )

        try:
            client = self._get_sync_client()
            response = client.post(
                f"{self._api_url}/api/v1/governance/evaluate",
                headers=self._headers(),
                json=payload,
            )

            if not response.is_success:
                if self._on_api_error == "fail_closed":
                    msg = f"Governance API error: HTTP {response.status_code}"
                    raise OpenBoxNetworkError(msg)
                return None

            data = response.json()
            return GovernanceVerdictResponse.from_dict(data)

        except OpenBoxNetworkError:
            raise
        except Exception as e:
            if self._on_api_error == "fail_closed":
                msg = f"Governance API unreachable: {e}"
                raise OpenBoxNetworkError(msg) from e
            return None

    async def poll_approval(
        self, params: ApprovalPollParams
    ) -> ApprovalResponse | None:
        """Poll for HITL approval status.

        Returns `None` on network failure so the caller can retry.

        Args:
            params: Identifiers for the pending approval.
        """
        try:
            client = self._get_client()
            response = await client.post(
                f"{self._api_url}/api/v1/governance/approval",
                headers=self._headers(),
                json={
                    "workflow_id": params.workflow_id,
                    "run_id": params.run_id,
                    "activity_id": params.activity_id,
                },
            )

            if not response.is_success:
                return None

            data = response.json()
            parsed = parse_approval_response(data)

            # SDK-side expiration check
            if parsed.approval_expiration_time and not parsed.expired:
                from datetime import datetime
                expiry = datetime.fromisoformat(
                    parsed.approval_expiration_time.replace("Z", "+00:00")
                )
                if expiry < datetime.now(tz=UTC):
                    parsed.expired = True

            return parsed

        except Exception:
            return None

    async def evaluate_raw(
        self, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Send a pre-built payload to the governance evaluate endpoint.

        Used by hook-level governance where the payload is fully assembled
        by the caller (no event_type translation needed).

        Args:
            payload: The raw dict to POST to `/api/v1/governance/evaluate`.
        """
        if os.environ.get("OPENBOX_DEBUG") == "1":
            import json
            print(
                f"[OpenBox Debug] span hook request: {json.dumps(payload, indent=2, default=str)}"
            )

        try:
            client = self._get_client()
            response = await client.post(
                f"{self._api_url}/api/v1/governance/evaluate",
                headers=self._headers(),
                json=payload,
            )

            if not response.is_success:
                if os.environ.get("OPENBOX_DEBUG") == "1":
                    print(
                        f"[OpenBox Debug] span hook error: HTTP {response.status_code}"
                        f" body={response.text[:500]}"
                    )
                if self._on_api_error == "fail_closed":
                    msg = f"Governance API error: HTTP {response.status_code}"
                    raise OpenBoxNetworkError(msg)
                return None

            data = response.json()
            return data  # type: ignore[no-any-return]

        except OpenBoxNetworkError:
            raise
        except Exception as e:
            if self._on_api_error == "fail_closed":
                msg = f"Governance API unreachable: {e}"
                raise OpenBoxNetworkError(msg) from e
            return None

    @staticmethod
    def halt_response(reason: str) -> GovernanceVerdictResponse:
        """Build a fail-closed HALT response for when the API is unreachable."""
        return GovernanceVerdictResponse(verdict=Verdict.HALT, reason=reason)

    # ─────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        return self._cached_headers
