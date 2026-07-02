"""OpenBox LangGraph SDK — Governance HTTP Client."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from openbox_core.client import check_expiration
from openbox_core.contracts.results import EvaluationResult
from openbox_core.contracts.results import Verdict as _CoreVerdict
from openbox_core.errors import ContractError as _CoreContractError
from openbox_core.errors import GovernanceAPIError as _CoreGovernanceAPIError
from openbox_core.errors import OpenBoxNetworkError as _CoreOpenBoxNetworkError

from openbox_langgraph.core_events import to_envelope
from openbox_langgraph.errors import OpenBoxConfigError, OpenBoxNetworkError
from openbox_langgraph.identity import (
    AgentIdentityConfig,
    create_agent_identity_headers,
    parse_optional_agent_identity_config,
)
from openbox_langgraph.types import (
    ApprovalResponse,
    GovernanceVerdictResponse,
    LangChainGovernanceEvent,
    Verdict,
    parse_approval_response,
    to_server_event_type,
)

if TYPE_CHECKING:
    from openbox_core.gate import GovernanceGate

_SDK_VERSION = "0.2.0"


def _network_fallback_result(on_api_error: str, msg: str) -> GovernanceVerdictResponse | None:
    """Apply the `on_api_error` policy to a NETWORK/transport failure.

    This is the ONLY place a `GovernanceClient` verdict call may return
    `None` — it corresponds exactly to a client-synthesized fallback
    (`EvaluationResult.fallback_allow`): `verdict=ALLOW`, `fallback_used=True`,
    and an EMPTY `raw` dict (nothing was ever parsed from a real body).

    A response BODY that happens to be `{"verdict": "block",
    "fallback_used": true}` never reaches this function — it is parsed by
    `_verdict_from_response_data` instead, whose `raw` is always the
    non-empty parsed dict, so the None-collapse below can never fire for it
    and the BLOCK is enforced normally.

    `msg` is the caller's fully-formatted message (kept caller-side so the
    two distinct failure messages — "Governance API error: HTTP {status}"
    for a bad response, "Governance API unreachable: {e}" for a raised
    exception — stay exactly as they were before this translation layer).
    """
    if on_api_error == "fail_closed":
        raise OpenBoxNetworkError(msg)
    result = EvaluationResult.fallback_allow(msg)
    return _collapse_client_synthesized_fallback(result)


def _verdict_from_response_data(data: dict[str, Any]) -> GovernanceVerdictResponse:
    """Parse a real HTTP response body into a `GovernanceVerdictResponse`.

    Routed through the base SDK's `EvaluationResult.from_dict` so `raw` is
    always the full parsed body — the non-empty `raw` is exactly what keeps
    `_collapse_client_synthesized_fallback` from ever mistaking a real
    (even oddly-shaped) Core response for a client-side fallback.
    """
    return GovernanceVerdictResponse.from_result(EvaluationResult.from_dict(data))


def _collapse_client_synthesized_fallback(
    result: EvaluationResult,
) -> GovernanceVerdictResponse | None:
    """Return `None` ONLY for a client-synthesized fail-open fallback.

    The three-part discriminator matches `EvaluationResult.fallback_allow`
    exactly and nothing else: `fallback_used=True` AND `verdict is ALLOW` AND
    `raw` is empty (no real body was ever parsed). A response body that
    happens to carry `fallback_used: true` alongside a blocking verdict, or
    alongside ALLOW but WITH a real (non-empty) body, is a real Core response
    and must be returned as a `GovernanceVerdictResponse` so callers enforce
    it — never silently collapsed to `None`/implicit-ALLOW.
    """
    if result.fallback_used and result.verdict is _CoreVerdict.ALLOW and not result.raw:
        return None
    return GovernanceVerdictResponse.from_result(result)


async def _gate_evaluate(
    gate: GovernanceGate, event: LangChainGovernanceEvent, on_api_error: str
) -> GovernanceVerdictResponse | None:
    """Evaluate one lifecycle event through the base SDK's strict gate.

    The single translation seam between `gate.aevaluate`'s base-SDK contract
    (`EvaluationResult`, `openbox_core` exceptions) and this SDK's own
    (`GovernanceVerdictResponse | None`, `openbox_langgraph.errors`) — every
    gate-routed call site in `evaluate_event` goes through this function so
    the translation is defined exactly once. Outcome policy (mirrors the legacy
    httpx path so the wired transport is behaviourally interchangeable):

    - `EvaluationResult.fallback_allow()` (client-synthesized fail-open on a
      NETWORK error): collapsed to `None` via the same discriminator used for
      the legacy path, so the pre-screen-`None` -> callback re-evaluation ->
      PII-redaction flow keeps firing regardless of which transport produced it.
    - `ContractError` (a malformed envelope — a bug in THIS SDK's own
      event->envelope mapping, raised pre-network by the strict gate, never
      from a Core response): ALWAYS a fail-open telemetry-drop (`None`),
      independent of `on_api_error`. Enforcing fail_closed here would let an
      SDK-side mapping defect block a user's graph for a reason their OWN policy
      never produced — strictly worse than dropping one governance event.
    - `GovernanceAPIError` / `OpenBoxNetworkError` (network-shaped failure) ->
      this SDK's `OpenBoxNetworkError`, same public exception the legacy path
      raises under fail_closed.
    - Any OTHER exception (e.g. a malformed Core 200 body the base parser
      cannot decode) is a transport-shaped fault, NOT a governance verdict:
      routed through `_network_fallback_result` so fail_open returns `None`
      (never crash the graph on a Core hiccup) and fail_closed raises
      `OpenBoxNetworkError` — matching the legacy httpx catch-all exactly.
    """
    try:
        result = await gate.aevaluate(to_envelope(event))
    except _CoreContractError:
        return None
    except (_CoreGovernanceAPIError, _CoreOpenBoxNetworkError) as e:
        raise OpenBoxNetworkError(str(e)) from e
    except Exception as e:
        return _network_fallback_result(on_api_error, f"Governance gate error: {e}")
    return _collapse_client_synthesized_fallback(result)


def build_auth_headers(
    api_key: str,
    *,
    method: str | None = None,
    pathname: str | None = None,
    body: bytes | str | None = None,
    agent_identity: AgentIdentityConfig | None = None,
) -> dict[str, str]:
    """Build standard auth headers for governance API calls.

    Single source of truth — used by GovernanceClient and hook_governance.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": f"OpenBox-LangGraph-SDK/{_SDK_VERSION}",
        "X-OpenBox-SDK-Version": _SDK_VERSION,
    }
    if agent_identity:
        if method is None or pathname is None:
            msg = "method and pathname are required when signing OpenBox requests."
            raise OpenBoxConfigError(msg)
        headers.update(
            create_agent_identity_headers(
                did=agent_identity.did,
                private_key=agent_identity.private_key,
                method=method,
                pathname=pathname,
                body=body,
            )
        )
    return headers


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
        agent_did: str | None = None,
        agent_private_key: str | None = None,
        gate: GovernanceGate | None = None,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout  # already in seconds
        self._on_api_error = on_api_error
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None
        self._agent_identity = parse_optional_agent_identity_config(
            did=agent_did,
            private_key=agent_private_key,
        )
        # Optional base-SDK gate. When wired (by the handler, from a core
        # runtime built off the SAME api_url/api_key/timeout/on_api_error),
        # `evaluate_event`'s ASYNC path routes lifecycle events through it
        # instead of this client's own httpx transport — see `evaluate_event`.
        # `None` (the default) preserves the exact legacy transport/serialization
        # for every existing caller that constructs a bare `GovernanceClient()`.
        # `evaluate_event_sync` (sync middleware hooks) is unaffected either way.
        self._gate = gate
        # Deduplication: prevent sending the same (activity_id, event_type) twice
        # within the same workflow run. Keyed by (workflow_id, run_id) so it resets
        # automatically on each new ainvoke() call. Shared by every evaluate_event
        # call site regardless of which transport (gate or legacy httpx) is active
        # for a given call — dedup is a client-level concern, not a transport one.
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
                headers=self._headers(
                    method="GET",
                    pathname="/api/v1/auth/validate",
                    body=b"",
                ),
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
        Silently drops duplicate (activity_id, event_type) pairs within the same run
        — this de-dup pre-check runs BEFORE either transport below, so it applies
        identically whether a `gate` is wired or not.

        When a `gate` was supplied at construction (see `__init__`), the event is
        routed through the base SDK's `EventEnvelope` + `GovernanceGate.aevaluate`
        instead of this client's own httpx transport — see `_gate_evaluate`.
        Overriding `evaluate_event` in a subclass (e.g. the golden-fixture
        harness's `RecordingGovernanceClient`) still fully intercepts either way,
        since the branch lives inside THIS method, never at a call site.

        Args:
            event: The governance event payload to evaluate.

        Raises:
            OpenBoxNetworkError: On network failure when `on_api_error` is `fail_closed`
                (from either transport).
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

        if os.environ.get("OPENBOX_DEBUG") == "1":
            import json

            print(
                f"[OpenBox Debug] governance request: "
                f"{json.dumps(event.to_dict(), indent=2, default=str)}"
            )

        if self._gate is not None:
            return await _gate_evaluate(self._gate, event, self._on_api_error)

        payload = event.to_dict()
        payload["event_type"] = server_event_type
        payload["task_queue"] = event.task_queue or "langgraph"
        payload["source"] = "workflow-telemetry"

        try:
            client = self._get_client()
            body = _json_body(payload)
            response = await client.post(
                f"{self._api_url}/api/v1/governance/evaluate",
                headers=self._headers(
                    method="POST",
                    pathname="/api/v1/governance/evaluate",
                    body=body,
                ),
                content=body,
            )

            if not response.is_success:
                return _network_fallback_result(
                    self._on_api_error, f"Governance API error: HTTP {response.status_code}"
                )

            data = response.json()
            return _verdict_from_response_data(data)

        except OpenBoxNetworkError:
            raise
        except Exception as e:
            return _network_fallback_result(
                self._on_api_error, f"Governance API unreachable: {e}"
            )

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
            body = _json_body(payload)
            response = client.post(
                f"{self._api_url}/api/v1/governance/evaluate",
                headers=self._headers(
                    method="POST",
                    pathname="/api/v1/governance/evaluate",
                    body=body,
                ),
                content=body,
            )

            if not response.is_success:
                return _network_fallback_result(
                    self._on_api_error, f"Governance API error: HTTP {response.status_code}"
                )

            data = response.json()
            return _verdict_from_response_data(data)

        except OpenBoxNetworkError:
            raise
        except Exception as e:
            return _network_fallback_result(
                self._on_api_error, f"Governance API unreachable: {e}"
            )

    async def poll_approval(self, params: ApprovalPollParams) -> ApprovalResponse | None:
        """Poll for HITL approval status.

        Returns `None` on network failure so the caller can retry.

        Args:
            params: Identifiers for the pending approval.
        """
        try:
            client = self._get_client()
            body = _json_body(
                {
                    "workflow_id": params.workflow_id,
                    "run_id": params.run_id,
                    "activity_id": params.activity_id,
                }
            )
            response = await client.post(
                f"{self._api_url}/api/v1/governance/approval",
                headers=self._headers(
                    method="POST",
                    pathname="/api/v1/governance/approval",
                    body=body,
                ),
                content=body,
            )

            if not response.is_success:
                return None

            data = response.json()
            # SDK-side expiration check — run on the raw dict BEFORE parsing
            # (matches openbox_core.client.check_expiration's own call order:
            # check_expiration(data) then ApprovalResult.from_dict(data)).
            # Handles ISO 'Z', ISO offset, and space-separated DB timestamp
            # formats; a malformed timestamp is logged and left un-flagged
            # rather than raised, so one bad timestamp string degrades to
            # "expiration not confirmed" instead of aborting the whole poll.
            check_expiration(data)
            return parse_approval_response(data)

        except Exception:
            return None

    async def evaluate_raw(self, payload: dict[str, Any]) -> dict[str, Any] | None:
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
            body = _json_body(payload)
            response = await client.post(
                f"{self._api_url}/api/v1/governance/evaluate",
                headers=self._headers(
                    method="POST",
                    pathname="/api/v1/governance/evaluate",
                    body=body,
                ),
                content=body,
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

    def _headers(self, *, method: str, pathname: str, body: bytes | str | None) -> dict[str, str]:
        return build_auth_headers(
            self._api_key,
            method=method,
            pathname=pathname,
            body=body,
            agent_identity=self._agent_identity,
        )


def _json_body(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
