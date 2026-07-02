"""Base instrumentation governs a covered family (HTTP) EXACTLY ONCE per
operation stage — never zero (a silent gap) and never twice (a double
evaluation). With the legacy in-repo hooks removed, base `openbox_core`
instrumentation is the ONLY hook runtime, so this test proves it is installed
exactly once and no second layer re-governs the same request.

Drives the REAL `OpenBoxLangGraphHandler.__init__` non-injected-client path
(the only path that wires `create_core_runtime`'s base instrumentation) so
installation is exercised for real, not re-implemented as a mock.

TWO separate local servers are required, not one: the governance API and the
"real operation" endpoint must live on DIFFERENT host:port pairs, because
base's `should_ignore_url` treats any URL sharing the configured `api_url`
prefix as self-instrumentation and silently skips it — a single shared server
would make every "operation" request also match the ignored-URL guard and
report zero evaluations regardless of correctness.

Exercises `asyncio.create_task` topology (LangGraph's actual tool-execution
shape — see `trace_context_registry.py`'s module docstring) to prove
exclusivity holds under the SAME execution pattern the fallback-shim tests
document as the hard case for context resolution.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import threading
from typing import Any

import pytest
import requests

pytest.importorskip("openbox_core")

from openbox_core.context import activity_scope
from openbox_core.contracts.context import ActivityContext

from openbox_langgraph.langgraph_handler import (
    OpenBoxLangGraphHandler,
    create_openbox_graph_handler,
)

_BOUND_CONTEXT = ActivityContext(
    workflow_id="wf-exclusivity",
    run_id="run-exclusivity",
    activity_id="act-exclusivity",
    activity_type="http_request",
)


class _CountingServer:
    """Loopback HTTP server counting GET hits — the governed "real
    operation" endpoint. A DIFFERENT server (below) plays the governance API
    so the self-instrumentation guard never shadows this one."""

    def __init__(self) -> None:
        self.hits = 0
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer.hits += 1
                body = b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:  # keep test output clean
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_port}/echo"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class _FakeGovernanceApi:
    """Loopback server answering the governance evaluate/auth-validate
    endpoints. Counts `/api/v1/governance/evaluate` POSTs — the assertion
    target for exclusivity."""

    def __init__(self) -> None:
        self.evaluate_hits = 0
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _respond_json(self, body: dict[str, Any]) -> None:
                payload = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:
                self._respond_json({"ok": True})

            def do_POST(self) -> None:
                if self.path.startswith("/api/v1/governance/evaluate"):
                    outer.evaluate_hits += 1
                length = int(self.headers.get("Content-Length", 0))
                if length:
                    self.rfile.read(length)
                self._respond_json({"verdict": "allow"})

            def log_message(self, *args: Any) -> None:  # keep test output clean
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_port}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def operation_server():
    srv = _CountingServer()
    yield srv
    srv.stop()


@pytest.fixture
def governance_api():
    srv = _FakeGovernanceApi()
    yield srv
    srv.stop()


def _teardown_handler(handler: OpenBoxLangGraphHandler) -> None:
    """Uninstall base instrumentation so this test never leaks global OTel/hook
    state into a later test in the same session."""
    if handler._core_runtime is not None:  # type: ignore[attr-defined]
        handler._core_runtime.uninstall_instrumentation()  # type: ignore[attr-defined]


class TestHttpFamilyExclusivityUnderFlag:
    def test_flag_true_governs_http_exactly_once_per_stage(
        self, operation_server, governance_api
    ) -> None:
        handler = create_openbox_graph_handler(
            graph=None,
            api_url=governance_api.url,
            api_key="obx_test_exclusivity",
            validate=False,
            use_core_instrumentation=True,
        )
        try:
            assert handler._core_runtime is not None  # type: ignore[attr-defined]
            store = handler._core_runtime.context_store  # type: ignore[attr-defined]
            with activity_scope(_BOUND_CONTEXT, store=store):
                before_op = operation_server.hits
                before_eval = governance_api.evaluate_hits
                response = requests.get(operation_server.url, timeout=5)
                assert response.status_code == 200

            # The real operation ran exactly once.
            assert operation_server.hits == before_op + 1
            # Governance was asked about it EXACTLY twice (started + completed
            # stage of the SAME operation) — never once (a stage silently
            # skipped by a last-writer-wins overwrite) and never four times
            # (both legacy AND base evaluating every stage independently).
            assert governance_api.evaluate_hits == before_eval + 2
        finally:
            _teardown_handler(handler)

    async def test_exclusivity_holds_under_asyncio_create_task_topology(
        self, operation_server, governance_api
    ) -> None:
        """LangGraph spawns tool execution as `asyncio.create_task` — proves
        the family-exclusivity switchboard holds under the SAME topology the
        fallback-shim tests document as the hard case, not just a
        directly-awaited call."""
        handler = create_openbox_graph_handler(
            graph=None,
            api_url=governance_api.url,
            api_key="obx_test_exclusivity_task",
            validate=False,
            use_core_instrumentation=True,
        )
        try:
            store = handler._core_runtime.context_store  # type: ignore[attr-defined]
            with activity_scope(_BOUND_CONTEXT, store=store):
                before_op = operation_server.hits
                before_eval = governance_api.evaluate_hits

                async def tool_task() -> requests.Response:
                    return await asyncio.to_thread(requests.get, operation_server.url, timeout=5)

                response = await asyncio.create_task(tool_task())
                assert response.status_code == 200

            assert operation_server.hits == before_op + 1
            assert governance_api.evaluate_hits == before_eval + 2
        finally:
            _teardown_handler(handler)
