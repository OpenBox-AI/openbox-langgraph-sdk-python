"""Shared fixtures for the core-runtime dual-write / context-binding test
modules (test_core_context_binding.py, test_context_cleanup_on_error.py,
test_context_fallback_task_and_thread.py, test_context_turn_isolation.py).

Two problems these fixtures solve, both discovered empirically while writing
those tests:

1. `OpenBoxLangGraphHandler.__init__` installs the REAL (process-wide) OTel
   HTTP governance hooks (`setup_opentelemetry_for_governance`) whenever
   `get_global_config()` returns a populated `api_url`/`api_key` —
   REGARDLESS of whether `opts.client` was injected (that only swaps
   `self._client`/`self._core_runtime`, a separate branch in `__init__`).
   `test_config_client_core.py::test_initialize_validates_and_stores_global_config`
   calls the real `initialize()` earlier in the SAME pytest session, leaving
   that global singleton populated for every test that runs after it. Left
   unguarded, this caused a genuine infinite-recursion hang: the installed
   `http_governance_hooks._patched_send` monkeypatch re-entered itself
   through `hook_governance.evaluate_sync`'s own outbound `httpx.Client.post`
   once `test_did_client_signing.py::test_hook_governance_signs_exact_body`'s
   raw `httpx.Client` use ran under that patch.

2. Under OTel's default no-op `TracerProvider`, every span gets
   `trace_id=0` — the handler's own `if trace_id:` guard treats that as "no
   span was created" and skips the dual-write registration entirely. These
   tests need REAL trace ids to assert dual-write/fallback/cleanup behavior
   on, exactly like `setup_opentelemetry_for_governance` would provide
   outside tests (but without pulling in its network/global-monkeypatch side
   effects — see fixture 1).
"""

from __future__ import annotations

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider

from openbox_langgraph.config import _GlobalConfigState, get_global_config


@pytest.fixture(autouse=True)
def _unconfigured_global_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the global api_url/api_key singleton to unconfigured for every test."""
    monkeypatch.setattr(
        "openbox_langgraph.config._global_config", _GlobalConfigState(), raising=True
    )
    assert not get_global_config().is_configured()


def _ensure_recording_tracer_provider() -> None:
    """Install a real (exporter-free) `TracerProvider` if none is set yet.

    `opentelemetry.trace.get_tracer(...)` returns a lazy proxy that
    re-resolves against the CURRENT global provider at `start_span()` time —
    not the provider active when `get_tracer()` was first called (verified
    empirically) — so this is safe to run after `langgraph_handler.py`'s
    module-level `_otel_tracer` already exists.

    Deliberately registers ZERO span processors/exporters — `set_tracer_provider`
    is a documented one-time global (subsequent calls are silent no-ops), so
    this never accumulates state or exports anything across the test session,
    unlike the OTel-HTTP-hook global pollution fixture 1's docstring documents
    and avoids.
    """
    current = otel_trace.get_tracer_provider()
    if not isinstance(current, TracerProvider):
        otel_trace.set_tracer_provider(TracerProvider())


_ensure_recording_tracer_provider()
