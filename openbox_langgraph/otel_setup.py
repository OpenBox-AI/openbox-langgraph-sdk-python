# openbox/otel_setup.py
"""Deprecated legacy OpenTelemetry hook installer — retained as an error shim.

The legacy in-repo hook governance (HTTP/DB/file body-capture + inline
gate evaluation) has been removed. Hook governance is now owned entirely by the
shared ``openbox_core`` base instrumentation, installed via
``create_core_runtime`` (an ``InstrumentationManager``) — the only hook runtime.

``setup_opentelemetry_for_governance`` remains importable/exported so pinned
callers get a CLEAR, actionable error instead of an ``ImportError``, but it no
longer installs anything: calling it raises ``OpenBoxConfigError``.
"""

from __future__ import annotations

from typing import Any

from openbox_langgraph.errors import OpenBoxConfigError

__all__ = ["setup_opentelemetry_for_governance"]


def setup_opentelemetry_for_governance(*_args: Any, **_kwargs: Any) -> None:
    """Deprecated no-op shim — raises ``OpenBoxConfigError``.

    Legacy in-repo OTel hook governance has been removed. Hook instrumentation
    is installed by ``create_core_runtime`` (base ``openbox_core``
    ``InstrumentationManager``) when a handler owns a core runtime — there is
    nothing for this function to set up.
    """
    raise OpenBoxConfigError(
        "setup_opentelemetry_for_governance has been removed. Legacy in-repo OTel "
        "hook governance no longer exists; hook governance is owned by openbox_core "
        "base instrumentation, installed automatically by create_openbox_graph_handler "
        "(use_core_instrumentation=True, the default). Remove this call."
    )
