"""Tests verifying that the per-span governance payload and span shape
match the canonical schema exactly.

Tests _build_http_span_data (span shape) and _build_payload (payload envelope)
from the actual implementation modules.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

from openbox_langgraph.http_governance_hooks import _build_http_span_data
from openbox_langgraph.hook_governance import _build_payload, extract_span_context

HEX16 = re.compile(r"^[0-9a-f]{16}$")
HEX32 = re.compile(r"^[0-9a-f]{32}$")


# ── helpers ──────────────────────────────────────────────────────────────────

def _mock_span(
    span_id: int = 0xABCDEF0123456789,
    trace_id: int = 0xABCDEF0123456789ABCDEF0123456789,
    name: str = "HTTP GET",
    attributes: dict | None = None,
) -> MagicMock:
    """Create a mock OTel span with valid span_context."""
    span = MagicMock()
    span.name = name
    span.attributes = attributes or {}

    ctx = MagicMock()
    ctx.span_id = span_id
    ctx.trace_id = trace_id
    span.get_span_context.return_value = ctx
    span.context = ctx

    # No parent by default
    span.parent = None
    return span


def _build_completed_http_span(
    method: str = "GET",
    url: str = "https://example.com/api",
    status_code: int = 200,
    duration_ms: float = 50.0,
) -> dict:
    """Build a completed HTTP span_data via _build_http_span_data."""
    span = _mock_span()
    return _build_http_span_data(
        span, method, url, "completed",
        request_body=None, request_headers=None,
        response_body=None, response_headers=None,
        http_status_code=status_code, duration_ms=duration_ms,
    )


def _build_payload_with_span(span_data: dict) -> dict:
    """Build a full governance payload wrapping a span_data dict."""
    span = _mock_span()
    mock_processor = MagicMock()
    mock_processor.get_activity_context_by_trace.return_value = {
        "workflow_id": "wf-123",
        "run_id": "run-456",
        "activity_id": "act-789",
        "activity_type": "my_tool",
        "workflow_type": "MyAgent",
        "task_queue": "langgraph",
        "source": "workflow-telemetry",
    }

    with patch("openbox_langgraph.hook_governance._span_processor", mock_processor):
        payload = _build_payload(span, span_data)
    return payload


# ── span shape tests ─────────────────────────────────────────────────────────


def test_span_base_fields_present():
    """Every required base span field must be present."""
    span_data = _build_completed_http_span()

    required = {
        "span_id", "trace_id", "parent_span_id",
        "name", "kind", "stage",
        "start_time", "end_time", "duration_ns",
        "attributes", "status", "events",
        "hook_type", "error",
    }
    missing = required - set(span_data.keys())
    assert not missing, f"Missing span base fields: {missing}"


def test_span_http_fields_present():
    """HTTP-specific span fields must all be present."""
    span_data = _build_completed_http_span()

    http_fields = {
        "http_method", "http_url",
        "request_body", "request_headers",
        "response_body", "response_headers",
        "http_status_code",
    }
    missing = http_fields - set(span_data.keys())
    assert not missing, f"Missing HTTP span fields: {missing}"


def test_span_id_formats():
    """span_id must be 16-hex, trace_id must be 32-hex."""
    span_data = _build_completed_http_span()

    assert HEX16.match(span_data["span_id"]), f"span_id not 16-hex: {span_data['span_id']!r}"
    assert HEX32.match(span_data["trace_id"]), f"trace_id not 32-hex: {span_data['trace_id']!r}"


def test_span_kind_and_hook_type():
    """kind must be CLIENT, hook_type must be http_request."""
    span_data = _build_completed_http_span()

    assert span_data["kind"] == "CLIENT"
    assert span_data["hook_type"] == "http_request"


def test_span_stage_completed():
    """stage must be 'completed' for a finished HTTP call."""
    span_data = _build_completed_http_span()
    assert span_data["stage"] == "completed"


def test_span_stage_started():
    """stage must be 'started' for a pre-request call."""
    span = _mock_span()
    span_data = _build_http_span_data(
        span, "GET", "https://example.com/api", "started",
    )
    assert span_data["stage"] == "started"
    assert span_data["end_time"] is None


def test_span_times_are_ints():
    """start_time, end_time, duration_ns must be integers (nanoseconds)."""
    span_data = _build_completed_http_span(duration_ms=50.0)

    assert isinstance(span_data["start_time"], int)
    assert isinstance(span_data["end_time"], int)
    assert isinstance(span_data["duration_ns"], int)
    assert span_data["duration_ns"] == 50_000_000  # 50ms in ns


def test_span_status_shape():
    """status must be {"code": str, "description": str|None}."""
    span_data = _build_completed_http_span()

    assert isinstance(span_data["status"], dict)
    assert "code" in span_data["status"]
    assert "description" in span_data["status"]
    assert isinstance(span_data["status"]["code"], str)


def test_span_success_status():
    """2xx response → status.code=UNSET, error=None."""
    span_data = _build_completed_http_span(status_code=200)

    assert span_data["status"]["code"] == "UNSET"
    assert span_data["status"]["description"] is None
    assert span_data["error"] is None
    assert span_data["http_status_code"] == 200


def test_span_error_status_4xx():
    """4xx response → status.code=ERROR, error set."""
    span_data = _build_completed_http_span(status_code=404)

    assert span_data["status"]["code"] == "ERROR"
    assert span_data["status"]["description"] == "HTTP 404"
    assert span_data["error"] == "HTTP 404"
    assert span_data["http_status_code"] == 404


def test_span_error_status_5xx():
    """5xx response → status.code=ERROR, error set."""
    span_data = _build_completed_http_span(status_code=500)

    assert span_data["status"]["code"] == "ERROR"
    assert span_data["error"] == "HTTP 500"


def test_span_http_method_url():
    """http_method and http_url must be set correctly."""
    span_data = _build_completed_http_span(method="POST", url="https://api.example.com/submit")

    assert span_data["http_method"] == "POST"
    assert span_data["http_url"] == "https://api.example.com/submit"


def test_span_events_is_list():
    """events must be a list."""
    span_data = _build_completed_http_span()
    assert isinstance(span_data["events"], list)


def test_span_nullable_fields_none_by_default():
    """request_body, request_headers, response_body, response_headers
    must be None when not provided."""
    span_data = _build_completed_http_span()

    assert span_data["request_body"] is None
    assert span_data["request_headers"] is None
    assert span_data["response_body"] is None
    assert span_data["response_headers"] is None


def test_span_no_extra_fields():
    """Span must not contain fields outside the HTTP span spec."""
    span_data = _build_completed_http_span()

    allowed = {
        # base
        "span_id", "trace_id", "parent_span_id",
        "name", "kind", "stage",
        "start_time", "end_time", "duration_ns",
        "attributes", "status", "events",
        "hook_type", "error",
        # http
        "http_method", "http_url",
        "request_body", "request_headers",
        "response_body", "response_headers",
        "http_status_code",
    }
    extra = set(span_data.keys()) - allowed
    assert not extra, f"Unexpected span fields: {extra}"


# ── payload envelope tests ───────────────────────────────────────────────────


def test_payload_top_level_fields():
    """Payload must contain required top-level keys."""
    span_data = _build_completed_http_span()
    payload = _build_payload_with_span(span_data)

    assert payload is not None
    required = {
        "workflow_id", "run_id", "activity_id",
        "spans", "span_count", "hook_trigger", "timestamp",
    }
    missing = required - set(payload.keys())
    assert not missing, f"Missing top-level keys: {missing}"


def test_payload_hook_trigger_is_true():
    """hook_trigger must be boolean True."""
    span_data = _build_completed_http_span()
    payload = _build_payload_with_span(span_data)

    assert payload["hook_trigger"] is True


def test_payload_values():
    """Payload field values must match the injected activity context."""
    span_data = _build_completed_http_span()
    payload = _build_payload_with_span(span_data)

    assert payload["workflow_id"] == "wf-123"
    assert payload["run_id"] == "run-456"
    assert payload["activity_id"] == "act-789"
    assert payload["span_count"] == 1
    assert len(payload["spans"]) == 1
    assert isinstance(payload["timestamp"], str) and len(payload["timestamp"]) > 0


def test_payload_span_is_span_data():
    """The span embedded in the payload must be the span_data we passed in."""
    span_data = _build_completed_http_span(method="DELETE", url="https://example.com/item/1")
    payload = _build_payload_with_span(span_data)

    embedded = payload["spans"][0]
    assert embedded["http_method"] == "DELETE"
    assert embedded["http_url"] == "https://example.com/item/1"
    assert embedded["hook_type"] == "http_request"


def test_payload_none_without_activity_context():
    """_build_payload returns None if no activity context exists."""
    span = _mock_span()
    mock_processor = MagicMock()
    mock_processor.get_activity_context_by_trace.return_value = None

    with patch("openbox_langgraph.hook_governance._span_processor", mock_processor):
        payload = _build_payload(span, {})
    assert payload is None


# ── extract_span_context tests ───────────────────────────────────────────────


def test_extract_span_context_valid():
    """extract_span_context returns correct hex strings for valid spans."""
    span = _mock_span(span_id=0x1234567890ABCDEF, trace_id=0xFEDCBA9876543210FEDCBA9876543210)
    span_id, trace_id, parent_span_id = extract_span_context(span)

    assert span_id == "1234567890abcdef"
    assert trace_id == "fedcba9876543210fedcba9876543210"
    assert parent_span_id is None


def test_extract_span_context_with_parent():
    """extract_span_context returns parent_span_id when parent exists."""
    span = _mock_span()
    parent = MagicMock()
    parent.span_id = 0x9876543210ABCDEF
    span.parent = parent

    _, _, parent_span_id = extract_span_context(span)
    assert parent_span_id == "9876543210abcdef"


def test_extract_span_context_invalid_span():
    """extract_span_context returns zero-filled IDs for invalid spans."""
    span = MagicMock()
    span.get_span_context.return_value = None
    span.context = None
    span.parent = None

    span_id, trace_id, parent_span_id = extract_span_context(span)
    assert span_id == "0" * 16
    assert trace_id == "0" * 32
    assert parent_span_id is None
