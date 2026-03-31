"""Tests for recent governance changes:

1. httpx hook refactoring — hooks now store span ref only, governance in _patched_send
2. agent_validatePrompt removal — all activity_type="agent_validatePrompt" replaced with "on_chat_model_start"
3. Subagent event-level-only governance — subagent_name propagation via _map_event
4. _prepare_completed_governance builds correct (span, url, span_data) tuple
"""

import os

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from openbox_langgraph import http_governance_hooks
from openbox_langgraph import langgraph_handler


# ═══════════════════════════════════════════════════════════════════
# Test 1: httpx hooks are span-ref-only (no governance calls in request/response hooks)
# ═══════════════════════════════════════════════════════════════════

class TestHttpxHooksSpanRefOnly:
    """Verify that httpx request/response hooks store span refs."""

    def test_httpx_request_hook_stores_span(self):
        """_httpx_request_hook should set _httpx_http_span."""
        with patch.object(http_governance_hooks._otel, '_span_processor', MagicMock()):
            mock_span = MagicMock()
            mock_span.name = "test_span"
            mock_request = MagicMock()
            mock_request.url = "https://api.example.com/data"

            http_governance_hooks._httpx_request_hook(mock_span, mock_request)

            stored = http_governance_hooks._httpx_http_span.get(None)
            assert stored is mock_span

    def test_httpx_request_hook_ignores_openbox_url(self):
        """_httpx_request_hook should skip ignored URLs."""
        with patch.object(http_governance_hooks._otel, '_span_processor', MagicMock()):
            with patch.object(http_governance_hooks._otel, '_ignored_url_prefixes', ['https://core.openbox.ai']):
                mock_span = MagicMock()
                mock_request = MagicMock()
                mock_request.url = "https://core.openbox.ai/v1/governance"

                http_governance_hooks._httpx_http_span.set(None)
                http_governance_hooks._httpx_request_hook(mock_span, mock_request)

                stored = http_governance_hooks._httpx_http_span.get(None)
                assert stored is None

    def test_httpx_response_hook_is_noop(self):
        """_httpx_response_hook should be a no-op (governance handled in _patched_send)."""
        result = http_governance_hooks._httpx_response_hook(MagicMock(), MagicMock(), MagicMock())
        assert result is None

    async def test_httpx_async_request_hook_stores_span(self):
        """_httpx_async_request_hook should store span ref."""
        with patch.object(http_governance_hooks._otel, '_span_processor', MagicMock()):
            mock_span = MagicMock()
            mock_request = MagicMock()
            mock_request.url = "https://api.example.com/data"

            http_governance_hooks._httpx_http_span.set(None)
            await http_governance_hooks._httpx_async_request_hook(mock_span, mock_request)

            stored = http_governance_hooks._httpx_http_span.get(None)
            assert stored is mock_span

    async def test_httpx_async_response_hook_is_noop(self):
        """_httpx_async_response_hook should be a no-op."""
        result = await http_governance_hooks._httpx_async_response_hook(
            MagicMock(), MagicMock(), MagicMock()
        )
        assert result is None


# ═══════════════════════════════════════════════════════════════════
# Test 2: No "agent_validatePrompt" string in source code
# ═══════════════════════════════════════════════════════════════════

class TestAgentValidatePromptRemoved:
    """Verify that 'agent_validatePrompt' has been completely removed from source."""

    def test_no_agent_validatePrompt_in_langgraph_handler(self):
        module_path = langgraph_handler.__file__
        with open(module_path, 'r') as f:
            content = f.read()
        assert 'agent_validatePrompt' not in content

    def test_no_agent_validatePrompt_in_http_governance_hooks(self):
        module_path = http_governance_hooks.__file__
        with open(module_path, 'r') as f:
            content = f.read()
        assert 'agent_validatePrompt' not in content

    def test_all_openbox_langgraph_modules_no_agent_validatePrompt(self):
        import glob
        openbox_dir = os.path.dirname(langgraph_handler.__file__)
        py_files = glob.glob(os.path.join(openbox_dir, '*.py'))

        for py_file in py_files:
            with open(py_file, 'r') as f:
                content = f.read()
            assert 'agent_validatePrompt' not in content, \
                f"Found 'agent_validatePrompt' in {py_file}"

    def test_on_chat_model_start_present_in_handler(self):
        module_path = langgraph_handler.__file__
        with open(module_path, 'r') as f:
            content = f.read()
        assert content.count('on_chat_model_start') > 0


# ═══════════════════════════════════════════════════════════════════
# Test 3: Subagent _map_event propagation
# ═══════════════════════════════════════════════════════════════════

class TestSubagentMapEvent:
    """Verify _map_event propagates subagent_name correctly."""

    def test_subagent_tool_start_sets_subagent_name(self):
        """on_tool_start with subagent_name should set it on governance event."""
        from openbox_langgraph.langgraph_handler import (
            OpenBoxLangGraphHandler, OpenBoxLangGraphHandlerOptions,
            _RootRunTracker, _RunBufferManager,
        )
        from openbox_langgraph.types import LangGraphStreamEvent

        opts = OpenBoxLangGraphHandlerOptions(
            resolve_subagent_name=lambda e: "my_subagent" if e.name == "test_tool" else None,
            client=MagicMock(),
        )
        opts.client.evaluate_event = AsyncMock(return_value=None)
        handler = OpenBoxLangGraphHandler(MagicMock(), opts)

        event = MagicMock(spec=LangGraphStreamEvent)
        event.event = "on_tool_start"
        event.run_id = "run-123"
        event.name = "test_tool"
        event.metadata = {"langgraph_node": "tool_node", "langgraph_step": 1}
        event.data = {"input": {"arg": "value"}}

        tracker = _RootRunTracker()
        buffer = _RunBufferManager()

        gov_event, is_root, is_start, label = handler._map_event(
            event, "thread-1", "workflow-1", "run-1", tracker, buffer
        )

        assert gov_event is not None
        assert label == "ToolStarted"
        assert gov_event.subagent_name == "my_subagent"

    def test_regular_tool_start_has_no_subagent_name(self):
        """on_tool_start without subagent should have subagent_name=None."""
        from openbox_langgraph.langgraph_handler import (
            OpenBoxLangGraphHandler, OpenBoxLangGraphHandlerOptions,
            _RootRunTracker, _RunBufferManager,
        )
        from openbox_langgraph.types import LangGraphStreamEvent

        opts = OpenBoxLangGraphHandlerOptions(
            resolve_subagent_name=lambda e: None,
            client=MagicMock(),
        )
        opts.client.evaluate_event = AsyncMock(return_value=None)
        handler = OpenBoxLangGraphHandler(MagicMock(), opts)

        event = MagicMock(spec=LangGraphStreamEvent)
        event.event = "on_tool_start"
        event.run_id = "run-456"
        event.name = "regular_tool"
        event.metadata = {"langgraph_node": "tool_node", "langgraph_step": 1}
        event.data = {"input": {"arg": "value"}}

        tracker = _RootRunTracker()
        buffer = _RunBufferManager()

        gov_event, is_root, is_start, label = handler._map_event(
            event, "thread-1", "workflow-1", "run-1", tracker, buffer
        )

        assert gov_event is not None
        assert label == "ToolStarted"
        assert gov_event.subagent_name is None

    def test_subagent_tool_end_preserves_subagent_name(self):
        """on_tool_end for subagent tool should carry subagent_name from buffer."""
        from openbox_langgraph.langgraph_handler import (
            OpenBoxLangGraphHandler, OpenBoxLangGraphHandlerOptions,
            _RootRunTracker, _RunBufferManager,
        )
        from openbox_langgraph.types import LangGraphStreamEvent

        opts = OpenBoxLangGraphHandlerOptions(
            resolve_subagent_name=lambda e: "my_subagent" if e.name == "test_tool" else None,
            client=MagicMock(),
        )
        opts.client.evaluate_event = AsyncMock(return_value=None)
        handler = OpenBoxLangGraphHandler(MagicMock(), opts)
        handler._span_processor = MagicMock()

        event = MagicMock(spec=LangGraphStreamEvent)
        event.event = "on_tool_end"
        event.run_id = "run-789"
        event.name = "test_tool"
        event.metadata = {"langgraph_node": "tool_node", "langgraph_step": 1}
        event.data = {"output": "result"}

        tracker = _RootRunTracker()
        buffer = _RunBufferManager()
        buffer.register(event.run_id, "tool", event.name, "thread-1", None, None, "my_subagent")

        gov_event, is_root, is_start, label = handler._map_event(
            event, "thread-1", "workflow-1", "run-1", tracker, buffer
        )

        assert gov_event is not None
        assert label == "ToolCompleted"
        assert gov_event.subagent_name == "my_subagent"


# ═══════════════════════════════════════════════════════════════════
# Test 4: _prepare_completed_governance builds correct tuple or None
# ═══════════════════════════════════════════════════════════════════

class TestPrepareCompletedGovernance:
    """Verify _prepare_completed_governance builds correct tuple when applicable."""

    def test_returns_none_when_not_configured(self):
        with patch.object(http_governance_hooks._hook_gov, 'is_configured', return_value=False):
            result = http_governance_hooks._prepare_completed_governance(
                MagicMock(), MagicMock(), "https://api.example.com",
                None, None, None, None, 200
            )
            assert result is None

    def test_returns_none_when_url_is_none(self):
        with patch.object(http_governance_hooks._hook_gov, 'is_configured', return_value=True):
            result = http_governance_hooks._prepare_completed_governance(
                MagicMock(), MagicMock(), None,
                None, None, None, None, 200
            )
            assert result is None

    def test_returns_none_when_span_is_none(self):
        with patch.object(http_governance_hooks._hook_gov, 'is_configured', return_value=True):
            result = http_governance_hooks._prepare_completed_governance(
                None, MagicMock(), "https://api.example.com",
                None, None, None, None, 200
            )
            assert result is None

    def test_returns_tuple_when_all_configured(self):
        with patch.object(http_governance_hooks._hook_gov, 'is_configured', return_value=True):
            mock_span = MagicMock()
            mock_span.attributes = {}
            mock_span.name = "http_response"
            mock_request = MagicMock()
            mock_request.method = "POST"

            with patch.object(http_governance_hooks._hook_gov, 'extract_span_context',
                              return_value=("span_id", "trace_id", None)):
                result = http_governance_hooks._prepare_completed_governance(
                    mock_span, mock_request, "https://api.example.com",
                    "request_body", {"Content-Type": "application/json"},
                    "response_body", {"Content-Type": "application/json"},
                    200, duration_ms=100.5
                )

                assert result is not None
                assert len(result) == 3
                span, url, span_data = result
                assert span is mock_span
                assert url == "https://api.example.com"
                assert isinstance(span_data, dict)
                assert span_data['stage'] == "completed"
                assert span_data['http_status_code'] == 200
