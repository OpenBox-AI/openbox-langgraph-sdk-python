"""Tests for governance behavior that survives the base-SDK migration:

1. `agent_validatePrompt` removal — no leftover references in the package.
2. Subagent event-level governance — subagent_name propagation via `_map_event`.

Legacy httpx-hook tests (span-ref-only request/response hooks,
`_prepare_completed_governance`) were removed with `http_governance_hooks`;
hook governance is now owned entirely by the base `openbox_core` instrumentation.
"""

import os
from unittest.mock import AsyncMock, MagicMock

from openbox_langgraph import langgraph_handler

# ═══════════════════════════════════════════════════════════════════
# No "agent_validatePrompt" string anywhere in the package
# ═══════════════════════════════════════════════════════════════════

class TestAgentValidatePromptRemoved:
    """Verify that 'agent_validatePrompt' has been completely removed from source."""

    def test_no_agent_validatePrompt_in_langgraph_handler(self):
        module_path = langgraph_handler.__file__
        with open(module_path) as f:
            content = f.read()
        assert 'agent_validatePrompt' not in content

    def test_all_openbox_langgraph_modules_no_agent_validatePrompt(self):
        import glob
        openbox_dir = os.path.dirname(langgraph_handler.__file__)
        py_files = glob.glob(os.path.join(openbox_dir, '*.py'))

        for py_file in py_files:
            with open(py_file) as f:
                content = f.read()
            assert 'agent_validatePrompt' not in content, \
                f"Found 'agent_validatePrompt' in {py_file}"

    def test_on_chat_model_start_present_in_handler(self):
        module_path = langgraph_handler.__file__
        with open(module_path) as f:
            content = f.read()
        assert content.count('on_chat_model_start') > 0


# ═══════════════════════════════════════════════════════════════════
# Subagent _map_event propagation (injected-client, lifecycle-only handler)
# ═══════════════════════════════════════════════════════════════════

class TestSubagentMapEvent:
    """Verify _map_event propagates subagent_name correctly."""

    def test_subagent_tool_start_sets_subagent_name(self):
        """on_tool_start with subagent_name should set it on governance event."""
        from openbox_langgraph.langgraph_handler import (
            OpenBoxLangGraphHandler,
            OpenBoxLangGraphHandlerOptions,
            _RootRunTracker,
            _RunBufferManager,
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

        gov_event, _is_root, _is_start, label = handler._map_event(
            event, "thread-1", "workflow-1", "run-1", tracker, buffer
        )

        assert gov_event is not None
        assert label == "ToolStarted"
        assert gov_event.subagent_name == "my_subagent"

    def test_regular_tool_start_has_no_subagent_name(self):
        """on_tool_start without subagent should have subagent_name=None."""
        from openbox_langgraph.langgraph_handler import (
            OpenBoxLangGraphHandler,
            OpenBoxLangGraphHandlerOptions,
            _RootRunTracker,
            _RunBufferManager,
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

        gov_event, _is_root, _is_start, label = handler._map_event(
            event, "thread-1", "workflow-1", "run-1", tracker, buffer
        )

        assert gov_event is not None
        assert label == "ToolStarted"
        assert gov_event.subagent_name is None

    def test_subagent_tool_end_preserves_subagent_name(self):
        """on_tool_end for subagent tool should carry subagent_name from buffer."""
        from openbox_langgraph.langgraph_handler import (
            OpenBoxLangGraphHandler,
            OpenBoxLangGraphHandlerOptions,
            _RootRunTracker,
            _RunBufferManager,
        )
        from openbox_langgraph.types import LangGraphStreamEvent

        opts = OpenBoxLangGraphHandlerOptions(
            resolve_subagent_name=lambda e: "my_subagent" if e.name == "test_tool" else None,
            client=MagicMock(),
        )
        opts.client.evaluate_event = AsyncMock(return_value=None)
        handler = OpenBoxLangGraphHandler(MagicMock(), opts)

        event = MagicMock(spec=LangGraphStreamEvent)
        event.event = "on_tool_end"
        event.run_id = "run-789"
        event.name = "test_tool"
        event.metadata = {"langgraph_node": "tool_node", "langgraph_step": 1}
        event.data = {"output": "result"}

        tracker = _RootRunTracker()
        buffer = _RunBufferManager()
        buffer.register(event.run_id, "tool", event.name, "thread-1", None, None, "my_subagent")

        gov_event, _is_root, _is_start, label = handler._map_event(
            event, "thread-1", "workflow-1", "run-1", tracker, buffer
        )

        assert gov_event is not None
        assert label == "ToolCompleted"
        assert gov_event.subagent_name == "my_subagent"
