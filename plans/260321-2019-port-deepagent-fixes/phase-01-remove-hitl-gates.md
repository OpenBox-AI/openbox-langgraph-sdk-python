# Phase 1: Remove hitl.enabled Gate + hitl.skip_tool_types

**Priority:** high
**Status:** planned
**Effort:** small

## Overview

Remove the `hitl.enabled` check and `hitl.skip_tool_types` from all HITL polling call sites in `langgraph_handler.py`. Matches the deepagent SDK changes.

## Related Code Files

- **Modify:** `openbox_langgraph/langgraph_handler.py`

## Changes

### 1. Line 558 — pre-screen HITL
```python
# Before:
if result and result.requires_hitl and self._config.hitl.enabled:

# After:
if result and result.requires_hitl:
```

### 2. Lines 845-846 — LLM completed HITL
```python
# Before:
if result.requires_hitl and self._config.hitl.enabled:
    if llm_activity_type not in (self._config.hitl.skip_tool_types or set()):

# After:
if result.requires_hitl:
```
Remove the `skip_tool_types` nesting — just poll directly.

### 3. Lines 873-876 — general event HITL
```python
# Before:
if result.requires_hitl and self._config.hitl.enabled:
    ...
    if activity_type not in (self._config.hitl.skip_tool_types or set()):

# After:
if result.requires_hitl:
```
Remove the `skip_tool_types` nesting.

## Success Criteria

- `REQUIRE_APPROVAL` always triggers polling (no enabled gate)
- `skip_tool_types` on `GovernanceConfig` (not `HITLConfig`) controls tool exclusion
- All existing tests pass
