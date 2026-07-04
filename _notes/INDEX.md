# _notes index

- [decision-base-only-hook-runtime](decision-base-only-hook-runtime.md) — legacy in-repo hooks removed; base openbox_core is the only hook runtime; pathlib/io.open coverage TODO belongs in base.
- [decision-toolnode-seam-activity-binding](decision-toolnode-seam-activity-binding.md) — bind activity at ToolNode seam by WRITING config["run_id"] (not reading — it's None there); zero-fallback plain ContextStore; strict-mode sharp edge with handle_tool_errors.
