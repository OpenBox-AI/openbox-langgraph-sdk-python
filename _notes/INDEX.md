# _notes index

- [decision-base-only-hook-runtime](decision-base-only-hook-runtime.md) — legacy in-repo hooks removed; base openbox_core is the only hook runtime; pathlib/io.open coverage TODO belongs in base.
- [decision-toolnode-seam-activity-binding](decision-toolnode-seam-activity-binding.md) — bind activity at ToolNode seam by WRITING config["run_id"] (not reading — it's None there); zero-fallback plain ContextStore; verified strict-mode behavior with handle_tool_errors default.
- [decision-langchain-core-callback-owned-lifecycle](decision-langchain-core-callback-owned-lifecycle.md) — tool/LLM lifecycle producer-owned via LangChain-Core callback; ActivityBridge deduplicates; stream events fallback for graph/chain only; fixes Demo 04 span ordering bugs.
- [demo-04-langgraph-only-content-builder](demo-04-langgraph-only-content-builder.md) — clean single-agent baseline vs demo-03's nested topology; create_react_agent deprecation notice is deliberate; image tools bypass fs sandbox (inherited, fix both demos together).
