# Demo 04 - Content builder on one LangGraph agent, governed by OpenBox

This demo runs a **single LangGraph prebuilt ReAct agent**
(`langgraph.prebuilt.create_react_agent`) with direct leaf tools, governed by
`create_openbox_graph_handler`.

## What the agent does

1. Reads `AGENTS.md` (brand voice) and lazily loads skill instructions via `load_skill`.
2. Researches directly with `web_search` (Tavily) and saves findings to
   `research/<slug>.md`.
3. Reads/writes artifacts under `output/` with sandboxed `read_file` / `write_file`.
4. Generates images with `generate_cover` / `generate_social_image` (Gemini) when
   the selected content type requires one.
5. Returns a concise final answer describing created files.

`web_search` and the image tools degrade gracefully (they return an error
message instead of raising) when `TAVILY_API_KEY` / `GOOGLE_API_KEY` are absent.

## How governance is wired

`build_governed_agent()` wraps the compiled graph with
`create_openbox_graph_handler`.

- `use_core_instrumentation=True`
- chain, tool, and LLM start/end events enabled
- `tool_type_map` for `http` / `builtin` tool classification
- `hitl={"poll_interval_ms": 200}`

## Expected OpenBox log shape

- workflow lifecycle rows for the run
- `llm_call` rows for each model turn
- direct tool rows named by their real tool names: `load_skill`, `web_search`,
  `read_file`, `write_file`, `generate_cover`, `generate_social_image`
- hook spans created inside direct tools attach to that tool's activity where
  base instrumentation can observe them
- no composite `task` rows

## Run

```bash
cd demo/demo_04_content_builder_langgraph
cp .env.example .env
uv sync
uv run demo_04_content_builder_langgraph
uv run demo_04_content_builder_langgraph "Write a LinkedIn post about prompt engineering"
```

Requires a live OpenBox Core instance + API key (`obx_live_*` / `obx_test_*`) with
a dashboard **agent** named to match `OPENBOX_AGENT_NAME` (default
`ContentBuilderAgent`), plus an OpenAI key. Generated content is written under
`output/` (gitignored).

## Layout

```
demo_04_content_builder_langgraph/
├── AGENTS.md
├── skills/{blog-post,social-media}/SKILL.md
└── src/demo_04_content_builder_langgraph/
    ├── paths.py
    ├── prompt.py
    ├── tools_fs.py
    ├── tools_content.py
    ├── agent.py
    └── main.py
```
