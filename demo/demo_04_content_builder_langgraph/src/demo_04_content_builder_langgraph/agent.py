"""Build the content-builder agent and OpenBox-governed wrapper."""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from openbox_langgraph import create_openbox_graph_handler

from .prompt import build_system_prompt
from .tools_content import generate_cover, generate_social_image, web_search
from .tools_fs import load_skill, read_file, write_file

_MODEL = "gpt-4o-mini"

_TOOL_TYPES = {
    "web_search": "http",
    "generate_cover": "http",
    "generate_social_image": "http",
    "read_file": "builtin",
    "write_file": "builtin",
    "load_skill": "builtin",
}


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"Missing required environment variable: {name}. "
            "Copy .env.example to .env and fill it in."
        )
    return value


def build_graph():
    """Construct the plain (ungoverned) content-builder agent."""
    llm = ChatOpenAI(model=_MODEL, temperature=0)
    tools = [
        load_skill,
        read_file,
        write_file,
        web_search,
        generate_cover,
        generate_social_image,
    ]
    return create_react_agent(llm, tools, prompt=build_system_prompt())


def build_governed_agent():
    """Return a governed handler wired to live OpenBox Core + base SDK hooks."""
    api_url = _require_env("OPENBOX_URL")
    api_key = _require_env("OPENBOX_API_KEY")
    agent_name = _require_env("OPENBOX_AGENT_NAME")
    _require_env("OPENAI_API_KEY")

    graph = build_graph()
    return create_openbox_graph_handler(
        graph=graph,
        api_url=api_url,
        api_key=api_key,
        agent_name=agent_name,
        use_core_instrumentation=True,
        send_chain_start_event=True,
        send_chain_end_event=True,
        send_tool_start_event=True,
        send_tool_end_event=True,
        send_llm_start_event=True,
        send_llm_end_event=True,
        tool_type_map=_TOOL_TYPES,
        hitl={"poll_interval_ms": 200},
    )
