"""OpenBox LangGraph SDK — Test Agent.

A minimal multi-turn LangGraph agent that exercises OpenBox governance:

- Guardrails: prompt screening via agent_validatePrompt
- Policies: tool-level allow/block/require_approval
- HITL: approval polling for sensitive tools (when configured in dashboard)
- Behavior Rules: outbound HTTP spans via httpx instrumentation

Run:
  uv run python agent.py

Environment:
  OPENBOX_URL, OPENBOX_API_KEY, OPENAI_API_KEY
  OPENBOX_DEBUG=1  (prints governance requests/responses and raw LangGraph events)
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from openbox_langgraph import create_openbox_graph_handler

load_dotenv()


# ─── In-memory storage ─────────────────────────────────────────────

_report_store: dict[str, str] = {}


# ─── Tools ─────────────────────────────────────────────────────────

@tool
async def search_web(query: str) -> str:
    """Search Wikipedia via HTTP GET."""
    import httpx  # lazy

    print(f"  [search_web] query: {query}")
    url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query&list=search&format=json&srsearch="
        + httpx.QueryParams({"q": query}).get("q", query)
    )
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

    hits = (data.get("query") or {}).get("search") or []
    if not hits:
        return "No results found."

    lines: list[str] = []
    for h in hits[:5]:
        title = h.get("title")
        snippet = (h.get("snippet") or "").replace("<span class=\"searchmatch\">", "").replace(
            "</span>", ""
        )
        lines.append(f"- {title}: {snippet}")

    return "\n".join(lines)


@tool
def write_report(title: str, content: str, classification: str = "public") -> str:
    """Save a report into an in-memory store."""
    report_id = f"RPT-{datetime.now(timezone.utc).strftime('%H%M%S%f')[:12]}"
    _report_store[report_id] = content
    print(f"  [write_report] created {report_id}: {title} [{classification}]")
    return "\n".join(
        [
            "Report saved successfully.",
            f"  Report ID      : {report_id}",
            f"  Title          : {title}",
            f"  Classification : {classification}",
            f"  Length         : {len(content)} characters",
            f"  Created        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
    )


# ─── REPL ──────────────────────────────────────────────────────────

async def _repl(governed: object, *, thread_id: str) -> None:
    print()
    print("══════════════════════════════════════════════════════════════")
    print("Session started. How can the test agent help you today?")
    print("══════════════════════════════════════════════════════════════")
    print()

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            return

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Session ended.")
            return

        try:
            result = await governed.ainvoke(
                {"messages": [{"role": "user", "content": user_text}]},
                config={"configurable": {"thread_id": thread_id}},
            )
            msgs = result.get("messages") or []
            last = msgs[-1].content if msgs else ""
            print(f"\nAgent: {last}\n")
        except Exception as e:  # noqa: BLE001
            print(str(e))


async def main() -> None:
    openbox_url = os.environ.get("OPENBOX_URL", "").strip()
    openbox_api_key = os.environ.get("OPENBOX_API_KEY", "").strip()

    if not openbox_url or not openbox_api_key:
        print("Missing OPENBOX_URL or OPENBOX_API_KEY in environment.", file=sys.stderr)
        sys.exit(1)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║      LangGraph Test Agent — OpenBox Governance Demo      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"OpenBox : {openbox_url}")
    print(f"Key     : {openbox_api_key[:12]}...")
    print()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    graph = create_react_agent(llm, tools=[search_web, write_report])

    governed = create_openbox_graph_handler(
        graph=graph,
        api_url=openbox_url,
        api_key=openbox_api_key,
        agent_name="LangGraphTestAgent",
        validate=True,
        on_api_error="fail_open",
        tool_type_map={
            "search_web": "http",
        },
        skip_chain_types={
            "agent",
            "call_model",
            "RunnableSequence",
            "Prompt",
            "ChatPromptTemplate",
        },
        hitl={
            "enabled": True,
            "poll_interval_ms": 5_000,
            "max_wait_ms": 300_000,
        },
    )

    thread_id = f"langgraph-test-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    await _repl(governed, thread_id=thread_id)


if __name__ == "__main__":
    asyncio.run(main())
