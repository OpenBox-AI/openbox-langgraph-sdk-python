"""Run the OpenBox-governed content-builder agent.

    uv run demo_04_content_builder_langgraph "Write a blog post about AI agents"
    uv run demo_04_content_builder_langgraph "Create a LinkedIn post about prompt engineering"
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from openbox_langgraph import (
    ApprovalExpiredError,
    ApprovalRejectedError,
    GovernanceBlockedError,
    GovernanceHaltError,
)

from .agent import build_governed_agent

console = Console()

_DEFAULT_REQUEST = "Write a blog post about how AI agents are transforming software development."
_GOVERNANCE_ERRORS = (
    GovernanceBlockedError,
    GovernanceHaltError,
    ApprovalRejectedError,
    ApprovalExpiredError,
)
_THREAD_ID = "content-builder-langgraph-demo"


def _text(content) -> str:
    """Flatten message content to plain text."""
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content)


def _print_message(msg) -> None:
    """Render one message from the final trace."""
    if isinstance(msg, HumanMessage):
        console.print(Panel(str(msg.content), title="You", border_style="blue"))
    elif isinstance(msg, AIMessage):
        text = _text(msg.content)
        if text.strip():
            console.print(Panel(Markdown(text), title="Agent", border_style="green"))
        for tc in msg.tool_calls or []:
            name, args = tc.get("name", "?"), tc.get("args", {})
            if name in ("generate_cover", "generate_social_image"):
                console.print("  [cyan]» generating image…[/]")
            elif name == "write_file":
                console.print(f"  [yellow]» writing:[/] {args.get('file_path', '')}")
            elif name == "load_skill":
                console.print(f"  [blue]» loading skill:[/] {args.get('name', '')}")
            elif name == "web_search":
                console.print(f"  [blue]» searching:[/] {str(args.get('query', ''))[:60]}")
    elif isinstance(msg, ToolMessage):
        if "error" in str(msg.content).lower():
            console.print(f"  [red]✗ {getattr(msg, 'name', 'tool')}: {msg.content}[/]")


async def _main() -> None:
    load_dotenv()
    request = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else _DEFAULT_REQUEST
    console.print(
        "\n[bold blue]Content Builder Agent[/] [dim](LangGraph-only, OpenBox-governed)[/]"
    )
    console.print(f"[dim]Request: {request}[/]\n")

    governed = build_governed_agent()
    try:
        result = await governed.ainvoke(
            {"messages": [("user", request)]},
            config={"configurable": {"thread_id": _THREAD_ID}},
        )
        messages = result.get("messages", []) if isinstance(result, dict) else []
        for msg in messages:
            _print_message(msg)
    except _GOVERNANCE_ERRORS as e:
        verdict = getattr(e, "verdict", "blocked")
        console.print(f"\n[bold red]GOVERNANCE {str(verdict).upper()}:[/] {e}")

    console.print("\n[bold green]✓ Done![/]")


def run() -> None:
    """Console-script entry point."""
    asyncio.run(_main())


if __name__ == "__main__":
    run()
