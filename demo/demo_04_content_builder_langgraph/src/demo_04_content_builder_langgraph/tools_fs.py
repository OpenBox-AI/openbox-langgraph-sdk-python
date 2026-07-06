"""Filesystem tools."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from .paths import OUTPUT_DIR, SKILLS_DIR


def _safe_resolve(root: Path, rel: str) -> Path:
    """Resolve `rel` to a file strictly inside `root`."""
    root = root.resolve()
    target = (root / rel).resolve()
    if root not in target.parents:
        raise ValueError(f"path must name a file inside the sandbox: {rel!r}")
    return target


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file under the output directory (creates parent dirs).

    Args:
        file_path: Relative path, e.g. 'blogs/my-post/post.md'.
        content: The text content to write.
    """
    try:
        path = _safe_resolve(OUTPUT_DIR, file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"File written to {path.relative_to(OUTPUT_DIR.resolve())}"
    except Exception as e:
        return f"Error: {e}"


@tool
def read_file(file_path: str) -> str:
    """Read a file from under the output directory.

    Args:
        file_path: Relative path, e.g. 'research/topic.md'.
    """
    try:
        path = _safe_resolve(OUTPUT_DIR, file_path)
        return path.read_text()
    except Exception as e:
        return f"Error: {e}"


@tool
def load_skill(name: str) -> str:
    """Load the full instructions for a named skill (e.g. 'blog-post').

    Args:
        name: Skill directory name under skills/ (see the catalog in the prompt).
    """
    try:
        path = _safe_resolve(SKILLS_DIR, f"{name}/SKILL.md")
        return path.read_text()
    except Exception as e:
        return f"Error: skill {name!r} could not be loaded ({e})"
