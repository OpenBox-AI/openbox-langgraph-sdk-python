"""Assemble the content-builder system prompt from on-disk configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

from .paths import AGENTS_MD, SKILLS_DIR


def _scan_scalars(block: str) -> dict:
    """Line-based fallback for frontmatter that is not strict YAML."""
    result: dict[str, str] = {}
    for line in block.splitlines():
        if not line or line[0] in " \t" or ":" not in line:
            continue
        key, _, value = line.partition(":")
        result[key.strip()] = value.strip().strip("\"'")
    return result


def _parse_frontmatter(text: str) -> dict:
    """Extract the leading YAML frontmatter block delimited by '---' fences."""
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    block = parts[1]
    try:
        data = yaml.safe_load(block)
        if isinstance(data, dict) and data:
            return data
    except yaml.YAMLError:
        pass
    return _scan_scalars(block)


def _skill_catalog(skills_dir: Path) -> str:
    """Render each skill as a catalog bullet."""
    lines: list[str] = []
    for skill_md in sorted(skills_dir.rglob("SKILL.md")):
        meta = _parse_frontmatter(skill_md.read_text())
        name = meta.get("name") or skill_md.parent.name
        desc = str(meta.get("description") or "").strip()
        lines.append(f"- **{name}** — {desc}")
    return "\n".join(lines) if lines else "(no skills found)"


def build_system_prompt() -> str:
    """Return the full system prompt."""
    memory = AGENTS_MD.read_text() if AGENTS_MD.exists() else ""
    catalog = _skill_catalog(SKILLS_DIR)
    return f"""{memory}

## Available Skills (load on demand)

The catalog below lists only skill names and descriptions. Before using a skill,
call `load_skill(<name>)` to read its full step-by-step instructions.

{catalog}

## Tools & Workflow

1. Pick the skill that matches the request; call `load_skill(<name>)` to read it.
2. Research FIRST by calling `web_search` directly (specific queries; run several
   if the topic is broad).
3. Save your findings to `research/<slug>.md` with `write_file`.
4. Call `read_file(path)` to re-read the saved research before drafting.
5. Call `write_file(path, content)` to save content (e.g. `blogs/<slug>/post.md`).
6. Generate images with `generate_cover(prompt, slug)` for a blog hero, or
   `generate_social_image(prompt, platform, slug)` for a social visual.

Choose and call one tool at a time. All file paths are relative to the output
directory.
"""
