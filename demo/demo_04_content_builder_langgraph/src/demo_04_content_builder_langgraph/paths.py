"""Filesystem anchors for the demo."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
SKILLS_DIR = BASE_DIR / "skills"
OUTPUT_DIR = BASE_DIR / "output"
AGENTS_MD = BASE_DIR / "AGENTS.md"
