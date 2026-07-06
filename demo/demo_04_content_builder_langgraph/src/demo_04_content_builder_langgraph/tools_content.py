"""External content tools."""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.tools import tool

from .paths import OUTPUT_DIR


@tool
def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news"] = "general",
) -> dict:
    """Search the web for current information.

    Args:
        query: The search query (be specific and detailed).
        max_results: Number of results to return (default 5).
        topic: "general" for most queries, "news" for current events.

    Returns:
        Search results with titles, URLs, and content excerpts (or an error dict).
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {"error": "TAVILY_API_KEY not set — skipping real web search"}
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        return client.search(query, max_results=max_results, topic=topic)
    except Exception as e:
        return {"error": f"Search failed: {e}"}


def _generate_image(prompt: str, rel_path: str) -> str:
    """Generate a PNG via Gemini and save it under OUTPUT_DIR/rel_path."""
    if not os.environ.get("GOOGLE_API_KEY"):
        return "GOOGLE_API_KEY not set — skipping image generation"
    try:
        from google import genai

        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
        )
        for part in response.parts or []:
            if part.inline_data is not None:
                out = OUTPUT_DIR / rel_path
                out.parent.mkdir(parents=True, exist_ok=True)
                part.as_image().save(str(out))
                return f"Image saved to {rel_path}"
        return "No image generated"
    except Exception as e:
        return f"Error: {e}"


@tool
def generate_cover(prompt: str, slug: str) -> str:
    """Generate a cover image for a blog post → blogs/<slug>/hero.png.

    Args:
        prompt: Detailed description of the image to generate.
        slug: Blog post slug.
    """
    return _generate_image(prompt, f"blogs/{slug}/hero.png")


@tool
def generate_social_image(prompt: str, platform: str, slug: str) -> str:
    """Generate an image for a social media post → <platform>/<slug>/image.png.

    Args:
        prompt: Detailed description of the image to generate.
        platform: Either "linkedin" or "tweets".
        slug: Post slug.
    """
    return _generate_image(prompt, f"{platform}/{slug}/image.png")
