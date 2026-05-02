"""Prompt templates reserved for future VLM/LLM instruction rewriting."""

from __future__ import annotations

STRUCTURED_REWRITE_PROMPT = """Rewrite the navigation instruction using only the provided structured label.
Do not invent landmarks, change left/right relations, change avoid/follow roles, or change the turn direction.
"""


def build_rewrite_prompt(structured_label: dict) -> str:
    """Build a conservative rewrite prompt for future LLM-based generation."""
    return f"{STRUCTURED_REWRITE_PROMPT}\nStructured label:\n{structured_label}"

