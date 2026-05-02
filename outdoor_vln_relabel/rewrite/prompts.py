"""Prompt construction for future constrained LLM rewrites."""

from __future__ import annotations

from .constraints import extract_constraints


def build_rewrite_prompt(record: dict, num_variants: int = 3) -> str:
    """Build a hard-constrained rewrite prompt for one record."""
    constraints = extract_constraints(record)
    landmark_lines = []
    for index, landmark in enumerate(constraints["landmarks"], start=1):
        landmark_lines.append(
            f"  {index}. {landmark['name']}, "
            f"role={landmark['role']}, relation={landmark['relation']}"
        )
    if not landmark_lines:
        landmark_lines.append("  none")

    forbidden = ", ".join(constraints["forbidden_follow_landmarks"])
    return "\n".join(
        [
            "You are rewriting navigation instructions for an outdoor ground robot.",
            "",
            "Original instruction:",
            f"\"{record.get('instruction', '')}\"",
            "",
            "Structured information:",
            f"- Terrain: {constraints.get('terrain')}",
            f"- Motion: {constraints.get('motion')}",
            "- Landmarks:",
            *landmark_lines,
            "",
            f"Rewrite the instruction into {num_variants} natural alternatives.",
            "",
            "Hard rules:",
            "- Do not invent new landmarks.",
            "- Do not remove any avoid obstacle constraint.",
            "- Do not change ahead/left/right/front-left/front-right relations.",
            "- Do not change the motion direction.",
            f"- Do not say to follow any of: {forbidden}.",
            "- Preserve landmark roles such as avoid, follow, go_toward, pass_between, and turn_at.",
            "- Keep the instruction executable by a ground robot.",
            "- Keep it concise.",
            "",
            "Output JSON:",
            "{",
            '  "rewrites": [',
            '    "...",',
            '    "...",',
            '    "..."',
            "  ]",
            "}",
        ]
    )

