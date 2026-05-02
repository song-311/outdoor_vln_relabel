"""LLM stub backend for prompt generation and response parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from outdoor_vln_relabel.io_utils import ensure_dir

from .prompts import build_rewrite_prompt


def generate_llm_prompt_file(
    records: List[dict], output_path: str, num_variants: int = 3
) -> None:
    """Write one constrained rewrite prompt per record to a JSONL file."""
    path = Path(output_path)
    if path.parent and str(path.parent) != ".":
        ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        for index, record in enumerate(records):
            payload = {
                "record_id": index,
                "segment_id": record.get("segment_id"),
                "original_instruction": record.get("instruction", ""),
                "prompt": build_rewrite_prompt(record, num_variants=num_variants),
            }
            file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def parse_llm_response_stub(response_text: str) -> List[str]:
    """Parse a JSON response that contains a rewrites list."""
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        return []
    rewrites = data.get("rewrites") if isinstance(data, dict) else None
    if not isinstance(rewrites, list):
        return []
    return [str(item) for item in rewrites if str(item).strip()]

