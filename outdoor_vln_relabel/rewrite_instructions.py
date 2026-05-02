"""CLI for constrained instruction rewriting."""

from __future__ import annotations

import argparse

from outdoor_vln_relabel.io_utils import load_jsonl, save_jsonl
from outdoor_vln_relabel.rewrite.llm_stub import generate_llm_prompt_file
from outdoor_vln_relabel.rewrite.rewrite_dataset import rewrite_records


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rewrite Outdoor-VLN instructions with hard constraints."
    )
    parser.add_argument("--input_jsonl", required=True, help="Input JSONL dataset")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL dataset")
    parser.add_argument(
        "--backend",
        choices=["rule_based", "llm_stub"],
        default="rule_based",
        help="Rewrite backend",
    )
    parser.add_argument("--num_variants", type=int, default=3, help="Variants per record")
    parser.add_argument(
        "--keep_original",
        action="store_true",
        help="Keep original records in output for rule_based backend",
    )
    parser.add_argument(
        "--prompt_output",
        default=None,
        help="Output prompt JSONL for llm_stub backend",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Optional maximum number of input records to process",
    )
    return parser.parse_args()


def main() -> None:
    """Run constrained rewrite or prompt export."""
    args = parse_args()
    records = load_jsonl(args.input_jsonl)
    if args.max_records is not None:
        records = records[: max(0, args.max_records)]

    if args.backend == "llm_stub":
        if not args.prompt_output:
            raise ValueError("--backend llm_stub requires --prompt_output")
        generate_llm_prompt_file(
            records, args.prompt_output, num_variants=args.num_variants
        )
        print(f"Loaded {len(records)} records")
        print(f"Wrote rewrite prompts to {args.prompt_output}")
        return

    rewritten = rewrite_records(
        records,
        backend=args.backend,
        num_variants=args.num_variants,
        keep_original=args.keep_original,
    )
    save_jsonl(rewritten, args.output_jsonl)
    print(f"Loaded {len(records)} records")
    print(f"Wrote {len(rewritten)} records to {args.output_jsonl}")


if __name__ == "__main__":
    main()

