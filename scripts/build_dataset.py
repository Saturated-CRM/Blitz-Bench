#!/usr/bin/env python3
"""Standalone script to build the prompt pool.

This is a thin wrapper around the package module. You can also use:
    saturated-blitz-bench build-dataset
    python -m saturated_blitz_bench.dataset_builder
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

try:
    from saturated_blitz_bench.dataset_builder import build_dataset
except ImportError:
    # Allow running from project root even without pip install
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from saturated_blitz_bench.dataset_builder import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the saturated-blitz-bench prompt pool")
    parser.add_argument("--output", default="prompts/workload_pool.jsonl")
    parser.add_argument("--total-prompts", type=int, default=5000)
    parser.add_argument("--max-input-tokens", type=int, default=80000)
    parser.add_argument("--tokenizer", default="cl100k_base")
    parser.add_argument("--quick", action="store_true", help="Quick build (100 prompts)")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Force rebuild even if dataset exists")

    args = parser.parse_args()

    if args.quick:
        args.total_prompts = 100

    build_dataset(
        output_path=args.output,
        total_prompts=args.total_prompts,
        max_input_tokens=args.max_input_tokens,
        tokenizer=args.tokenizer,
        category_filter=args.category,
        force=args.force,
    )


if __name__ == "__main__":
    main()
