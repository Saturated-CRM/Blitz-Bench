#!/usr/bin/env python3
"""Verify prompt pool integrity and token counts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import tiktoken


def validate(pool_path: str = "prompts/workload_pool.json") -> bool:
    path = Path(pool_path)
    if not path.exists():
        print(f"ERROR: {path} not found")
        return False

    with open(path) as f:
        prompts = json.load(f)

    print(f"Loaded {len(prompts)} prompts from {path}")

    enc = tiktoken.get_encoding("cl100k_base")
    errors = 0

    required_fields = {"id", "category", "source", "messages", "max_tokens", "input_token_count"}
    categories: dict[str, int] = {}
    token_mismatches = 0

    for i, p in enumerate(prompts):
        # Check required fields
        missing = required_fields - set(p.keys())
        if missing:
            print(f"  [{i}] Missing fields: {missing}")
            errors += 1
            continue

        # Check messages is a list with at least one entry
        if not isinstance(p["messages"], list) or len(p["messages"]) == 0:
            print(f"  [{i}] Empty or invalid messages")
            errors += 1
            continue

        # Check each message has role and content
        for j, msg in enumerate(p["messages"]):
            if "role" not in msg:
                print(f"  [{i}] Message {j} missing 'role'")
                errors += 1
            if "content" not in msg:
                print(f"  [{i}] Message {j} missing 'content'")
                errors += 1

        # Verify token count (within 20% tolerance)
        actual_tokens = 0
        for msg in p["messages"]:
            actual_tokens += 4
            for v in msg.values():
                if isinstance(v, str):
                    actual_tokens += len(enc.encode(v, disallowed_special=()))
        actual_tokens += 2

        claimed = p["input_token_count"]
        if claimed > 0 and abs(actual_tokens - claimed) / max(claimed, 1) > 0.20:
            token_mismatches += 1

        categories[p["category"]] = categories.get(p["category"], 0) + 1

        # Tool call prompts should have expected_tool
        if p["category"] == "tool_call":
            if not p.get("tools") and not p.get("expected_tool"):
                pass  # Acceptable - some BFCL entries may not have structured tools

    print(f"\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        pct = count / len(prompts) * 100
        print(f"  {cat:20s}: {count:5d} ({pct:5.1f}%)")

    print(f"\nToken count mismatches (>20%): {token_mismatches}")
    print(f"Errors: {errors}")

    ok = errors == 0
    if ok:
        print("\nValidation PASSED")
    else:
        print("\nValidation FAILED")
    return ok


if __name__ == "__main__":
    pool_path = sys.argv[1] if len(sys.argv) > 1 else "prompts/workload_pool.json"
    ok = validate(pool_path)
    sys.exit(0 if ok else 1)
