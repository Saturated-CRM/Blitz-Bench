"""Load and validate the prompt pool JSON file."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from saturated_blitz_bench.workload.categories import Category

logger = logging.getLogger(__name__)


class Prompt:
    """A single prompt from the workload pool."""

    __slots__ = (
        "id",
        "category",
        "source",
        "messages",
        "tools",
        "expected_tool",
        "max_tokens",
        "input_token_count",
        "source_metadata",
    )

    def __init__(self, data: dict[str, Any]) -> None:
        self.id: str = data["id"]
        self.category: str = data["category"]
        self.source: str = data["source"]
        self.messages: list[dict[str, str]] = data["messages"]
        self.tools: list[dict] | None = data.get("tools")
        self.expected_tool: dict | None = data.get("expected_tool")
        self.max_tokens: int = data.get("max_tokens", 1024)
        self.input_token_count: int = data.get("input_token_count", 0)
        self.source_metadata: dict | None = data.get("source_metadata")


class PromptPool:
    """The full set of curated prompts, grouped by category."""

    def __init__(self, prompts: list[Prompt]) -> None:
        self.all_prompts = prompts
        self.by_category: dict[str, list[Prompt]] = {}
        for p in prompts:
            self.by_category.setdefault(p.category, []).append(p)

    @property
    def size(self) -> int:
        return len(self.all_prompts)

    def category_counts(self) -> dict[str, int]:
        return {cat: len(ps) for cat, ps in self.by_category.items()}


def load_prompt_pool(path: str | Path) -> PromptPool:
    """Load the workload pool (JSONL or JSON) and return a PromptPool."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt pool not found at {path}. "
            "Run `saturated-blitz-bench build-dataset` first."
        )

    raw: list[dict[str, Any]] = []
    with open(path) as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            # Legacy JSON array format
            raw = json.load(f)
        else:
            # JSONL format (one JSON object per line)
            for line in f:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))

    prompts: list[Prompt] = []
    skipped = 0
    for i, entry in enumerate(raw):
        try:
            prompts.append(Prompt(entry))
        except (KeyError, TypeError) as e:
            skipped += 1
            logger.warning("Skipping invalid prompt at index %d: %s", i, e)

    if not prompts:
        raise ValueError(
            f"No valid prompts loaded from {path} "
            f"({len(raw)} entries, {skipped} skipped due to errors)"
        )

    if skipped:
        logger.warning("Skipped %d invalid entries out of %d total", skipped, len(raw))

    pool = PromptPool(prompts)

    logger.info(
        "Loaded %d prompts from %s — categories: %s",
        pool.size,
        path,
        pool.category_counts(),
    )
    return pool
