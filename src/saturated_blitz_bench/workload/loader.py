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
    """Load the workload pool JSON and return a PromptPool."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt pool not found at {path}. "
            "Run `saturated-blitz-bench build-dataset` first."
        )

    with open(path) as f:
        raw: list[dict[str, Any]] = json.load(f)

    prompts = [Prompt(entry) for entry in raw]
    pool = PromptPool(prompts)

    logger.info(
        "Loaded %d prompts from %s — categories: %s",
        pool.size,
        path,
        pool.category_counts(),
    )
    return pool
