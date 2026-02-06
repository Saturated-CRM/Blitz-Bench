"""Prompt selection with weighted random dispatch by category."""

from __future__ import annotations

import random
from typing import Any

from saturated_blitz_bench.workload.loader import Prompt, PromptPool


class Scheduler:
    """Selects prompts from the pool using weighted random by category."""

    def __init__(
        self,
        pool: PromptPool,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.pool = pool

        # Build category list and weights from categories that have prompts
        available_categories = [
            cat for cat, prompts in pool.by_category.items() if prompts
        ]
        if not available_categories:
            raise ValueError("Prompt pool has no prompts")

        if weights:
            # Use provided weights, falling back to equal for missing
            raw = {cat: weights.get(cat, 0.0) for cat in available_categories}
        else:
            raw = {cat: 1.0 for cat in available_categories}

        # Normalize
        total = sum(raw.values())
        if total == 0:
            total = len(available_categories)
            raw = {cat: 1.0 for cat in available_categories}

        self._categories = list(raw.keys())
        self._weights = [raw[cat] / total for cat in self._categories]

    def select(self) -> Prompt:
        """Select a random prompt using weighted category distribution."""
        (category,) = random.choices(self._categories, weights=self._weights, k=1)
        prompts = self.pool.by_category[category]
        return random.choice(prompts)
