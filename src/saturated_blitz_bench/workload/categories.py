"""Workload category definitions and distribution weights."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Category(str, Enum):
    SHORT_CHAT = "short_chat"
    MEDIUM_CHAT = "medium_chat"
    TOOL_CALL = "tool_call"
    CODE_GENERATION = "code_generation"
    LONG_CONTEXT = "long_context"
    MULTI_TURN = "multi_turn"
    REASONING = "reasoning"


@dataclass(frozen=True)
class CategorySpec:
    name: Category
    weight: float
    input_token_min: int
    input_token_max: int
    typical_output_min: int
    typical_output_max: int
    max_tokens: int
    description: str
    buckets: list[tuple[int, int]]


CATEGORY_SPECS: dict[Category, CategorySpec] = {
    Category.SHORT_CHAT: CategorySpec(
        name=Category.SHORT_CHAT,
        weight=0.25,
        input_token_min=50,
        input_token_max=500,
        typical_output_min=100,
        typical_output_max=1000,
        max_tokens=1024,
        description="Quick Q&A, casual conversation, simple instructions",
        buckets=[(50, 100), (100, 200), (200, 350), (350, 500)],
    ),
    Category.MEDIUM_CHAT: CategorySpec(
        name=Category.MEDIUM_CHAT,
        weight=0.15,
        input_token_min=500,
        input_token_max=4000,
        typical_output_min=200,
        typical_output_max=2000,
        max_tokens=2048,
        description="Detailed explanations, essays, content writing, analysis",
        buckets=[(500, 1000), (1000, 2000), (2000, 3000), (3000, 4000)],
    ),
    Category.TOOL_CALL: CategorySpec(
        name=Category.TOOL_CALL,
        weight=0.20,
        input_token_min=2000,
        input_token_max=16000,
        typical_output_min=200,
        typical_output_max=2000,
        max_tokens=4096,
        description="Tool definitions + multi-turn agentic context",
        buckets=[(2000, 5000), (5000, 8000), (8000, 12000), (12000, 16000)],
    ),
    Category.CODE_GENERATION: CategorySpec(
        name=Category.CODE_GENERATION,
        weight=0.10,
        input_token_min=1000,
        input_token_max=8000,
        typical_output_min=500,
        typical_output_max=4000,
        max_tokens=4096,
        description="Write functions, full file implementations, code review",
        buckets=[(1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000)],
    ),
    Category.LONG_CONTEXT: CategorySpec(
        name=Category.LONG_CONTEXT,
        weight=0.15,
        input_token_min=8000,
        input_token_max=80000,
        typical_output_min=200,
        typical_output_max=1000,
        max_tokens=2048,
        description="Roleplay chat histories, document summarization",
        buckets=[(8000, 16000), (16000, 32000), (32000, 50000), (50000, 80000)],
    ),
    Category.MULTI_TURN: CategorySpec(
        name=Category.MULTI_TURN,
        weight=0.10,
        input_token_min=2000,
        input_token_max=16000,
        typical_output_min=200,
        typical_output_max=1500,
        max_tokens=2048,
        description="Simulated conversation history (5-20 turns)",
        buckets=[(2000, 4000), (4000, 8000), (8000, 12000), (12000, 16000)],
    ),
    Category.REASONING: CategorySpec(
        name=Category.REASONING,
        weight=0.05,
        input_token_min=500,
        input_token_max=4000,
        typical_output_min=500,
        typical_output_max=4000,
        max_tokens=4096,
        description="Math, logic, step-by-step CoT, deep analysis",
        buckets=[(500, 1000), (1000, 2000), (2000, 3000), (3000, 4000)],
    ),
}


DEFAULT_WEIGHTS: dict[str, float] = {
    cat.value: spec.weight for cat, spec in CATEGORY_SPECS.items()
}
