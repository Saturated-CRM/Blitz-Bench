"""Tests for the prompt scheduler."""

from saturated_blitz_bench.core.scheduler import Scheduler
from saturated_blitz_bench.workload.loader import Prompt, PromptPool


def _make_pool() -> PromptPool:
    """Create a small test prompt pool."""
    prompts = []
    for i in range(10):
        prompts.append(
            Prompt(
                {
                    "id": f"short_{i}",
                    "category": "short_chat",
                    "source": "test",
                    "messages": [{"role": "user", "content": f"Hello {i}"}],
                    "max_tokens": 1024,
                    "input_token_count": 50 + i * 10,
                }
            )
        )
    for i in range(5):
        prompts.append(
            Prompt(
                {
                    "id": f"tool_{i}",
                    "category": "tool_call",
                    "source": "test",
                    "messages": [{"role": "user", "content": f"Call tool {i}"}],
                    "tools": [{"type": "function", "function": {"name": "test"}}],
                    "max_tokens": 4096,
                    "input_token_count": 2000 + i * 500,
                }
            )
        )
    return PromptPool(prompts)


def test_scheduler_selects_from_pool():
    pool = _make_pool()
    scheduler = Scheduler(pool)
    for _ in range(50):
        prompt = scheduler.select()
        assert prompt.id.startswith("short_") or prompt.id.startswith("tool_")


def test_scheduler_respects_weights():
    pool = _make_pool()
    # Heavy weight on short_chat
    scheduler = Scheduler(pool, weights={"short_chat": 0.99, "tool_call": 0.01})
    counts = {"short_chat": 0, "tool_call": 0}
    for _ in range(1000):
        prompt = scheduler.select()
        counts[prompt.category] += 1
    # short_chat should dominate
    assert counts["short_chat"] > counts["tool_call"] * 5


def test_scheduler_handles_missing_weights():
    pool = _make_pool()
    # Only provide weight for one category
    scheduler = Scheduler(pool, weights={"short_chat": 1.0})
    prompt = scheduler.select()
    assert prompt.category == "short_chat"
