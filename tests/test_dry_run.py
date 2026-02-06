"""Comprehensive dry-run tests verifying concurrency, timing, and correctness.

These tests run the full BenchmarkRunner with DryRunClient to validate that:
- Concurrency is maintained at the configured level (saturated)
- Requests are not over-dispatched (never exceed max_concurrency)
- Duration is respected
- Drain phase works properly
- Category distribution matches configured weights
- Errors and timeouts are handled gracefully
- All metrics are collected properly
- Throughput/RPM/latency calculations are mathematically correct
"""

import asyncio
import time
from collections import Counter
from unittest.mock import patch

import pytest

from saturated_blitz_bench.config import BenchmarkConfig
from saturated_blitz_bench.core.dry_run_client import DryRunClient
from saturated_blitz_bench.core.runner import BenchmarkRunner
from saturated_blitz_bench.metrics.calculator import compute_aggregate_metrics
from saturated_blitz_bench.workload.loader import Prompt, PromptPool


def _make_multi_category_pool(per_category: int = 50) -> PromptPool:
    """Create a pool with multiple categories for distribution testing."""
    prompts = []
    categories = [
        ("short_chat", 1024, 100),
        ("medium_chat", 2048, 800),
        ("tool_call", 4096, 3000),
        ("code_generation", 4096, 2000),
        ("long_context", 2048, 20000),
        ("multi_turn", 2048, 4000),
        ("reasoning", 4096, 1000),
    ]
    for cat, max_tok, input_tok in categories:
        for i in range(per_category):
            data = {
                "id": f"{cat}_{i}",
                "category": cat,
                "source": "test",
                "messages": [{"role": "user", "content": f"Test prompt {i} for {cat}"}],
                "max_tokens": max_tok,
                "input_token_count": input_tok,
            }
            if cat == "tool_call":
                data["tools"] = [
                    {"type": "function", "function": {"name": "test_fn", "parameters": {}}}
                ]
            prompts.append(Prompt(data))
    return PromptPool(prompts)


def _make_simple_pool(n: int = 20) -> PromptPool:
    """Create a simple single-category pool."""
    prompts = [
        Prompt(
            {
                "id": f"test_{i}",
                "category": "short_chat",
                "source": "test",
                "messages": [{"role": "user", "content": f"Hello {i}"}],
                "max_tokens": 1024,
                "input_token_count": 150,
            }
        )
        for i in range(n)
    ]
    return PromptPool(prompts)


def _fast_client(**overrides) -> DryRunClient:
    """Create a fast DryRunClient for tests that need short durations."""
    defaults = dict(
        ttft_range=(0.02, 0.05),
        tokens_range=(5, 15),
        itl_range=(0.003, 0.008),
        error_rate=0.0,
        timeout_rate=0.0,
    )
    defaults.update(overrides)
    return DryRunClient(**defaults)


def _make_dry_run_config(
    concurrency: int = 8,
    duration: int = 5,
    warmup: int = 0,
    timeout: int = 30,
) -> BenchmarkConfig:
    config = BenchmarkConfig()
    config.test.max_concurrency = concurrency
    config.test.duration_seconds = duration
    config.test.warmup_seconds = warmup
    config.test.request_timeout = timeout
    return config


@pytest.mark.asyncio
async def test_dry_run_maintains_concurrency():
    """Effective concurrency should stay near max_concurrency during steady state."""
    config = _make_dry_run_config(concurrency=16, duration=5)
    pool = _make_simple_pool(50)

    runner = BenchmarkRunner(config, pool, dry_run=True)
    runner.client = _fast_client()
    records = await runner.run()

    assert len(records) > 0
    success = [r for r in records if r.status == "success"]
    assert len(success) > 10

    eff = runner.concurrency_tracker.effective_concurrency()
    assert eff > config.test.max_concurrency * 0.4, (
        f"Effective concurrency {eff:.1f} too low "
        f"(expected >{config.test.max_concurrency * 0.4})"
    )


@pytest.mark.asyncio
async def test_dry_run_does_not_over_send():
    """Concurrency should never exceed max_concurrency (semaphore enforcement)."""
    config = _make_dry_run_config(concurrency=8, duration=4)
    pool = _make_simple_pool(30)

    runner = BenchmarkRunner(config, pool, dry_run=True)
    runner.client = _fast_client()

    peak_seen = 0
    original_increment = runner.concurrency_tracker.increment

    def tracking_increment():
        original_increment()
        nonlocal peak_seen
        current = runner.concurrency_tracker.current
        peak_seen = max(peak_seen, current)

    runner.concurrency_tracker.increment = tracking_increment

    records = await runner.run()

    assert len(records) > 0
    assert peak_seen <= config.test.max_concurrency, (
        f"Peak concurrency {peak_seen} exceeded max {config.test.max_concurrency}"
    )


@pytest.mark.asyncio
async def test_dry_run_respects_duration():
    """No requests should be dispatched after the configured duration."""
    duration = 3
    config = _make_dry_run_config(concurrency=4, duration=duration, timeout=5)
    pool = _make_simple_pool(20)

    runner = BenchmarkRunner(config, pool, dry_run=True)
    runner.client = _fast_client()
    start = time.time()
    records = await runner.run()
    elapsed = time.time() - start

    assert elapsed < duration + 10, f"Took {elapsed:.1f}s, expected ~{duration}s"
    assert elapsed >= duration - 0.5, f"Finished too early: {elapsed:.1f}s"

    for r in records:
        dispatch_offset = r.request_sent_at - runner._start_time
        assert dispatch_offset < duration + 0.5, (
            f"Request dispatched at +{dispatch_offset:.1f}s, "
            f"after duration {duration}s"
        )


@pytest.mark.asyncio
async def test_dry_run_drain_completes():
    """All in-flight requests should complete or be cancelled during drain."""
    config = _make_dry_run_config(concurrency=8, duration=3, timeout=10)
    pool = _make_simple_pool(30)

    runner = BenchmarkRunner(config, pool, dry_run=True)
    runner.client = _fast_client()
    records = await runner.run()

    for r in records:
        assert r.status in ("success", "error", "timeout"), (
            f"Record {r.prompt_id} still has status={r.status}"
        )
        assert r.completed_at is not None, (
            f"Record {r.prompt_id} has no completed_at"
        )

    assert runner.concurrency_tracker.current == 0


@pytest.mark.asyncio
async def test_dry_run_category_distribution():
    """Request distribution should roughly match configured weights."""
    config = _make_dry_run_config(concurrency=16, duration=6)
    pool = _make_multi_category_pool(per_category=100)

    runner = BenchmarkRunner(config, pool, dry_run=True)
    runner.client = _fast_client()
    records = await runner.run()

    total = len(records)
    assert total > 50, f"Too few requests: {total}"

    counts = Counter(r.category for r in records)
    weights = config.workload.distribution.model_dump()

    # short_chat (25%) should have more than reasoning (5%)
    if "short_chat" in counts and "reasoning" in counts:
        assert counts["short_chat"] > counts["reasoning"], (
            f"short_chat ({counts['short_chat']}) should be > "
            f"reasoning ({counts['reasoning']})"
        )

    for cat, weight in weights.items():
        if weight > 0 and cat in pool.by_category:
            assert counts.get(cat, 0) > 0, (
                f"Category {cat} (weight={weight}) got 0 requests"
            )


@pytest.mark.asyncio
async def test_dry_run_handles_errors_gracefully():
    """Benchmark should continue normally even with elevated error rate."""
    config = _make_dry_run_config(concurrency=8, duration=4)
    pool = _make_simple_pool(30)

    runner = BenchmarkRunner(config, pool, dry_run=True)
    runner.client = _fast_client(error_rate=0.15)

    records = await runner.run()

    total = len(records)
    assert total > 20, f"Too few requests: {total}"

    errors = [r for r in records if r.status == "error"]
    success = [r for r in records if r.status == "success"]

    assert len(errors) > 0, "Expected some errors with 15% error rate"
    assert len(success) > 0, "Expected some successes"

    error_rate = len(errors) / total
    assert 0.03 < error_rate < 0.40, f"Error rate {error_rate:.1%} out of expected range"


@pytest.mark.asyncio
async def test_dry_run_handles_timeouts():
    """Timeout records should be properly recorded."""
    config = _make_dry_run_config(concurrency=8, duration=4)
    pool = _make_simple_pool(30)

    runner = BenchmarkRunner(config, pool, dry_run=True)
    runner.client = _fast_client(timeout_rate=0.15)

    records = await runner.run()

    timeouts = [r for r in records if r.status == "timeout"]
    success = [r for r in records if r.status == "success"]

    assert len(timeouts) > 0, "Expected some timeouts with 15% timeout rate"
    assert len(success) > 0, "Expected some successes"

    for t in timeouts:
        assert t.error, "Timeout record should have error message"
        assert t.completed_at is not None


@pytest.mark.asyncio
async def test_dry_run_metrics_collected():
    """All success records should have valid timing and token metrics."""
    config = _make_dry_run_config(concurrency=8, duration=4)
    pool = _make_simple_pool(30)

    runner = BenchmarkRunner(config, pool, dry_run=True)
    runner.client = _fast_client()

    records = await runner.run()

    success = [r for r in records if r.status == "success"]
    assert len(success) > 10

    for r in success:
        assert r.request_sent_at > 0
        assert r.first_token_at is not None
        assert r.first_token_at > r.request_sent_at
        assert r.completed_at is not None
        assert r.completed_at >= r.first_token_at

        assert r.ttft is not None and r.ttft > 0
        assert r.e2e_latency is not None and r.e2e_latency > 0
        assert r.e2e_latency >= r.ttft

        assert r.output_tokens > 0
        assert r.output_tps is not None and r.output_tps > 0

        if r.output_tokens > 1:
            assert len(r.itl_values) >= 1


@pytest.mark.asyncio
async def test_dry_run_expected_request_count():
    """Request count should be reasonable given concurrency and latency."""
    concurrency = 8
    duration = 5
    config = _make_dry_run_config(concurrency=concurrency, duration=duration)
    pool = _make_simple_pool(30)

    runner = BenchmarkRunner(config, pool, dry_run=True)
    runner.client = _fast_client()

    records = await runner.run()

    assert len(records) > 50, f"Too few requests: {len(records)}"
    assert len(records) < 5000, f"Too many requests: {len(records)}"


@pytest.mark.asyncio
async def test_dry_run_no_network_calls():
    """Dry run should never make actual HTTP requests."""
    config = _make_dry_run_config(concurrency=4, duration=2)
    pool = _make_simple_pool(10)

    runner = BenchmarkRunner(config, pool, dry_run=True)
    runner.client = _fast_client()

    import httpx

    original_stream = httpx.AsyncClient.stream
    call_count = 0

    def tracking_stream(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_stream(*args, **kwargs)

    with patch.object(httpx.AsyncClient, "stream", side_effect=tracking_stream):
        records = await runner.run()

    assert len(records) > 0
    assert call_count == 0, f"Made {call_count} real HTTP calls during dry run"


@pytest.mark.asyncio
async def test_dry_run_client_basic():
    """DryRunClient produces valid SSE-like chunks."""
    client = DryRunClient(
        ttft_range=(0.01, 0.02),
        tokens_range=(5, 10),
        itl_range=(0.001, 0.005),
        error_rate=0.0,
        timeout_rate=0.0,
    )

    chunks = []
    async for chunk in client.stream_chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        model="test",
        max_tokens=100,
    ):
        chunks.append(chunk)

    assert len(chunks) >= 2  # at least 1 content + 1 final
    assert "usage" in chunks[-1]
    assert chunks[-1]["usage"]["completion_tokens"] > 0
    # Should NOT include prompt_tokens (runner keeps pool's accurate count)
    assert "prompt_tokens" not in chunks[-1]["usage"]
    for c in chunks[:-1]:
        assert c["choices"][0]["delta"].get("content")

    await client.close()


@pytest.mark.asyncio
async def test_dry_run_client_does_not_override_input_tokens():
    """DryRunClient should not provide prompt_tokens so pool values are preserved."""
    config = _make_dry_run_config(concurrency=1, duration=2)
    pool_input_tokens = 12345
    pool = PromptPool([
        Prompt({
            "id": "tok_test",
            "category": "short_chat",
            "source": "test",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 64,
            "input_token_count": pool_input_tokens,
        })
    ])

    runner = BenchmarkRunner(config, pool, dry_run=True)
    runner.client = _fast_client()
    records = await runner.run()

    success = [r for r in records if r.status == "success"]
    assert len(success) > 0
    # input_tokens should be the pool's value, NOT overridden by DryRunClient
    for r in success:
        assert r.input_tokens == pool_input_tokens, (
            f"input_tokens={r.input_tokens} but expected {pool_input_tokens}"
        )


@pytest.mark.asyncio
async def test_dry_run_metrics_math_consistency():
    """Verify throughput, RPM, and per-request metrics are mathematically consistent."""
    config = _make_dry_run_config(concurrency=8, duration=5, warmup=0)
    pool = _make_simple_pool(30)

    runner = BenchmarkRunner(config, pool, dry_run=True)
    runner.client = _fast_client()

    start_time = time.time()
    records = await runner.run()
    total_duration = time.time() - start_time

    metrics = compute_aggregate_metrics(
        records=records,
        total_duration=total_duration,
        warmup_seconds=0,
        benchmark_start_time=start_time,
    )

    # --- Core identity checks ---
    total_requests = metrics["total_requests"]
    total_attempted = metrics["total_attempted"]
    failed = metrics["failed_requests"]
    eff_dur = metrics["effective_duration_seconds"]

    assert total_attempted == total_requests + failed
    assert metrics["error_rate"] == (
        failed / total_attempted if total_attempted > 0 else 0
    )

    # --- RPM check ---
    expected_rpm = total_requests / eff_dur * 60
    assert abs(metrics["rpm"] - expected_rpm) < 0.01

    # --- Throughput = total_output_tokens / effective_duration ---
    total_output = metrics["total_output_tokens"]
    expected_throughput = total_output / eff_dur
    assert abs(metrics["system_throughput_tps"] - expected_throughput) < 0.01

    # --- Per-request: avg_output_tokens * RPM/60 ~ throughput ---
    success = [r for r in records if r.status == "success"]
    if success:
        avg_out_per_req = total_output / total_requests
        req_per_sec = total_requests / eff_dur
        reconstructed_throughput = avg_out_per_req * req_per_sec
        assert abs(metrics["system_throughput_tps"] - reconstructed_throughput) < 0.5

    # --- Input throughput check ---
    total_input = metrics["total_input_tokens"]
    expected_input_tps = total_input / eff_dur
    assert abs(metrics["input_throughput_tps"] - expected_input_tps) < 0.01

    # --- TTFT: should equal avg of per-request TTFTs ---
    ttft_values = [r.ttft for r in success if r.ttft is not None]
    if ttft_values:
        expected_ttft_avg = sum(ttft_values) / len(ttft_values)
        assert abs(metrics["ttft"]["avg"] - expected_ttft_avg) < 0.001

    # --- Per-category counts should sum to total ---
    cat_total = sum(
        info["count"] for info in metrics["per_category"].values()
    )
    assert cat_total == total_requests
