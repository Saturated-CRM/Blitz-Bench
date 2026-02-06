"""Aggregate metrics computation: percentiles, averages, throughput."""

from __future__ import annotations

import math
from typing import Any

from saturated_blitz_bench.metrics.collector import RequestRecord


def percentile(values: list[float], p: float) -> float | None:
    """Compute the p-th percentile of a sorted list of values."""
    if not values:
        return None
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def compute_distribution(values: list[float]) -> dict[str, float | None]:
    """Compute avg, p50, p90, p95, p99 for a list of values."""
    if not values:
        return {"avg": None, "p50": None, "p90": None, "p95": None, "p99": None}
    return {
        "avg": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def compute_aggregate_metrics(
    records: list[RequestRecord],
    total_duration: float,
    warmup_seconds: float,
    gpu_count: int | None = None,
    benchmark_start_time: float = 0.0,
) -> dict[str, Any]:
    """Compute all aggregate metrics from a list of RequestRecords.

    Records within the warmup period are excluded from aggregate stats.
    """
    # Filter out warmup period and non-success
    warmup_cutoff = benchmark_start_time + warmup_seconds
    steady = [
        r
        for r in records
        if r.status == "success" and r.request_sent_at >= warmup_cutoff
    ]
    all_attempted = [r for r in records if r.request_sent_at >= warmup_cutoff]
    effective_duration = total_duration - warmup_seconds

    if not steady or effective_duration <= 0:
        return _empty_metrics()

    # Token totals
    total_input_tokens = sum(r.input_tokens for r in steady)
    total_output_tokens = sum(r.output_tokens for r in steady)
    total_requests = len(steady)
    total_attempted = len(all_attempted)
    failed = total_attempted - total_requests

    # Throughput
    system_throughput = total_output_tokens / effective_duration
    input_throughput = total_input_tokens / effective_duration
    combined_throughput = (total_input_tokens + total_output_tokens) / effective_duration
    throughput_per_gpu = system_throughput / gpu_count if gpu_count else None

    # RPM
    rpm = total_requests / effective_duration * 60

    # Latency distributions
    ttft_values = [r.ttft for r in steady if r.ttft is not None]
    e2e_values = [r.e2e_latency for r in steady if r.e2e_latency is not None]
    tps_values = [r.output_tps for r in steady if r.output_tps is not None]

    # ITL: flatten all ITL values
    all_itl: list[float] = []
    for r in steady:
        all_itl.extend(r.itl_values)

    # Tool call accuracy
    tool_call_prompts = [r for r in steady if r.tool_call_correct is not None]
    tool_call_correct = sum(1 for r in tool_call_prompts if r.tool_call_correct)
    tool_call_accuracy = (
        tool_call_correct / len(tool_call_prompts) if tool_call_prompts else None
    )

    # Per-category breakdowns
    categories = sorted(set(r.category for r in steady))
    per_category: dict[str, dict[str, Any]] = {}
    for cat in categories:
        cat_records = [r for r in steady if r.category == cat]
        cat_ttft = [r.ttft for r in cat_records if r.ttft is not None]
        cat_e2e = [r.e2e_latency for r in cat_records if r.e2e_latency is not None]
        cat_tps = [r.output_tps for r in cat_records if r.output_tps is not None]
        per_category[cat] = {
            "count": len(cat_records),
            "ttft": compute_distribution(cat_ttft),
            "e2e_latency": compute_distribution(cat_e2e),
            "output_tps": compute_distribution(cat_tps),
            "avg_input_tokens": (
                sum(r.input_tokens for r in cat_records) / len(cat_records)
            ),
            "avg_output_tokens": (
                sum(r.output_tokens for r in cat_records) / len(cat_records)
            ),
        }

    return {
        "total_requests": total_requests,
        "total_attempted": total_attempted,
        "failed_requests": failed,
        "error_rate": failed / total_attempted if total_attempted else 0,
        "rpm": rpm,
        "rpm_10": rpm * 10,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "system_throughput_tps": system_throughput,
        "input_throughput_tps": input_throughput,
        "combined_throughput_tps": combined_throughput,
        "throughput_per_gpu": throughput_per_gpu,
        "effective_duration_seconds": effective_duration,
        "ttft": compute_distribution(ttft_values),
        "e2e_latency": compute_distribution(e2e_values),
        "output_tps": compute_distribution(tps_values),
        "itl": compute_distribution(all_itl),
        "tool_call_accuracy": tool_call_accuracy,
        "tool_call_total": len(tool_call_prompts),
        "tool_call_correct": tool_call_correct,
        "per_category": per_category,
    }


def _empty_metrics() -> dict[str, Any]:
    empty_dist = {"avg": None, "p50": None, "p90": None, "p95": None, "p99": None}
    return {
        "total_requests": 0,
        "total_attempted": 0,
        "failed_requests": 0,
        "error_rate": 0,
        "rpm": 0,
        "rpm_10": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "system_throughput_tps": 0,
        "input_throughput_tps": 0,
        "combined_throughput_tps": 0,
        "throughput_per_gpu": None,
        "effective_duration_seconds": 0,
        "ttft": empty_dist,
        "e2e_latency": empty_dist,
        "output_tps": empty_dist,
        "itl": empty_dist,
        "tool_call_accuracy": None,
        "tool_call_total": 0,
        "tool_call_correct": 0,
        "per_category": {},
    }
