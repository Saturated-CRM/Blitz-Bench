"""Time-bucketed metrics for charts (10-second windows)."""

from __future__ import annotations

from typing import Any

from saturated_blitz_bench.metrics.collector import RequestRecord
from saturated_blitz_bench.metrics.calculator import compute_distribution


def compute_timeseries(
    records: list[RequestRecord],
    benchmark_start: float,
    bucket_seconds: float = 10.0,
) -> list[dict[str, Any]]:
    """Bucket completed requests into time windows and compute per-bucket metrics.

    Returns a list of dicts, one per bucket, with:
        - time_offset: seconds from start
        - output_tokens: total output tokens in this bucket
        - throughput_tps: output tokens / bucket_seconds
        - request_count: requests completed in this bucket
        - ttft_p50, ttft_p95: TTFT percentiles for requests completed in this bucket
        - e2e_p50, e2e_p95: E2E latency percentiles
    """
    if not records:
        return []

    success = [r for r in records if r.status == "success" and r.completed_at]
    if not success:
        return []

    end_time = max(r.completed_at for r in success)  # type: ignore[arg-type]
    total_span = end_time - benchmark_start
    num_buckets = max(1, int(total_span / bucket_seconds) + 1)

    buckets: list[list[RequestRecord]] = [[] for _ in range(num_buckets)]
    for r in success:
        idx = int((r.completed_at - benchmark_start) / bucket_seconds)  # type: ignore[operator]
        idx = min(idx, num_buckets - 1)
        buckets[idx].append(r)

    result = []
    for i, bucket_records in enumerate(buckets):
        out_tokens = sum(r.output_tokens for r in bucket_records)
        ttft_vals = [r.ttft for r in bucket_records if r.ttft is not None]
        e2e_vals = [r.e2e_latency for r in bucket_records if r.e2e_latency is not None]

        ttft_dist = compute_distribution(ttft_vals)
        e2e_dist = compute_distribution(e2e_vals)

        result.append(
            {
                "time_offset": round(i * bucket_seconds, 1),
                "output_tokens": out_tokens,
                "throughput_tps": round(out_tokens / bucket_seconds, 1),
                "request_count": len(bucket_records),
                "ttft_p50": ttft_dist["p50"],
                "ttft_p95": ttft_dist["p95"],
                "e2e_p50": e2e_dist["p50"],
                "e2e_p95": e2e_dist["p95"],
            }
        )

    return result
