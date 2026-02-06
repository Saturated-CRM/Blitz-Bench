"""Tests for metrics computation."""

import time

from saturated_blitz_bench.metrics.calculator import (
    compute_aggregate_metrics,
    compute_distribution,
    percentile,
)
from saturated_blitz_bench.metrics.collector import RequestRecord
from saturated_blitz_bench.metrics.concurrency_tracker import ConcurrencyTracker
from saturated_blitz_bench.metrics.timeseries import compute_timeseries


def test_percentile_basic():
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(vals, 50) == 5.5
    assert percentile(vals, 0) == 1
    assert percentile(vals, 100) == 10


def test_percentile_empty():
    assert percentile([], 50) is None


def test_compute_distribution():
    vals = list(range(1, 101))  # 1..100
    dist = compute_distribution(vals)
    assert dist["avg"] == 50.5
    assert dist["p50"] is not None
    assert dist["p99"] is not None


def test_compute_distribution_empty():
    dist = compute_distribution([])
    assert dist["avg"] is None
    assert dist["p50"] is None


def _make_record(
    status="success",
    ttft=0.1,
    e2e=1.0,
    tps=50.0,
    input_tokens=100,
    output_tokens=50,
    category="short_chat",
    sent_at=100.0,
) -> RequestRecord:
    r = RequestRecord(
        prompt_id="test",
        category=category,
        request_sent_at=sent_at,
        first_token_at=sent_at + ttft if ttft else None,
        completed_at=sent_at + e2e if e2e else None,
        ttft=ttft,
        e2e_latency=e2e,
        output_tps=tps,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        status=status,
    )
    return r


def test_aggregate_metrics_basic():
    records = [
        _make_record(sent_at=100.0, ttft=0.1, e2e=1.0, input_tokens=100, output_tokens=50),
        _make_record(sent_at=101.0, ttft=0.2, e2e=2.0, input_tokens=200, output_tokens=100),
        _make_record(sent_at=102.0, ttft=0.15, e2e=1.5, input_tokens=150, output_tokens=75),
    ]
    metrics = compute_aggregate_metrics(
        records, total_duration=10.0, warmup_seconds=0.0, benchmark_start_time=99.0,
    )
    assert metrics["total_requests"] == 3
    assert metrics["total_input_tokens"] == 450
    assert metrics["total_output_tokens"] == 225
    assert metrics["system_throughput_tps"] == 225 / 10.0
    assert metrics["error_rate"] == 0


def test_aggregate_metrics_with_errors():
    records = [
        _make_record(status="success", sent_at=100.0),
        _make_record(status="error", sent_at=101.0),
    ]
    metrics = compute_aggregate_metrics(
        records, total_duration=10.0, warmup_seconds=0.0, benchmark_start_time=99.0
    )
    assert metrics["total_requests"] == 1
    assert metrics["failed_requests"] == 1
    assert metrics["error_rate"] == 0.5


def test_aggregate_metrics_warmup_filter():
    records = [
        _make_record(sent_at=100.0),  # within warmup
        _make_record(sent_at=135.0),  # after warmup
    ]
    metrics = compute_aggregate_metrics(
        records, total_duration=60.0, warmup_seconds=30.0, benchmark_start_time=100.0
    )
    # Only the second record should be included
    assert metrics["total_requests"] == 1


def test_request_record_finalize():
    r = RequestRecord(
        prompt_id="test",
        category="short_chat",
        request_sent_at=100.0,
        input_tokens=100,
    )
    r.first_token_at = 100.15
    r.completed_at = 101.5
    r.output_tokens = 50
    r.token_timestamps = [100.15, 100.2, 100.25, 100.3]
    r.finalize()

    assert r.ttft is not None
    assert abs(r.ttft - 0.15) < 0.001
    assert r.e2e_latency is not None
    assert abs(r.e2e_latency - 1.5) < 0.001
    assert r.output_tps is not None
    assert len(r.itl_values) == 3


def test_concurrency_tracker():
    ct = ConcurrencyTracker()
    ct.start()
    assert ct.current == 0
    ct.increment()
    ct.increment()
    assert ct.current == 2
    ct.sample()
    ct.decrement()
    assert ct.current == 1
    ct.sample()


def test_timeseries_empty():
    result = compute_timeseries([], 100.0)
    assert result == []


def test_timeseries_basic():
    records = [
        _make_record(sent_at=100.0, e2e=1.0),
        _make_record(sent_at=105.0, e2e=2.0),
        _make_record(sent_at=115.0, e2e=1.5),
    ]
    # completed_at = sent_at + e2e
    ts = compute_timeseries(records, benchmark_start=100.0, bucket_seconds=10.0)
    assert len(ts) > 0
    assert ts[0]["time_offset"] == 0.0


def test_aggregate_metrics_all_errors():
    records = [
        _make_record(status="error", sent_at=100.0),
        _make_record(status="error", sent_at=101.0),
        _make_record(status="timeout", sent_at=102.0),
    ]
    metrics = compute_aggregate_metrics(
        records, total_duration=10.0, warmup_seconds=0.0, benchmark_start_time=99.0
    )
    # No success records → empty metrics
    assert metrics["total_requests"] == 0
    assert metrics["system_throughput_tps"] == 0


def test_aggregate_metrics_empty_records():
    metrics = compute_aggregate_metrics(
        [], total_duration=10.0, warmup_seconds=0.0, benchmark_start_time=99.0
    )
    assert metrics["total_requests"] == 0
    assert metrics["ttft"]["avg"] is None


def test_request_record_finalize_no_first_token():
    """Finalize with no first_token_at leaves ttft=None."""
    r = RequestRecord(
        prompt_id="test",
        category="short_chat",
        request_sent_at=100.0,
        input_tokens=100,
    )
    r.completed_at = 101.0
    r.output_tokens = 0
    r.finalize()
    assert r.ttft is None
    assert r.e2e_latency is not None
    assert abs(r.e2e_latency - 1.0) < 0.001
    assert r.output_tps is None


def test_concurrency_tracker_clamps_to_zero():
    ct = ConcurrencyTracker()
    ct.start()
    assert ct.current == 0
    ct.decrement()  # Would go negative without clamp
    assert ct.current == 0
    ct.decrement()  # Still clamped
    assert ct.current == 0


def test_concurrency_tracker_effective_one_sample():
    ct = ConcurrencyTracker()
    ct.start()
    ct.increment()
    ct.sample()
    # < 2 samples → returns 0.0
    assert ct.effective_concurrency() == 0.0
