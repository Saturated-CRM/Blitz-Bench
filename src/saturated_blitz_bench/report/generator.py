"""Orchestrate report generation (JSON, HTML, CSV)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from saturated_blitz_bench.config import BenchmarkConfig
from saturated_blitz_bench.metrics.calculator import compute_aggregate_metrics
from saturated_blitz_bench.metrics.collector import RequestRecord
from saturated_blitz_bench.metrics.concurrency_tracker import ConcurrencyTracker
from saturated_blitz_bench.metrics.timeseries import compute_timeseries
from saturated_blitz_bench.report.html_report import write_html_report
from saturated_blitz_bench.report.json_report import write_csv_raw_data, write_json_report

logger = logging.getLogger(__name__)


def generate_report(
    records: list[RequestRecord],
    config: BenchmarkConfig,
    concurrency_tracker: ConcurrencyTracker,
    benchmark_start_time: float,
    total_duration: float,
) -> dict[str, Any]:
    """Compute metrics and write all report outputs. Returns the metrics dict."""
    # Compute aggregate metrics
    metrics = compute_aggregate_metrics(
        records=records,
        total_duration=total_duration,
        warmup_seconds=config.test.warmup_seconds,
        gpu_count=config.metadata.gpu_count,
        benchmark_start_time=benchmark_start_time,
    )

    # Compute timeseries
    ts = compute_timeseries(records, benchmark_start_time)

    # Concurrency data
    concurrency_ts = concurrency_tracker.get_timeseries()
    effective_concurrency = concurrency_tracker.effective_concurrency()

    # Prepare output directory
    report_dir = Path(config.output.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    base_name = f"blitz-{timestamp}"

    fmt = config.output.format

    if fmt in ("json", "both"):
        json_path = report_dir / f"{base_name}.json"
        write_json_report(json_path, metrics, ts, concurrency_ts, effective_concurrency, config)
        logger.info("JSON report: %s", json_path)

    if fmt in ("html", "both"):
        html_path = report_dir / f"{base_name}.html"
        write_html_report(
            html_path, metrics, ts, concurrency_ts, effective_concurrency, config, records
        )
        logger.info("HTML report: %s", html_path)

    if config.output.include_raw_data:
        csv_path = report_dir / f"{base_name}.csv"
        write_csv_raw_data(csv_path, records)
        logger.info("Raw data CSV: %s", csv_path)

    return metrics
