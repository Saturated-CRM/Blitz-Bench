"""Machine-readable JSON report output."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from saturated_blitz_bench.config import BenchmarkConfig
from saturated_blitz_bench.metrics.collector import RequestRecord
from saturated_blitz_bench.utils.helpers import utc_now_iso


def write_json_report(
    path: Path,
    metrics: dict[str, Any],
    timeseries: list[dict[str, Any]],
    concurrency_ts: list[tuple[float, int]],
    effective_concurrency: float,
    config: BenchmarkConfig,
) -> None:
    """Write comprehensive JSON report."""
    report = {
        "benchmark": "saturated-blitz-bench",
        "version": "0.1.0",
        "timestamp": utc_now_iso(),
        "config": {
            "endpoint": config.endpoint.model_dump(),
            "test": config.test.model_dump(),
            "metadata": config.metadata.model_dump(),
        },
        "metrics": metrics,
        "effective_concurrency": round(effective_concurrency, 1),
        "timeseries": timeseries,
        "concurrency_timeseries": [
            {"time": round(t, 1), "concurrency": c} for t, c in concurrency_ts
        ],
    }

    # Remove api_key from output
    report["config"]["endpoint"].pop("api_key", None)

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def write_csv_raw_data(path: Path, records: list[RequestRecord]) -> None:
    """Write per-request raw data as CSV."""
    if not records:
        return

    fieldnames = [
        "prompt_id",
        "category",
        "status",
        "request_sent_at",
        "first_token_at",
        "completed_at",
        "ttft",
        "e2e_latency",
        "output_tps",
        "input_tokens",
        "output_tokens",
        "tool_call_correct",
        "error",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = r.to_dict()
            # Only include the fieldnames we want
            writer.writerow({k: row.get(k, "") for k in fieldnames})
