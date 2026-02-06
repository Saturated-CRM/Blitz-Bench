"""Rich HTML report with embedded Chart.js visualizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from saturated_blitz_bench.config import BenchmarkConfig
from saturated_blitz_bench.metrics.collector import RequestRecord
from saturated_blitz_bench.utils.helpers import utc_now_iso

TEMPLATE_DIR = Path(__file__).parent / "templates"


def write_html_report(
    path: Path,
    metrics: dict[str, Any],
    timeseries: list[dict[str, Any]],
    concurrency_ts: list[tuple[float, int]],
    effective_concurrency: float,
    config: BenchmarkConfig,
    records: list[RequestRecord],
) -> None:
    """Generate the self-contained HTML report."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,
    )
    template = env.get_template("report.html.j2")

    # Build scatter data (input_tokens vs ttft) for successful requests
    scatter_data = [
        {"input_tokens": r.input_tokens, "ttft": r.ttft, "category": r.category}
        for r in records
        if r.status == "success" and r.ttft is not None
    ]

    concurrency_json = [
        {"time": round(t, 1), "concurrency": c} for t, c in concurrency_ts
    ]

    html = template.render(
        config=config,
        metrics=metrics,
        effective_concurrency=round(effective_concurrency, 1),
        timestamp=utc_now_iso(),
        timeseries_json=json.dumps(timeseries, default=str),
        concurrency_json=json.dumps(concurrency_json),
        per_category_json=json.dumps(metrics.get("per_category", {}), default=str),
        scatter_json=json.dumps(scatter_data, default=str),
    )

    with open(path, "w") as f:
        f.write(html)
