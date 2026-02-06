"""Click CLI definition for saturated-blitz-bench."""

from __future__ import annotations

import asyncio
import logging
import sys
import time

import click

from saturated_blitz_bench import __version__
from saturated_blitz_bench.config import BenchmarkConfig, load_config
from saturated_blitz_bench.core.runner import BenchmarkRunner
from saturated_blitz_bench.report.generator import generate_report
from saturated_blitz_bench.utils.helpers import format_duration
from saturated_blitz_bench.utils.logging import console, setup_logging
from saturated_blitz_bench.workload.loader import load_prompt_pool

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """saturated-blitz-bench — AI inference provider stress benchmark."""


@main.command()
@click.option("--base-url", type=str, help="OpenAI-compatible base URL")
@click.option("--api-key", type=str, default="", help="API key")
@click.option("--model", type=str, help="Model name")
@click.option("--concurrency", type=int, help="Max concurrent requests")
@click.option("--duration", type=int, help="Test duration in seconds")
@click.option("--warmup", type=int, help="Warm-up seconds to exclude")
@click.option("--timeout", type=int, help="Per-request timeout in seconds")
@click.option("--temperature", type=float, help="Sampling temperature")
@click.option("--prompt-pool", type=str, help="Path to prompt pool JSON")
@click.option("--gpu-config", type=str, help="GPU configuration description")
@click.option("--gpu-count", type=int, help="Number of GPUs")
@click.option("--engine", type=str, help="Inference engine name")
@click.option("--config", "config_path", type=str, help="Path to config YAML file")
@click.option("--output-dir", type=str, help="Report output directory")
@click.option("--format", "output_format", type=click.Choice(["html", "json", "both"]))
@click.option("--verbose", is_flag=True, help="Enable debug logging")
def run(
    base_url: str | None,
    api_key: str | None,
    model: str | None,
    concurrency: int | None,
    duration: int | None,
    warmup: int | None,
    timeout: int | None,
    temperature: float | None,
    prompt_pool: str | None,
    gpu_config: str | None,
    gpu_count: int | None,
    engine: str | None,
    config_path: str | None,
    output_dir: str | None,
    output_format: str | None,
    verbose: bool,
) -> None:
    """Run the saturated-concurrency benchmark."""
    setup_logging(verbose)

    # Build CLI overrides dict
    overrides: dict = {}
    if base_url:
        overrides.setdefault("endpoint", {})["base_url"] = base_url
    if api_key is not None:
        overrides.setdefault("endpoint", {})["api_key"] = api_key
    if model:
        overrides.setdefault("endpoint", {})["model"] = model
    if concurrency:
        overrides.setdefault("test", {})["max_concurrency"] = concurrency
    if duration:
        overrides.setdefault("test", {})["duration_seconds"] = duration
    if warmup is not None:
        overrides.setdefault("test", {})["warmup_seconds"] = warmup
    if timeout:
        overrides.setdefault("test", {})["request_timeout"] = timeout
    if temperature is not None:
        overrides.setdefault("test", {})["temperature"] = temperature
    if prompt_pool:
        overrides.setdefault("workload", {})["prompt_pool"] = prompt_pool
    if gpu_config:
        overrides.setdefault("metadata", {})["gpu_config"] = gpu_config
    if gpu_count:
        overrides.setdefault("metadata", {})["gpu_count"] = gpu_count
    if engine:
        overrides.setdefault("metadata", {})["inference_engine"] = engine
    if output_dir:
        overrides.setdefault("output", {})["report_dir"] = output_dir
    if output_format:
        overrides.setdefault("output", {})["format"] = output_format

    config = load_config(config_path, overrides)

    console.print(
        f"\n[bold red]saturated-blitz-bench[/bold red] v{__version__}\n"
        f"  Model:       {config.endpoint.model}\n"
        f"  Endpoint:    {config.endpoint.base_url}\n"
        f"  Concurrency: {config.test.max_concurrency}\n"
        f"  Duration:    {format_duration(config.test.duration_seconds)}\n"
        f"  Warm-up:     {config.test.warmup_seconds}s\n"
    )

    # Load prompts
    try:
        pool = load_prompt_pool(config.workload.prompt_pool)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    console.print(f"  Prompts:     {pool.size} loaded\n")

    # Run benchmark
    runner = BenchmarkRunner(config, pool)
    start_time = time.time()

    try:
        records = asyncio.run(runner.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — generating report from collected data...[/yellow]")
        records = runner.results

    total_duration = time.time() - start_time

    console.print(f"\n[green]Benchmark complete in {format_duration(total_duration)}[/green]\n")

    # Generate report
    metrics = generate_report(
        records=records,
        config=config,
        concurrency_tracker=runner.concurrency_tracker,
        benchmark_start_time=start_time,
        total_duration=total_duration,
    )

    # Print summary
    console.print(f"  Total Requests:     {metrics['total_requests']:,}")
    console.print(f"  System Throughput:  {metrics['system_throughput_tps']:,.0f} tok/s")
    console.print(f"  RPM:                {metrics['rpm']:,.1f}")
    ttft_avg = metrics["ttft"]["avg"]
    if ttft_avg is not None:
        console.print(f"  Avg TTFT:           {ttft_avg * 1000:,.0f} ms")
    console.print(f"  Error Rate:         {metrics['error_rate'] * 100:.1f}%")
    console.print()


@main.command("build-dataset")
@click.option("--output", default="prompts/workload_pool.json", help="Output path")
@click.option("--total-prompts", default=5000, type=int, help="Total prompts to generate")
@click.option("--max-input-tokens", default=80000, type=int, help="Max input tokens")
@click.option("--tokenizer", default="cl100k_base", help="Tiktoken encoding")
@click.option("--quick", is_flag=True, help="Quick build (100 prompts)")
@click.option("--category", type=str, help="Build only one category")
@click.option("--force", is_flag=True, help="Force rebuild even if dataset already exists")
def build_dataset_cmd(
    output: str,
    total_prompts: int,
    max_input_tokens: int,
    tokenizer: str,
    quick: bool,
    category: str | None,
    force: bool,
) -> None:
    """Download and build the curated prompt pool from HuggingFace datasets."""
    setup_logging(False)

    if quick:
        total_prompts = 100

    console.print(f"\n[bold]Building dataset → {output}[/bold]")
    console.print(f"  Total prompts: {total_prompts}")
    console.print(f"  Max input tokens: {max_input_tokens}")
    console.print(f"  Tokenizer: {tokenizer}\n")

    try:
        from saturated_blitz_bench.dataset_builder import build_dataset

        build_dataset(
            output_path=output,
            total_prompts=total_prompts,
            max_input_tokens=max_input_tokens,
            tokenizer=tokenizer,
            category_filter=category,
            force=force,
        )
    except ImportError as e:
        console.print(
            f"[red]Error:[/red] {e}\n\n"
            "Install dataset build dependencies with:\n"
            "  pip install saturated-blitz-bench[dataset]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
