"""Structured logging and rich live progress display."""

from __future__ import annotations

import logging
import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

console = Console()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy libs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


class LiveProgress:
    """Rich live display showing benchmark progress."""

    def __init__(self, max_concurrency: int, duration_seconds: int) -> None:
        self.max_concurrency = max_concurrency
        self.duration_seconds = duration_seconds
        self.start_time = 0.0
        self.active = 0
        self.completed = 0
        self.errors = 0
        self.total_output_tokens = 0
        self._live: Live | None = None

    def start(self) -> None:
        self.start_time = time.time()
        self._live = Live(self._render(), console=console, refresh_per_second=2)
        self._live.start()

    def update(
        self,
        active: int,
        completed: int,
        errors: int,
        total_output_tokens: int,
    ) -> None:
        self.active = active
        self.completed = completed
        self.errors = errors
        self.total_output_tokens = total_output_tokens
        if self._live:
            self._live.update(self._render())

    def stop(self) -> None:
        if self._live:
            self._live.stop()

    def _render(self) -> Panel:
        elapsed = time.time() - self.start_time if self.start_time else 0
        remaining = max(0, self.duration_seconds - elapsed)
        throughput = self.total_output_tokens / elapsed if elapsed > 0 else 0
        rpm = (self.completed / elapsed * 60) if elapsed > 0 else 0

        elapsed_m, elapsed_s = divmod(int(elapsed), 60)
        total_m, total_s = divmod(self.duration_seconds, 60)

        pct = min(elapsed / self.duration_seconds, 1.0) if self.duration_seconds else 0
        bar_width = 35
        filled = int(bar_width * pct)
        bar = "█" * filled + "░" * (bar_width - filled)

        lines = [
            f" Active: {self.active}/{self.max_concurrency}  |  "
            f"Completed: {self.completed:,}  |  Errors: {self.errors}",
            f" Throughput: {throughput:,.0f} tok/s  |  RPM: {rpm:,.0f}",
            f" {bar}  {elapsed_m}:{elapsed_s:02d} / {total_m}:{total_s:02d}",
        ]

        return Panel(
            "\n".join(lines),
            title="[bold red]saturated-blitz-bench[/bold red]",
            border_style="red",
        )
