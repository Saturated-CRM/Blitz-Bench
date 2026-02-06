"""Track effective concurrency over time."""

from __future__ import annotations

import time
from threading import Lock


class ConcurrencyTracker:
    """Thread-safe tracker of in-flight request count over time.

    Samples are taken every second and used to compute
    the time-weighted average effective concurrency.
    """

    def __init__(self) -> None:
        self._active = 0
        self._lock = Lock()
        self._samples: list[tuple[float, int]] = []
        self._start_time = 0.0

    def start(self) -> None:
        self._start_time = time.time()

    def increment(self) -> None:
        with self._lock:
            self._active += 1

    def decrement(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)

    def sample(self) -> None:
        """Record the current concurrency level."""
        with self._lock:
            self._samples.append((time.time(), self._active))

    @property
    def current(self) -> int:
        with self._lock:
            return self._active

    def effective_concurrency(self) -> float:
        """Compute time-weighted average concurrency."""
        if len(self._samples) < 2:
            return 0.0

        total_weighted = 0.0
        total_time = 0.0
        for i in range(1, len(self._samples)):
            dt = self._samples[i][0] - self._samples[i - 1][0]
            total_weighted += self._samples[i - 1][1] * dt
            total_time += dt

        if total_time == 0:
            return 0.0
        return total_weighted / total_time

    def get_timeseries(self) -> list[tuple[float, int]]:
        """Return (relative_time, concurrency) pairs."""
        if not self._samples:
            return []
        t0 = self._samples[0][0]
        return [(t - t0, c) for t, c in self._samples]
