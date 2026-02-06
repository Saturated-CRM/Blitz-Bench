"""RequestRecord dataclass and results collection."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestRecord:
    """Stores all metrics for a single benchmark request."""

    # Identity
    prompt_id: str = ""
    category: str = ""

    # Timing
    request_sent_at: float = 0.0
    first_token_at: float | None = None
    completed_at: float | None = None

    # Computed latencies (seconds)
    ttft: float | None = None
    e2e_latency: float | None = None
    output_tps: float | None = None
    itl_values: list[float] = field(default_factory=list)

    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0

    # Token arrival timestamps for ITL
    token_timestamps: list[float] = field(default_factory=list)

    # Tool call validation
    tool_call_correct: bool | None = None

    # Status
    status: str = "pending"  # success | timeout | error
    error: str = ""

    def compute_itl(self) -> None:
        """Compute inter-token latency from token timestamps."""
        if len(self.token_timestamps) < 2:
            self.itl_values = []
            return
        self.itl_values = [
            self.token_timestamps[i] - self.token_timestamps[i - 1]
            for i in range(1, len(self.token_timestamps))
        ]

    def finalize(self) -> None:
        """Compute derived metrics after request completes."""
        if self.first_token_at and self.request_sent_at:
            self.ttft = self.first_token_at - self.request_sent_at
        if self.completed_at and self.request_sent_at:
            self.e2e_latency = self.completed_at - self.request_sent_at
        if (
            self.completed_at
            and self.first_token_at
            and self.output_tokens > 0
        ):
            gen_time = self.completed_at - self.first_token_at
            if gen_time > 0:
                self.output_tps = self.output_tokens / gen_time
        self.compute_itl()

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "category": self.category,
            "request_sent_at": self.request_sent_at,
            "first_token_at": self.first_token_at,
            "completed_at": self.completed_at,
            "ttft": self.ttft,
            "e2e_latency": self.e2e_latency,
            "output_tps": self.output_tps,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "status": self.status,
            "error": self.error,
            "tool_call_correct": self.tool_call_correct,
            "itl_avg": (
                sum(self.itl_values) / len(self.itl_values)
                if self.itl_values
                else None
            ),
        }
