"""Configuration loading: YAML file + CLI overrides, validated with pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class EndpointConfig(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    model: str = "deepseek-ai/DeepSeek-V3"


class TestConfig(BaseModel):
    max_concurrency: int = Field(default=64, ge=1)
    duration_seconds: int = Field(default=600, ge=10)
    warmup_seconds: int = Field(default=30, ge=0)
    request_timeout: int = Field(default=180, ge=10)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class WorkloadDistribution(BaseModel):
    short_chat: float = 0.25
    medium_chat: float = 0.15
    tool_call: float = 0.20
    code_generation: float = 0.10
    long_context: float = 0.15
    multi_turn: float = 0.10
    reasoning: float = 0.05

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "WorkloadDistribution":
        total = (
            self.short_chat
            + self.medium_chat
            + self.tool_call
            + self.code_generation
            + self.long_context
            + self.multi_turn
            + self.reasoning
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Distribution weights must sum to 1.0, got {total}")
        return self


class WorkloadConfig(BaseModel):
    prompt_pool: str = "prompts/workload_pool.jsonl"
    distribution: WorkloadDistribution = WorkloadDistribution()


class MetadataConfig(BaseModel):
    deployment_name: str = ""
    gpu_config: str = ""
    gpu_count: int | None = None
    inference_engine: str = ""
    quantization: str = ""
    tensor_parallel: int | None = None
    max_model_len: int | None = None
    notes: str = ""


class OutputConfig(BaseModel):
    report_dir: str = "./reports"
    format: str = Field(default="both", pattern="^(html|json|both)$")
    include_raw_data: bool = True


class BenchmarkConfig(BaseModel):
    endpoint: EndpointConfig = EndpointConfig()
    test: TestConfig = TestConfig()
    workload: WorkloadConfig = WorkloadConfig()
    metadata: MetadataConfig = MetadataConfig()
    output: OutputConfig = OutputConfig()


def load_config(
    config_path: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> BenchmarkConfig:
    """Load config from YAML file, then apply CLI overrides."""
    data: dict[str, Any] = {}

    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}

    if cli_overrides:
        _deep_merge(data, cli_overrides)

    return BenchmarkConfig(**data)


def _deep_merge(base: dict, override: dict) -> None:
    """Merge override dict into base dict recursively (in-place)."""
    for key, value in override.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
