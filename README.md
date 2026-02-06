# saturated-blitz-bench

AI inference provider stress benchmark — saturated concurrency, realistic workloads, streaming-first metrics.

## Quick Start

source .venv/bin/activate

```bash
pip install saturated-blitz-bench

# Build the dataset (first time only)
saturated-blitz-bench build-dataset

# Run benchmark
saturated-blitz-bench run \
  --base-url http://localhost:8000/v1 \
  --model deepseek-ai/DeepSeek-V3 \
  --concurrency 64 \
  --duration 600
```

## What This Is

A 10-15 minute stress benchmark for AI inference providers. Unlike academic quality benchmarks, this measures operational performance: can your deployment handle the load?

- **Saturated concurrency** — asyncio semaphore ensures exactly N requests active at all times
- **Realistic workload** — mixed prompt categories from real HuggingFace datasets (50 to 80K input tokens)
- **Streaming-first** — measures TTFT, ITL, and per-request TPS as the user experiences them
- **Time-boxed** — runs for a configurable duration, not a fixed request count

## Installation

```bash
# Runtime only
pip install saturated-blitz-bench

# With dataset build tools
pip install saturated-blitz-bench[dataset]

# With dev tools
pip install saturated-blitz-bench[all]
```
