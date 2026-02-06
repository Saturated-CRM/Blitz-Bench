# saturated-blitz-bench

AI inference provider stress benchmark — saturated concurrency, realistic workloads, streaming-first metrics.

## What This Is

A stress benchmark for AI inference providers. Unlike academic quality benchmarks, this measures operational performance: can your deployment handle the load?

- **Saturated concurrency** — asyncio semaphore ensures exactly N requests active at all times
- **Realistic workload** — 7 prompt categories from real HuggingFace datasets (50 to 80K input tokens)
- **Streaming-first** — measures TTFT, ITL, and per-request TPS from the SSE stream
- **Tool call validation** — sends real tool definitions and validates the model's tool call response
- **Time-boxed** — runs for a configurable duration, not a fixed request count

## Installation

```bash
# From source (recommended during development)
pip install -e ".[all]"

# Runtime only
pip install saturated-blitz-bench

# With dataset build tools (needed for build-dataset command)
pip install saturated-blitz-bench[dataset]
```

## Quick Start

### 1. Build the dataset (first time only)

```bash
# Full build — 5000 prompts from HuggingFace (takes a few minutes)
saturated-blitz-bench build-dataset

# Quick build for testing — 100 prompts
saturated-blitz-bench build-dataset --quick

# Force rebuild (overwrites existing dataset)
saturated-blitz-bench build-dataset --force
```

This downloads and curates prompts from HuggingFace datasets into `prompts/workload_pool.jsonl`. You only need to do this once. If you need access to gated datasets, create a `.env` file with your HuggingFace token:

```
HF_TOKEN=hf_your_token_here
```

### 2. Dry run (no network, validates everything works)

```bash
saturated-blitz-bench run --dry-run
```

This simulates the full benchmark without hitting any endpoint. Useful for verifying your setup works. The `--dry-run` flag creates a simulated client with realistic timing (TTFT, ITL, token generation) and a small error/timeout rate.

You can combine `--dry-run` with any other flags:

```bash
# Dry run with custom concurrency and duration
saturated-blitz-bench run --dry-run --concurrency 128 --duration 120

# Dry run with verbose logging
saturated-blitz-bench run --dry-run --verbose
```

**Note:** Tool call accuracy will show 0% during dry runs. This is expected — the simulated client generates placeholder text, not actual tool calls. Tool call validation works correctly against real endpoints.

### 3. Run a real benchmark

```bash
# Basic test against a local vLLM/TGI/SGLang endpoint
saturated-blitz-bench run \
  --base-url http://localhost:8000/v1 \
  --model "your-model-name" \
  --concurrency 64 \
  --duration 600
```

**Quick smoke test first** (low concurrency, short duration):

```bash
saturated-blitz-bench run \
  --base-url http://localhost:8000/v1 \
  --model "your-model-name" \
  --concurrency 4 \
  --duration 30 \
  --warmup 0
```

**Full production benchmark** (e.g. GLM-4 on 8x H200):

```bash
saturated-blitz-bench run \
  --base-url http://your-server:8000/v1 \
  --api-key "your-api-key" \
  --model "glm-4" \
  --concurrency 256 \
  --duration 600 \
  --warmup 30 \
  --timeout 300 \
  --gpu-config "8x H200 SXM" \
  --gpu-count 8 \
  --engine "vLLM" \
  --output-dir ./reports \
  --format both
```

### 4. Using a config file

For repeatable benchmarks, use a YAML config file instead of CLI flags:

```bash
cp config.example.yaml my_benchmark.yaml
# Edit my_benchmark.yaml with your settings
saturated-blitz-bench run --config my_benchmark.yaml
```

CLI flags override config file values, so you can use both:

```bash
# Use config file but override concurrency
saturated-blitz-bench run --config my_benchmark.yaml --concurrency 128
```

## CLI Reference

```
saturated-blitz-bench run [OPTIONS]

Endpoint:
  --base-url TEXT      OpenAI-compatible base URL (e.g. http://localhost:8000/v1)
  --api-key TEXT       API key (optional)
  --model TEXT         Model name for the request

Test parameters:
  --concurrency INT    Max concurrent requests (default: 64)
  --duration INT       Test duration in seconds (default: 600)
  --warmup INT         Warm-up seconds excluded from metrics (default: 30)
  --timeout INT        Per-request timeout in seconds (default: 180)
  --temperature FLOAT  Sampling temperature (default: 0.7)

Workload:
  --prompt-pool TEXT   Path to workload pool JSONL file

Hardware metadata (shown in report):
  --gpu-config TEXT    GPU configuration (e.g. "8x H200 SXM")
  --gpu-count INT      Number of GPUs (enables per-GPU throughput calculation)
  --engine TEXT        Inference engine name (e.g. "vLLM 0.8.x")

Output:
  --config TEXT        Path to YAML config file
  --output-dir TEXT    Report output directory (default: ./reports)
  --format TEXT        Report format: html | json | both (default: both)
  --verbose            Enable debug logging
  --dry-run            Simulate benchmark without real LLM requests
```

## Output

Reports are saved to `./reports/` (configurable with `--output-dir`):

| File | Description |
|------|-------------|
| `blitz-<timestamp>.json` | Machine-readable metrics (throughput, latency distributions, per-category breakdown) |
| `blitz-<timestamp>.html` | Visual report with Chart.js graphs (throughput over time, latency scatter, concurrency) |
| `blitz-<timestamp>.csv` | Per-request raw data (prompt_id, category, TTFT, E2E latency, tokens, status, errors) |

## Metrics

| Metric | Description |
|--------|-------------|
| **System Throughput (tok/s)** | Total output tokens / effective duration |
| **RPM** | Successful requests per minute |
| **TTFT** | Time to first token (p50, p90, p95, p99) |
| **ITL** | Inter-token latency (p50, p90, p95, p99) |
| **E2E Latency** | End-to-end request latency (p50, p90, p95, p99) |
| **Output TPS** | Per-request output tokens per second |
| **Effective Concurrency** | Time-weighted average of in-flight requests |
| **Tool Call Accuracy** | % of tool_call prompts where the model returned the correct function name + required arguments |
| **Throughput per GPU** | System throughput / gpu_count (when `--gpu-count` is provided) |

All latency metrics exclude the warmup period.

## Workload Categories

| Category | Weight | Input Tokens | Description |
|----------|--------|-------------|-------------|
| short_chat | 25% | 50–500 | Quick Q&A, casual conversation |
| medium_chat | 15% | 500–4K | Detailed explanations, analysis |
| tool_call | 20% | 100–16K | Tool definitions + function calling |
| code_generation | 10% | 500–8K | Competitive programming, SWE tasks |
| long_context | 15% | 20K–70K | Document summarization, book analysis |
| multi_turn | 10% | 2K–16K | Multi-turn conversation history |
| reasoning | 5% | 100–4K | Math, logic, step-by-step reasoning |

Weights are configurable in the YAML config or by editing the distribution section.

## Tuning for Your Deployment

**For high-throughput setups** (8x H200, large batch servers):
- Use `--concurrency 256` or higher — the default 64 may not saturate the GPUs
- Use `--duration 600` (10 min) for stable measurements
- Use `--warmup 30` to let the server reach steady state

**For reasoning models** (DeepSeek-R1, GLM-4, QwQ):
- Reasoning models may generate long thinking traces (10K+ tokens)
- The default `max_tokens: 4096` for reasoning prompts caps output length
- To allow longer traces, edit the config YAML or rebuild the dataset with adjusted token limits
- The benchmark measures throughput/latency correctly regardless of thinking token format

**For tool-calling models**:
- 20% of the workload includes real tool definitions in OpenAI format
- Tool call accuracy is measured for prompts that have an `expected_tool` label
- The model must return the correct function name and required arguments to count as correct

## Requirements

- Python 3.10+
- An OpenAI-compatible inference endpoint (`/v1/chat/completions` with SSE streaming)
- Works with: vLLM, TGI, SGLang, Triton, OpenAI API, any OpenAI-compatible proxy
