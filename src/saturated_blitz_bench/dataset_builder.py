"""Download, filter, convert, and export the unified prompt pool from HuggingFace datasets.

Usage (via CLI):
    saturated-blitz-bench build-dataset
    saturated-blitz-bench build-dataset --quick
    saturated-blitz-bench build-dataset --total-prompts 5000

Usage (standalone):
    python -m saturated_blitz_bench.dataset_builder --output prompts/workload_pool.json
"""

from __future__ import annotations

import json
import logging
import random
import re
import uuid
from itertools import chain
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy dependency check — only fails when you actually call build_dataset()
# ---------------------------------------------------------------------------

_HAS_DEPS = True
_MISSING: list[str] = []

try:
    import tiktoken
except ImportError:
    _HAS_DEPS = False
    _MISSING.append("tiktoken")

try:
    from datasets import IterableDataset, load_dataset  # noqa: F401
except ImportError:
    _HAS_DEPS = False
    _MISSING.append("datasets")

try:
    from tqdm import tqdm
except ImportError:
    _HAS_DEPS = False
    _MISSING.append("tqdm")


def _ensure_deps() -> None:
    if not _HAS_DEPS:
        raise ImportError(
            f"Missing required packages: {', '.join(_MISSING)}\n"
            "Install them with:\n"
            f"  pip install {' '.join(_MISSING)}\n"
            "Or install the [dataset] extra:\n"
            "  pip install saturated-blitz-bench[dataset]"
        )


# ---------------------------------------------------------------------------
# Category distribution and token ranges
# ---------------------------------------------------------------------------

CATEGORIES = {
    "short_chat":      {"weight": 0.25, "min_tok": 50,   "max_tok": 500,   "max_tokens": 1024,
                        "buckets": [(50, 100), (100, 200), (200, 350), (350, 500)]},
    "medium_chat":     {"weight": 0.15, "min_tok": 500,  "max_tok": 4000,  "max_tokens": 2048,
                        "buckets": [(500, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]},
    "tool_call":       {"weight": 0.20, "min_tok": 200,  "max_tok": 16000, "max_tokens": 4096,
                        "buckets": [(200, 2000), (2000, 5000), (5000, 8000), (8000, 16000)]},
    "code_generation": {"weight": 0.10, "min_tok": 1000, "max_tok": 8000,  "max_tokens": 4096,
                        "buckets": [(1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000)]},
    "long_context":    {"weight": 0.15, "min_tok": 8000, "max_tok": 80000, "max_tokens": 2048,
                        "buckets": [(8000, 16000), (16000, 32000), (32000, 50000), (50000, 80000)]},
    "multi_turn":      {"weight": 0.10, "min_tok": 2000, "max_tok": 16000, "max_tokens": 2048,
                        "buckets": [(2000, 4000), (4000, 8000), (8000, 12000), (12000, 16000)]},
    "reasoning":       {"weight": 0.05, "min_tok": 500,  "max_tok": 4000,  "max_tokens": 4096,
                        "buckets": [(500, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]},
}

REASONING_KEYWORDS = [
    "math", "calculate", "solve", "proof", "prove", "equation", "theorem",
    "logic", "reasoning", "step by step", "think through", "analyze",
    "algorithm", "derive", "compute", "integral", "derivative", "probability",
    "statistics", "hypothesis", "deduce", "infer", "contradiction",
]

# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

_ENC_CACHE: dict[str, tiktoken.Encoding] = {}


def _get_encoding(name: str = "cl100k_base") -> tiktoken.Encoding:
    if name not in _ENC_CACHE:
        _ENC_CACHE[name] = tiktoken.get_encoding(name)
    return _ENC_CACHE[name]


def _count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text, disallowed_special=()))


def _count_messages_tokens(messages: list[dict[str, str]], enc: tiktoken.Encoding) -> int:
    total = 0
    for msg in messages:
        total += 4  # per-message overhead
        for v in msg.values():
            if isinstance(v, str):
                total += _count_tokens(v, enc)
    total += 2  # reply priming
    return total


def _make_entry(
    category: str,
    source: str,
    messages: list[dict[str, str]],
    token_count: int,
    max_tokens: int,
    source_metadata: dict | None = None,
    tools: list[dict] | None = None,
    expected_tool: dict | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": uuid.uuid4().hex[:12],
        "category": category,
        "source": source,
        "messages": messages,
        "max_tokens": max_tokens,
        "input_token_count": token_count,
        "source_metadata": source_metadata or {},
    }
    if tools:
        entry["tools"] = tools
    if expected_tool:
        entry["expected_tool"] = expected_tool
    return entry


# ---------------------------------------------------------------------------
# Source-specific converters
# ---------------------------------------------------------------------------

def _process_wildchat(
    enc: tiktoken.Encoding,
    targets: dict[str, int],
    max_input_tokens: int,
) -> dict[str, list[dict]]:
    """Process WildChat-1M for short_chat, medium_chat, multi_turn, reasoning."""
    results: dict[str, list[dict]] = {
        "short_chat": [], "medium_chat": [], "multi_turn": [], "reasoning": [],
    }
    needed = sum(targets.get(cat, 0) for cat in results)
    if needed == 0:
        return results

    logger.info("Loading WildChat-1M (streaming)...")
    ds = load_dataset(
        "allenai/WildChat-1M", split="train",
        streaming=True,
    )

    seen = 0
    for row in tqdm(ds, desc="WildChat", total=needed * 10):
        if row.get("language") and row["language"] != "English":
            continue
        conversation = row.get("conversation", [])
        if not conversation:
            continue
        if row.get("toxic", False) or row.get("redacted", False):
            continue

        first_user = None
        for msg in conversation:
            if msg.get("role") == "user" and msg.get("content", "").strip():
                first_user = msg["content"].strip()
                break
        if not first_user or len(first_user.split()) < 3:
            continue

        first_user_tokens = _count_tokens(first_user, enc)
        messages_single = [{"role": "user", "content": first_user}]
        single_tok = _count_messages_tokens(messages_single, enc)
        meta = {"dataset": "allenai/WildChat-1M",
                "original_id": row.get("conversation_hash", "")}

        if (len(results["short_chat"]) < targets.get("short_chat", 0)
                and 50 <= first_user_tokens <= 500):
            results["short_chat"].append(
                _make_entry("short_chat", "wildchat", messages_single, single_tok, 1024, meta))
        elif (len(results["medium_chat"]) < targets.get("medium_chat", 0)
                and 500 < first_user_tokens <= 4000):
            results["medium_chat"].append(
                _make_entry("medium_chat", "wildchat", messages_single, single_tok, 2048, meta))
        elif (len(results["reasoning"]) < targets.get("reasoning", 0)
                and 500 <= first_user_tokens <= 4000
                and any(kw in first_user.lower() for kw in REASONING_KEYWORDS)):
            results["reasoning"].append(
                _make_entry("reasoning", "wildchat", messages_single, single_tok, 4096, meta))

        # Multi-turn (independent of the above)
        if (len(results["multi_turn"]) < targets.get("multi_turn", 0)
                and len(conversation) >= 5):
            mt_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in conversation
                if m.get("role") in ("user", "assistant") and m.get("content", "").strip()
            ]
            if len(mt_messages) >= 5:
                mt_tok = _count_messages_tokens(mt_messages, enc)
                if 2000 <= mt_tok <= min(16000, max_input_tokens):
                    results["multi_turn"].append(
                        _make_entry("multi_turn", "wildchat", mt_messages, mt_tok, 2048, meta))

        if all(len(results[c]) >= targets.get(c, 0) for c in results):
            break
        seen += 1
        if seen > needed * 50:
            break

    for cat in results:
        logger.info("  WildChat -> %s: %d prompts", cat, len(results[cat]))
    return results


def _process_nemotron(
    enc: tiktoken.Encoding,
    target_count: int,
    max_input_tokens: int,
) -> list[dict]:
    """Process Nemotron-Agentic-v1 for tool_call category."""
    if target_count <= 0:
        return []

    results: list[dict] = []
    logger.info("Loading Nemotron-Agentic-v1...")

    # Nemotron uses named splits, not "train"
    streams = []
    for split_name in ("tool_calling", "interactive_agent"):
        try:
            s = load_dataset(
                "nvidia/Nemotron-Agentic-v1", split=split_name,
                streaming=True,
            )
            streams.append(s)
        except Exception as e:
            logger.warning("  Nemotron split '%s' unavailable: %s", split_name, e)

    if not streams:
        logger.warning("  Nemotron: no splits loaded")
        return results

    combined = chain(*(iter(s) for s in streams))

    for row in tqdm(combined, desc="Nemotron", total=target_count * 5):
        if len(results) >= target_count:
            break
        try:
            conversations = row.get("conversations") or row.get("messages", [])
            tools_raw = row.get("tools", [])
            if not conversations or not tools_raw:
                continue

            # Normalise tools to list[dict]
            if isinstance(tools_raw, str):
                try:
                    tools_raw = json.loads(tools_raw)
                except json.JSONDecodeError:
                    continue
            if not isinstance(tools_raw, list):
                continue

            prompt_messages: list[dict[str, str]] = []
            expected_tool_info = None

            for msg in conversations:
                role = msg.get("role", msg.get("from", ""))
                content = msg.get("content", msg.get("value", ""))
                if role in ("system", "user"):
                    prompt_messages.append({"role": role, "content": str(content)})
                elif role == "assistant":
                    tc_list = msg.get("tool_calls", [])
                    if tc_list:
                        tc = tc_list[0] if isinstance(tc_list, list) else tc_list
                        func = tc.get("function", tc) if isinstance(tc, dict) else {}
                        args_raw = func.get("arguments", "{}")
                        try:
                            parsed_args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                        except json.JSONDecodeError:
                            parsed_args = {}
                        expected_tool_info = {
                            "name": func.get("name", ""),
                            "required_args": list(parsed_args.keys()),
                        }
                        break
                    prompt_messages.append({"role": "assistant", "content": str(content)})
                elif role == "tool":
                    prompt_messages.append({"role": "tool", "content": str(content)})

            if not prompt_messages or not expected_tool_info:
                continue

            tok_count = _count_messages_tokens(prompt_messages, enc)
            if 200 <= tok_count <= min(16000, max_input_tokens):
                results.append(_make_entry(
                    "tool_call", "nemotron", prompt_messages, tok_count, 4096,
                    {"dataset": "nvidia/Nemotron-Agentic-v1"},
                    tools=tools_raw, expected_tool=expected_tool_info,
                ))
        except Exception:
            continue

    logger.info("  Nemotron -> tool_call: %d prompts", len(results))
    return results


def _process_glaive(
    enc: tiktoken.Encoding,
    target_count: int,
    max_input_tokens: int,
) -> list[dict]:
    """Process Glaive function calling for additional tool_call variety."""
    if target_count <= 0:
        return []

    results: list[dict] = []
    logger.info("Loading Glaive function-calling-v2...")

    try:
        ds = load_dataset(
            "glaiveai/glaive-function-calling-v2", split="train",
            streaming=True,
        )
    except Exception as e:
        logger.warning("  Glaive unavailable: %s", e)
        return results

    for row in tqdm(ds, desc="Glaive", total=target_count * 5):
        if len(results) >= target_count:
            break
        try:
            system_prompt = row.get("system", "")
            chat_text = row.get("chat", "")
            if not chat_text:
                continue

            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Try to extract tool schemas from system prompt
            tools: list[dict] = []
            try:
                for match in re.findall(r'\[[\s\S]*?\]', system_prompt):
                    parsed = json.loads(match)
                    if isinstance(parsed, list) and parsed and "name" in str(parsed[0]):
                        tools = [
                            {"type": "function", "function": f} if "type" not in f else f
                            for f in parsed
                        ]
                        break
            except (json.JSONDecodeError, Exception):
                pass

            parts = chat_text.split("USER: ")
            for part in parts[1:]:
                if "ASSISTANT: " not in part:
                    messages.append({"role": "user", "content": part.strip()})
                    continue
                user_part, rest = part.split("ASSISTANT: ", 1)
                messages.append({"role": "user", "content": user_part.strip()})

                if "<functioncall>" in rest:
                    fc_start = rest.index("<functioncall>") + len("<functioncall>")
                    fc_end = rest.index("</", fc_start) if "</" in rest[fc_start:] else len(rest)
                    fc_json = json.loads(rest[fc_start:fc_end].strip())
                    expected_tool = {
                        "name": fc_json.get("name", ""),
                        "required_args": list(fc_json.get("arguments", {}).keys()),
                    }
                    tok_count = _count_messages_tokens(messages, enc)
                    if 200 <= tok_count <= min(16000, max_input_tokens):
                        results.append(_make_entry(
                            "tool_call", "glaive", messages, tok_count, 4096,
                            {"dataset": "glaiveai/glaive-function-calling-v2"},
                            tools=tools or None, expected_tool=expected_tool,
                        ))
                    break
                messages.append({"role": "assistant", "content": rest.strip()})
        except Exception:
            continue

    logger.info("  Glaive -> tool_call: %d prompts", len(results))
    return results


def _process_bfcl(
    enc: tiktoken.Encoding,
    target_count: int,
    max_input_tokens: int,
) -> list[dict]:
    """Process Berkeley Function Calling Leaderboard for tool_call variety."""
    if target_count <= 0:
        return []

    results: list[dict] = []
    logger.info("Loading BFCL...")

    # BFCL has many configs; some have JSON schema issues.
    # Try the safe ones first and catch errors per-row.
    configs_to_try = [
        "BFCL_v3_live_simple", "BFCL_v3_live_multiple",
        "BFCL_v3_simple", "BFCL_v3_multiple",
    ]
    bfcl_rows: list[dict] = []
    for config in configs_to_try:
        try:
            ds = load_dataset(
                "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                config, split="train", streaming=True,
            )
            for row in ds:
                bfcl_rows.append(row)
                if len(bfcl_rows) >= target_count * 10:
                    break
        except Exception as e:
            logger.debug("  BFCL config '%s' skipped: %s", config, type(e).__name__)
            continue
        if len(bfcl_rows) >= target_count * 10:
            break

    if not bfcl_rows:
        logger.warning("  BFCL: no configs loaded successfully")
        return results

    for row in tqdm(bfcl_rows, desc="BFCL"):
        if len(results) >= target_count:
            break
        try:
            question_raw = row.get("question", "")
            if isinstance(question_raw, list):
                question_text = "\n".join(
                    q.get("content", q.get("text", str(q))) if isinstance(q, dict) else str(q)
                    for q in question_raw
                )
            else:
                question_text = str(question_raw)
            if not question_text.strip():
                continue

            func_defs = row.get("function", [])
            if isinstance(func_defs, str):
                func_defs = json.loads(func_defs)
            if not func_defs:
                continue

            tools: list[dict] = []
            for f in (func_defs if isinstance(func_defs, list) else [func_defs]):
                if isinstance(f, dict):
                    tools.append({"type": "function", "function": f} if "type" not in f else f)

            messages = [{"role": "user", "content": question_text.strip()}]
            tok_count = _count_messages_tokens(messages, enc)
            if 100 <= tok_count <= min(16000, max_input_tokens):
                expected_tool = None
                gt = row.get("ground_truth", row.get("answer", ""))
                if gt:
                    try:
                        gt_parsed = json.loads(gt) if isinstance(gt, str) else gt
                        if isinstance(gt_parsed, list) and gt_parsed:
                            gt_parsed = gt_parsed[0]
                        if isinstance(gt_parsed, dict):
                            expected_tool = {
                                "name": gt_parsed.get("name", list(gt_parsed.keys())[0] if gt_parsed else ""),
                                "required_args": [],
                            }
                    except Exception:
                        pass

                results.append(_make_entry(
                    "tool_call", "bfcl", messages, tok_count, 4096,
                    {"dataset": "gorilla-llm/Berkeley-Function-Calling-Leaderboard"},
                    tools=tools, expected_tool=expected_tool,
                ))
        except Exception:
            continue

    logger.info("  BFCL -> tool_call: %d prompts", len(results))
    return results


def _process_longbench(
    enc: tiktoken.Encoding,
    target_count: int,
    max_input_tokens: int,
) -> list[dict]:
    """Process LongBench for long_context category."""
    if target_count <= 0:
        return []

    results: list[dict] = []
    subsets = ["gov_report", "multi_news", "narrativeqa"]

    for subset in subsets:
        if len(results) >= target_count:
            break
        logger.info("  Loading LongBench/%s...", subset)
        try:
            ds = load_dataset(
                "THUDM/LongBench", subset, split="test",
            )
        except Exception as e:
            logger.warning("  LongBench/%s unavailable: %s", subset, e)
            continue

        for row in tqdm(ds, desc=f"LongBench/{subset}"):
            if len(results) >= target_count:
                break
            context = row.get("context", "")
            question = row.get("input", "")
            if not context or not question:
                continue

            if subset == "gov_report":
                user_content = f"{context}\n\n---\n\nPlease provide a comprehensive summary of the above report."
            elif subset == "multi_news":
                user_content = f"{context}\n\n---\n\nBased on the articles above, write a multi-document summary."
            else:
                user_content = f"{context}\n\n---\n\nBased on the above, {question}"

            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer based on the provided context."},
                {"role": "user", "content": user_content},
            ]
            tok_count = _count_messages_tokens(messages, enc)
            if 8000 <= tok_count <= min(max_input_tokens, 80000):
                results.append(_make_entry(
                    "long_context", "longbench", messages, tok_count, 2048,
                    {"dataset": f"THUDM/LongBench/{subset}"},
                ))

    logger.info("  LongBench -> long_context: %d prompts", len(results))
    return results


def _process_longbench_v2(
    enc: tiktoken.Encoding,
    target_count: int,
    max_input_tokens: int,
) -> list[dict]:
    """Process LongBench-v2 for additional long_context prompts."""
    if target_count <= 0:
        return []

    results: list[dict] = []
    logger.info("  Loading LongBench-v2...")

    try:
        ds = load_dataset("THUDM/LongBench-v2", split="train")
    except Exception as e:
        logger.warning("  LongBench-v2 unavailable: %s", e)
        return results

    for row in tqdm(ds, desc="LongBench-v2"):
        if len(results) >= target_count:
            break
        context = row.get("context", "")
        question = row.get("question", row.get("input", ""))
        if not context or not question:
            continue

        user_content = f"{context}\n\n---\n\nBased on the above text, please answer: {question}"
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Provide a detailed answer based on the context."},
            {"role": "user", "content": user_content},
        ]
        tok_count = _count_messages_tokens(messages, enc)
        if 8000 <= tok_count <= min(max_input_tokens, 80000):
            results.append(_make_entry(
                "long_context", "longbench_v2", messages, tok_count, 2048,
                {"dataset": "THUDM/LongBench-v2"},
            ))

    logger.info("  LongBench-v2 -> long_context: %d prompts", len(results))
    return results


def _process_swe_smith(
    enc: tiktoken.Encoding,
    target_count: int,
    max_input_tokens: int,
) -> list[dict]:
    """Process SWE-smith trajectories for code_generation category."""
    if target_count <= 0:
        return []

    results: list[dict] = []
    logger.info("Loading SWE-smith trajectories...")

    try:
        ds = load_dataset(
            "SWE-bench/SWE-smith-trajectories", split="train",
            streaming=True,
        )
    except Exception as e:
        logger.warning("  SWE-smith unavailable: %s", e)
        return results

    for row in tqdm(ds, desc="SWE-smith", total=target_count * 5):
        if len(results) >= target_count:
            break
        try:
            messages_raw = row.get("messages", "")
            if isinstance(messages_raw, str):
                messages_raw = json.loads(messages_raw)
            if not isinstance(messages_raw, list) or len(messages_raw) < 2:
                continue

            prompt_messages: list[dict[str, str]] = []
            for msg in messages_raw[:6]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("system", "user", "assistant") and content:
                    prompt_messages.append({"role": role, "content": str(content)})
            if not prompt_messages:
                continue
            if prompt_messages[-1]["role"] != "user":
                prompt_messages.append({
                    "role": "user",
                    "content": "Based on the above context, analyze the issue and provide a fix with code.",
                })

            tok_count = _count_messages_tokens(prompt_messages, enc)
            if 1000 <= tok_count <= min(8000, max_input_tokens):
                results.append(_make_entry(
                    "code_generation", "swe_smith", prompt_messages, tok_count, 4096,
                    {"dataset": "SWE-bench/SWE-smith-trajectories"},
                ))
        except Exception:
            continue

    logger.info("  SWE-smith -> code_generation: %d prompts", len(results))
    return results


# ---------------------------------------------------------------------------
# Sampling & balancing
# ---------------------------------------------------------------------------

def _sample_balanced(
    prompts: list[dict],
    category: str,
    target: int,
) -> list[dict]:
    """Sample prompts uniformly across token-length buckets for a category."""
    buckets_spec = CATEGORIES[category]["buckets"]
    per_bucket = max(1, target // len(buckets_spec))

    bucketed: dict[int, list[dict]] = {i: [] for i in range(len(buckets_spec))}
    for p in prompts:
        tok = p["input_token_count"]
        for i, (lo, hi) in enumerate(buckets_spec):
            if lo <= tok < hi:
                bucketed[i].append(p)
                break

    sampled: list[dict] = []
    for i in range(len(buckets_spec)):
        pool = bucketed[i]
        if pool:
            sampled.extend(random.sample(pool, min(per_bucket, len(pool))))

    remaining = target - len(sampled)
    if remaining > 0:
        used_ids = {p["id"] for p in sampled}
        leftover = [p for p in prompts if p["id"] not in used_ids]
        if leftover:
            sampled.extend(random.sample(leftover, min(remaining, len(leftover))))

    return sampled[:target]


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_dataset(
    output_path: str = "prompts/workload_pool.json",
    total_prompts: int = 5000,
    max_input_tokens: int = 80000,
    tokenizer: str = "cl100k_base",
    category_filter: str | None = None,
    force: bool = False,
) -> None:
    """Main dataset build pipeline.

    If the output file already exists and ``force`` is False, the build is
    skipped and the existing dataset is reused.  HuggingFace ``datasets``
    caches raw downloads automatically (~/.cache/huggingface/), so re-runs
    only re-download what is missing.
    """
    _ensure_deps()

    out_path = Path(output_path)
    if out_path.exists() and not force:
        size_mb = out_path.stat().st_size / (1024 * 1024)
        with open(out_path) as f:
            count = len(json.load(f))
        print(
            f"\nDataset already exists at {out_path} "
            f"({count} prompts, {size_mb:.1f} MB).\n"
            f"Use --force to rebuild from scratch."
        )
        return

    enc = _get_encoding(tokenizer)

    # Compute per-category targets
    if category_filter:
        cats = {category_filter: CATEGORIES[category_filter]}
        targets = {category_filter: total_prompts}
    else:
        cats = CATEGORIES
        targets = {cat: max(1, int(total_prompts * spec["weight"])) for cat, spec in cats.items()}

    # Overshoot 3x to give balanced sampling enough material
    fetch_targets = {cat: count * 3 for cat, count in targets.items()}

    logger.info("Target counts: %s", targets)
    all_prompts: dict[str, list[dict]] = {cat: [] for cat in cats}

    # ---- WildChat ----
    if any(cat in cats for cat in ("short_chat", "medium_chat", "multi_turn", "reasoning")):
        wc_targets = {
            cat: fetch_targets.get(cat, 0)
            for cat in ("short_chat", "medium_chat", "multi_turn", "reasoning")
            if cat in cats
        }
        wc = _process_wildchat(enc, wc_targets, max_input_tokens)
        for cat, prompts in wc.items():
            all_prompts.setdefault(cat, []).extend(prompts)

    # ---- Nemotron (tool_call primary) ----
    if "tool_call" in cats:
        all_prompts["tool_call"].extend(
            _process_nemotron(enc, int(fetch_targets["tool_call"] * 0.75), max_input_tokens))

    # ---- Glaive (tool_call supplement) ----
    if "tool_call" in cats:
        all_prompts["tool_call"].extend(
            _process_glaive(enc, int(fetch_targets["tool_call"] * 0.15), max_input_tokens))

    # ---- BFCL (tool_call supplement) ----
    if "tool_call" in cats:
        all_prompts["tool_call"].extend(
            _process_bfcl(enc, int(fetch_targets["tool_call"] * 0.10), max_input_tokens))

    # ---- LongBench ----
    if "long_context" in cats:
        all_prompts["long_context"].extend(
            _process_longbench(enc, int(fetch_targets["long_context"] * 0.67), max_input_tokens))

    # ---- LongBench-v2 ----
    if "long_context" in cats:
        all_prompts["long_context"].extend(
            _process_longbench_v2(enc, int(fetch_targets["long_context"] * 0.33), max_input_tokens))

    # ---- SWE-smith (code_generation) ----
    if "code_generation" in cats:
        all_prompts["code_generation"].extend(
            _process_swe_smith(enc, fetch_targets.get("code_generation", 0), max_input_tokens))

    # ---- Sample & Balance ----
    logger.info("Sampling and balancing...")
    final_prompts: list[dict] = []
    stats: dict[str, Any] = {"per_category": {}, "total": 0}

    for cat, target_count in targets.items():
        pool = all_prompts.get(cat, [])
        if not pool:
            logger.warning("No prompts available for category '%s'", cat)
            continue
        sampled = _sample_balanced(pool, cat, target_count)
        final_prompts.extend(sampled)
        stats["per_category"][cat] = {
            "count": len(sampled),
            "target": target_count,
            "pool_size": len(pool),
            "token_range": [
                min(p["input_token_count"] for p in sampled) if sampled else 0,
                max(p["input_token_count"] for p in sampled) if sampled else 0,
            ],
        }

    random.shuffle(final_prompts)
    stats["total"] = len(final_prompts)

    # ---- Export ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final_prompts, f, separators=(",", ":"))  # compact

    stats_path = out_path.with_name(out_path.stem + "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    file_size_mb = out_path.stat().st_size / (1024 * 1024)

    print(f"\nDataset built successfully:")
    print(f"  Total prompts: {len(final_prompts)}")
    print(f"  File size:     {file_size_mb:.1f} MB")
    print(f"  Output:        {out_path}")
    print(f"  Stats:         {stats_path}")
    print(f"\nPer-category breakdown:")
    for cat, info in stats["per_category"].items():
        print(
            f"  {cat:20s}: {info['count']:5d} / {info['target']:5d} target "
            f"(pool: {info['pool_size']}, tokens: {info['token_range'][0]}-{info['token_range'][1]})"
        )


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Build the saturated-blitz-bench prompt pool")
    parser.add_argument("--output", default="prompts/workload_pool.json")
    parser.add_argument("--total-prompts", type=int, default=5000)
    parser.add_argument("--max-input-tokens", type=int, default=80000)
    parser.add_argument("--tokenizer", default="cl100k_base")
    parser.add_argument("--quick", action="store_true", help="Quick build (100 prompts)")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Force rebuild even if dataset exists")
    args = parser.parse_args()

    if args.quick:
        args.total_prompts = 100

    build_dataset(
        output_path=args.output,
        total_prompts=args.total_prompts,
        max_input_tokens=args.max_input_tokens,
        tokenizer=args.tokenizer,
        category_filter=args.category,
        force=args.force,
    )
