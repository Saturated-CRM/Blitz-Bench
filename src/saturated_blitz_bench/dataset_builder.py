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
import os
import random
import re
import uuid
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
    from datasets import load_dataset  # noqa: F401
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
# HuggingFace authentication
# ---------------------------------------------------------------------------

def _login_hf() -> None:
    """Load HF_TOKEN from .env file (if present) and login to HuggingFace Hub."""
    token = os.environ.get("HF_TOKEN")

    if not token:
        # Try loading from .env in project root
        for env_path in (Path(".env"), Path(__file__).resolve().parents[2] / ".env"):
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("HF_TOKEN=") and not line.startswith("#"):
                        token = line.split("=", 1)[1].strip().strip("'\"")
                        os.environ["HF_TOKEN"] = token
                        break
            if token:
                break

    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            logger.info("Authenticated with HuggingFace Hub")
        except Exception as e:
            logger.warning("HF login failed: %s (continuing without auth)", e)
    else:
        logger.warning(
            "No HF_TOKEN found. Set HF_TOKEN in .env or environment for "
            "faster downloads and access to gated datasets."
        )


# ---------------------------------------------------------------------------
# Category distribution and token ranges
# ---------------------------------------------------------------------------

CATEGORIES = {
    "short_chat":      {"weight": 0.25, "min_tok": 50,   "max_tok": 500,   "max_tokens": 1024,
                        "buckets": [(50, 100), (100, 200), (200, 350), (350, 500)]},
    "medium_chat":     {"weight": 0.15, "min_tok": 500,  "max_tok": 4000,  "max_tokens": 2048,
                        "buckets": [(500, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]},
    "tool_call":       {"weight": 0.20, "min_tok": 100,  "max_tok": 16000, "max_tokens": 4096,
                        "buckets": [(100, 500), (500, 2000), (2000, 5000), (5000, 16000)]},
    "code_generation": {"weight": 0.10, "min_tok": 500,  "max_tok": 8000,  "max_tokens": 4096,
                        "buckets": [(500, 1500), (1500, 3000), (3000, 5000), (5000, 8000)]},
    "long_context":    {"weight": 0.15, "min_tok": 20000, "max_tok": 70000, "max_tokens": 2048,
                        "buckets": [(20000, 32000), (32000, 45000), (45000, 58000), (58000, 70000)]},
    "multi_turn":      {"weight": 0.10, "min_tok": 2000, "max_tok": 16000, "max_tokens": 2048,
                        "buckets": [(2000, 4000), (4000, 8000), (8000, 12000), (12000, 16000)]},
    "reasoning":       {"weight": 0.05, "min_tok": 100,  "max_tok": 4000,  "max_tokens": 4096,
                        "buckets": [(100, 500), (500, 1000), (1000, 2000), (2000, 4000)]},
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
                and 100 <= first_user_tokens <= 4000
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


def _process_hermes_fc(
    enc: tiktoken.Encoding,
    target_count: int,
    max_input_tokens: int,
) -> list[dict]:
    """Process NousResearch/hermes-function-calling-v1 for tool_call (primary source).

    Configs: func_calling_singleturn (1.9K), glaive_func_calling (5.2K), func_calling (1.9K).
    Format: conversations [{from, value}], tools (JSON string).
    """
    if target_count <= 0:
        return []

    results: list[dict] = []
    logger.info("Loading Hermes function-calling-v1...")

    configs = ["func_calling_singleturn", "glaive_func_calling", "func_calling"]
    for config in configs:
        if len(results) >= target_count:
            break
        try:
            ds = load_dataset(
                "NousResearch/hermes-function-calling-v1",
                config, split="train",
            )
        except Exception as e:
            logger.warning("  Hermes-FC config '%s' unavailable: %s", config, e)
            continue

        for row in tqdm(ds, desc=f"Hermes-FC/{config}"):
            if len(results) >= target_count:
                break
            try:
                convos = row.get("conversations", [])
                tools_raw = row.get("tools", "")
                if not convos:
                    continue

                # Parse tools from JSON string
                tools: list[dict] = []
                if tools_raw:
                    try:
                        parsed_tools = json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
                        if isinstance(parsed_tools, list):
                            tools = parsed_tools
                    except json.JSONDecodeError:
                        pass

                # Build messages up to the assistant's tool call
                prompt_messages: list[dict[str, str]] = []
                expected_tool = None

                for msg in convos:
                    role_raw = msg.get("from", "")
                    value = msg.get("value", "")
                    if not value:
                        continue

                    if role_raw == "system":
                        prompt_messages.append({"role": "system", "content": value})
                    elif role_raw in ("human", "user"):
                        prompt_messages.append({"role": "user", "content": value})
                    elif role_raw in ("gpt", "assistant"):
                        # Check for tool_call in the response
                        tc_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', value, re.DOTALL)
                        if tc_match:
                            try:
                                tc_data = json.loads(tc_match.group(1))
                                expected_tool = {
                                    "name": tc_data.get("name", ""),
                                    "required_args": list(tc_data.get("arguments", {}).keys()),
                                }
                            except json.JSONDecodeError:
                                pass
                            break  # Stop before assistant response
                        prompt_messages.append({"role": "assistant", "content": value})

                if not prompt_messages or not any(m["role"] == "user" for m in prompt_messages):
                    continue

                tok_count = _count_messages_tokens(prompt_messages, enc)
                if tools:
                    tok_count += _count_tokens(json.dumps(tools), enc)

                if 100 <= tok_count <= min(16000, max_input_tokens):
                    results.append(_make_entry(
                        "tool_call", "hermes_fc", prompt_messages, tok_count, 4096,
                        {"dataset": "NousResearch/hermes-function-calling-v1", "config": config},
                        tools=tools or None, expected_tool=expected_tool,
                    ))
            except Exception:
                continue

    logger.info("  Hermes-FC -> tool_call: %d prompts", len(results))
    return results


def _process_glaive(
    enc: tiktoken.Encoding,
    target_count: int,
    max_input_tokens: int,
) -> list[dict]:
    """Process Glaive function-calling-v2 for tool_call category (supplement)."""
    if target_count <= 0:
        return []

    results: list[dict] = []
    logger.info("Loading Glaive function-calling-v2 (streaming)...")

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
            found_fc = False
            for part in parts[1:]:
                if "ASSISTANT: " not in part:
                    messages.append({"role": "user", "content": part.strip()})
                    continue
                user_part, rest = part.split("ASSISTANT: ", 1)
                messages.append({"role": "user", "content": user_part.strip()})

                if "<functioncall>" in rest:
                    fc_start = rest.index("<functioncall>") + len("<functioncall>")
                    # Find the end of the JSON object more robustly
                    fc_text = rest[fc_start:]
                    # Try to find closing tag or end of line
                    for end_marker in ("</functioncall>", "</", "\n\n", "\nFUNCTION"):
                        if end_marker in fc_text:
                            fc_text = fc_text[:fc_text.index(end_marker)]
                            break
                    fc_text = fc_text.strip()
                    # Handle single-quoted JSON (common in Glaive)
                    try:
                        fc_json = json.loads(fc_text)
                    except json.JSONDecodeError:
                        fc_text = fc_text.replace("'", '"')
                        try:
                            fc_json = json.loads(fc_text)
                        except json.JSONDecodeError:
                            continue
                    expected_tool = {
                        "name": fc_json.get("name", ""),
                        "required_args": list(fc_json.get("arguments", {}).keys())
                        if isinstance(fc_json.get("arguments"), dict) else [],
                    }
                    tok_count = _count_messages_tokens(messages, enc)
                    if 100 <= tok_count <= min(16000, max_input_tokens):
                        results.append(_make_entry(
                            "tool_call", "glaive", messages, tok_count, 4096,
                            {"dataset": "glaiveai/glaive-function-calling-v2"},
                            tools=tools or None, expected_tool=expected_tool,
                        ))
                    found_fc = True
                    break
                messages.append({"role": "assistant", "content": rest.strip()})

            # If no functioncall tag found, still usable as a tool_call prompt
            # (the system prompt has tool definitions, we just lack expected_tool)
            if not found_fc and tools and len(messages) >= 2:
                tok_count = _count_messages_tokens(messages, enc)
                if 100 <= tok_count <= min(16000, max_input_tokens):
                    results.append(_make_entry(
                        "tool_call", "glaive", messages, tok_count, 4096,
                        {"dataset": "glaiveai/glaive-function-calling-v2"},
                        tools=tools,
                    ))
        except Exception:
            continue

    logger.info("  Glaive -> tool_call: %d prompts", len(results))
    return results


def _process_gsm8k(
    enc: tiktoken.Encoding,
    target_count: int,
) -> list[dict]:
    """Process OpenAI GSM8K for reasoning category."""
    if target_count <= 0:
        return []

    results: list[dict] = []
    logger.info("Loading GSM8K...")

    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
    except Exception as e:
        logger.warning("  GSM8K unavailable: %s", e)
        return results

    for row in tqdm(ds, desc="GSM8K"):
        if len(results) >= target_count:
            break
        question = row.get("question", "")
        if not question or len(question.split()) < 10:
            continue

        messages = [
            {"role": "system", "content": "You are a helpful math tutor. Solve the problem step by step, showing your work clearly."},
            {"role": "user", "content": question},
        ]
        tok_count = _count_messages_tokens(messages, enc)
        if 100 <= tok_count <= 4000:
            results.append(_make_entry(
                "reasoning", "gsm8k", messages, tok_count, 4096,
                {"dataset": "openai/gsm8k"},
            ))

    logger.info("  GSM8K -> reasoning: %d prompts", len(results))
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
            ds = load_dataset("THUDM/LongBench", subset, split="test")
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
            if 20000 <= tok_count <= min(70000, max_input_tokens):
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
        if 20000 <= tok_count <= min(70000, max_input_tokens):
            results.append(_make_entry(
                "long_context", "longbench_v2", messages, tok_count, 2048,
                {"dataset": "THUDM/LongBench-v2"},
            ))

    logger.info("  LongBench-v2 -> long_context: %d prompts", len(results))
    return results


def _process_longalign(
    enc: tiktoken.Encoding,
    target_count: int,
    max_input_tokens: int,
) -> list[dict]:
    """Process THUDM/LongAlign-10k for long_context category.

    9,888 examples with `length` field (8K-64K tokens).
    Messages are in [{role, content}] format.
    """
    if target_count <= 0:
        return []

    results: list[dict] = []
    logger.info("Loading LongAlign-10k...")

    try:
        ds = load_dataset("THUDM/LongAlign-10k", split="train")
    except Exception as e:
        logger.warning("  LongAlign-10k unavailable: %s", e)
        return results

    for row in tqdm(ds, desc="LongAlign"):
        if len(results) >= target_count:
            break
        try:
            length = row.get("length", 0)
            # Quick pre-filter using the length field (total conversation length)
            # We want user input in 20K-70K range; the length includes assistant
            # response, so filter generously and measure accurately below
            if length < 15000:
                continue

            raw_messages = row.get("messages", [])
            if not raw_messages:
                continue

            # Take only user messages as the prompt
            prompt_messages: list[dict[str, str]] = []
            for msg in raw_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user" and content:
                    prompt_messages.append({"role": "user", "content": content})

            if not prompt_messages:
                continue

            tok_count = _count_messages_tokens(prompt_messages, enc)
            if 20000 <= tok_count <= min(70000, max_input_tokens):
                results.append(_make_entry(
                    "long_context", "longalign", prompt_messages, tok_count, 2048,
                    {"dataset": "THUDM/LongAlign-10k"},
                ))
        except Exception:
            continue

    logger.info("  LongAlign -> long_context: %d prompts", len(results))
    return results


def _process_code_contests(
    enc: tiktoken.Encoding,
    target_count: int,
    max_input_tokens: int,
) -> list[dict]:
    """Process deepmind/code_contests for code_generation category.

    3,760 train problems with description, public_tests, solutions.
    """
    if target_count <= 0:
        return []

    results: list[dict] = []
    logger.info("Loading code_contests...")

    try:
        ds = load_dataset("deepmind/code_contests", split="train")
    except Exception as e:
        logger.warning("  code_contests unavailable: %s", e)
        return results

    for row in tqdm(ds, desc="code_contests"):
        if len(results) >= target_count:
            break
        try:
            description = row.get("description", "")
            if not description or len(description) < 100:
                continue

            # Build a realistic coding prompt with examples
            public_tests = row.get("public_tests", {})
            test_inputs = public_tests.get("input", []) if isinstance(public_tests, dict) else []
            test_outputs = public_tests.get("output", []) if isinstance(public_tests, dict) else []

            user_content = f"{description}"
            if test_inputs and test_outputs:
                user_content += "\n\nExamples:"
                for i, (inp, out) in enumerate(zip(test_inputs[:3], test_outputs[:3])):
                    user_content += f"\n\nInput:\n{inp}\nOutput:\n{out}"

            user_content += "\n\nProvide an efficient solution in Python with clear comments."

            messages = [
                {"role": "system", "content": "You are an expert competitive programmer. Solve the following problem efficiently."},
                {"role": "user", "content": user_content},
            ]
            tok_count = _count_messages_tokens(messages, enc)
            if 500 <= tok_count <= min(8000, max_input_tokens):
                results.append(_make_entry(
                    "code_generation", "code_contests", messages, tok_count, 4096,
                    {"dataset": "deepmind/code_contests", "name": row.get("name", "")},
                ))
        except Exception:
            continue

    logger.info("  code_contests -> code_generation: %d prompts", len(results))
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
    logger.info("Loading SWE-smith trajectories (streaming)...")

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
            if 500 <= tok_count <= min(8000, max_input_tokens):
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
    output_path: str = "prompts/workload_pool.jsonl",
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

    Dataset sources:
      - WildChat-1M: short_chat, medium_chat, multi_turn, reasoning (keyword)
      - NousResearch hermes-function-calling-v1: tool_call (primary, 11.5K)
      - Glaive function-calling-v2: tool_call (supplement, 113K)
      - OpenAI GSM8K: reasoning (dedicated math/logic, 7.4K)
      - THUDM LongAlign-10k: long_context (primary, 9.9K, 8K-64K tokens)
      - THUDM LongBench + LongBench-v2: long_context (supplement)
      - deepmind code_contests: code_generation (primary, 3.7K)
      - SWE-bench SWE-smith-trajectories: code_generation (supplement)
    """
    _ensure_deps()

    out_path = Path(output_path)
    if out_path.exists() and not force:
        size_mb = out_path.stat().st_size / (1024 * 1024)
        count = sum(1 for line in open(out_path) if line.strip())
        print(
            f"\nDataset already exists at {out_path} "
            f"({count} prompts, {size_mb:.1f} MB).\n"
            f"Use --force to rebuild from scratch."
        )
        return

    # Authenticate with HuggingFace
    _login_hf()

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

    # ---- WildChat (short_chat, medium_chat, multi_turn, some reasoning) ----
    if any(cat in cats for cat in ("short_chat", "medium_chat", "multi_turn", "reasoning")):
        wc_targets = {
            cat: fetch_targets.get(cat, 0)
            for cat in ("short_chat", "medium_chat", "multi_turn", "reasoning")
            if cat in cats
        }
        wc = _process_wildchat(enc, wc_targets, max_input_tokens)
        for cat, prompts in wc.items():
            all_prompts.setdefault(cat, []).extend(prompts)

    # ---- GSM8K (dedicated reasoning) ----
    if "reasoning" in cats:
        all_prompts["reasoning"].extend(
            _process_gsm8k(enc, fetch_targets.get("reasoning", 0)))

    # ---- Hermes-FC (tool_call primary — 60%) ----
    if "tool_call" in cats:
        all_prompts["tool_call"].extend(
            _process_hermes_fc(enc, int(fetch_targets["tool_call"] * 0.60), max_input_tokens))

    # ---- Glaive (tool_call supplement — 40%) ----
    if "tool_call" in cats:
        all_prompts["tool_call"].extend(
            _process_glaive(enc, int(fetch_targets["tool_call"] * 0.40), max_input_tokens))

    # ---- LongAlign-10k (long_context primary — 60%) ----
    if "long_context" in cats:
        all_prompts["long_context"].extend(
            _process_longalign(enc, int(fetch_targets["long_context"] * 0.60), max_input_tokens))

    # ---- LongBench (long_context supplement — 25%) ----
    if "long_context" in cats:
        all_prompts["long_context"].extend(
            _process_longbench(enc, int(fetch_targets["long_context"] * 0.25), max_input_tokens))

    # ---- LongBench-v2 (long_context supplement — 15%) ----
    if "long_context" in cats:
        all_prompts["long_context"].extend(
            _process_longbench_v2(enc, int(fetch_targets["long_context"] * 0.15), max_input_tokens))

    # ---- code_contests (code_generation primary — 60%) ----
    if "code_generation" in cats:
        all_prompts["code_generation"].extend(
            _process_code_contests(enc, int(fetch_targets["code_generation"] * 0.60), max_input_tokens))

    # ---- SWE-smith (code_generation supplement — 40%) ----
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

    # ---- Export (JSONL: one JSON object per line) ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for prompt in final_prompts:
            f.write(json.dumps(prompt, separators=(",", ":")) + "\n")

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
