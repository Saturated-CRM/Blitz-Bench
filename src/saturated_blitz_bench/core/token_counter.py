"""Token counting with tiktoken fallback."""

from __future__ import annotations

import logging
from functools import lru_cache

import tiktoken

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _get_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in a text string using tiktoken."""
    enc = _get_encoding(encoding_name)
    return len(enc.encode(text, disallowed_special=()))


def count_messages_tokens(
    messages: list[dict[str, str]],
    encoding_name: str = "cl100k_base",
) -> int:
    """Approximate token count for a chat messages array.

    Uses the cl100k_base overhead estimation (4 tokens per message).
    """
    enc = _get_encoding(encoding_name)
    total = 0
    for msg in messages:
        total += 4  # message overhead
        for value in msg.values():
            if isinstance(value, str):
                total += len(enc.encode(value, disallowed_special=()))
    total += 2  # reply priming
    return total


def count_text_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens for a plain text string — used for output counting fallback."""
    enc = _get_encoding(encoding_name)
    return len(enc.encode(text, disallowed_special=()))
