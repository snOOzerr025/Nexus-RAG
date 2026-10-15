"""Token counting utilities using tiktoken.

Provides accurate token counting for OpenAI models using the cl100k_base
encoding (used by text-embedding-3-small and GPT-4o).
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _get_encoding() -> tiktoken.Encoding:
    """Get the cl100k_base encoding (cached singleton).

    cl100k_base is used by:
    - text-embedding-3-small / text-embedding-3-large
    - gpt-4o / gpt-4o-mini
    - gpt-4-turbo / gpt-3.5-turbo
    """
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string.

    Args:
        text: The text to tokenize and count.

    Returns:
        Number of tokens in the text.
    """
    encoding = _get_encoding()
    return len(encoding.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to a maximum number of tokens.

    Args:
        text: The text to truncate.
        max_tokens: Maximum number of tokens to keep.

    Returns:
        The truncated text, decoded back from token boundaries.
    """
    encoding = _get_encoding()
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def split_by_tokens(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks of a specified token size with overlap.

    Uses a sliding window approach over the token sequence. Each chunk
    is exactly chunk_size tokens (except possibly the last chunk), with
    chunk_overlap tokens shared between consecutive chunks.

    Args:
        text: The text to split.
        chunk_size: Number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between chunks.

    Returns:
        List of text chunks, each decoded from token boundaries.

    Raises:
        ValueError: If chunk_overlap >= chunk_size.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})"
        )

    encoding = _get_encoding()
    tokens = encoding.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        if end >= len(tokens):
            break
        start += step

    return chunks
