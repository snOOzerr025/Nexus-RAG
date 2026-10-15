"""OpenAI API client wrapper.

Handles embedding generation, chat completions, retry logic with
exponential backoff, and batch processing. Reuses a single httpx
session for connection pooling.
"""

from __future__ import annotations

import openai
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIClient:
    """Wrapper around the OpenAI API for embeddings and chat completions.

    Features:
    - Automatic retry with exponential backoff for rate limits and timeouts
    - Batch embedding for efficiency (up to 100 texts per call)
    - Connection pooling via shared httpx client
    - Structured logging for all API calls
    """

    def __init__(
        self,
        api_key: str,
        embedding_model: str = "text-embedding-3-small",
        generation_model: str = "gpt-4o",
        embedding_dimensions: int = 1536,
        max_retries: int = 3,
        base_url: str | None = None,
    ) -> None:
        """Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key.
            embedding_model: Model for generating embeddings.
            generation_model: Model for chat completions.
            embedding_dimensions: Embedding vector dimensions.
            max_retries: Maximum retry attempts for failed calls.
            base_url: Optional custom API base URL (e.g. Groq endpoint).
        """
        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**client_kwargs)
        self._embedding_model = embedding_model
        self._generation_model = generation_model
        self._embedding_dimensions = embedding_dimensions
        self._max_retries = max_retries

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, automatically batching large requests.

        Splits into batches of 100 texts each (OpenAI batch limit).

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (same order as input texts).
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

            logger.debug(
                "embedding_batch_complete",
                batch_start=i,
                batch_size=len(batch),
                total=len(texts),
            )

        return all_embeddings

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (alias for embed_texts).

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        return await self.embed_texts(texts)

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector (list of floats).
        """
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def get_embedding(self, text: str) -> list[float]:
        """Alias for embed_single."""
        return await self.embed_single(text)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            )
        ),
        before_sleep=lambda retry_state: logger.warning(
            "openai_retry",
            attempt=retry_state.attempt_number,
            error=str(retry_state.outcome.exception()) if retry_state.outcome else "unknown",
        ),
    )
    async def _embed_batch_with_retry(
        self, texts: list[str]
    ) -> list[list[float]]:
        """Embed a batch with retry logic for transient errors.

        Args:
            texts: Batch of texts (max 100).

        Returns:
            List of embedding vectors.
        """
        response = await self._client.embeddings.create(
            model=self._embedding_model,
            input=texts,
            dimensions=self._embedding_dimensions,
        )
        # Sort by index to ensure order matches input
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            )
        ),
        before_sleep=lambda retry_state: logger.warning(
            "openai_chat_retry",
            attempt=retry_state.attempt_number,
            error=str(retry_state.outcome.exception()) if retry_state.outcome else "unknown",
        ),
    )
    async def chat_completion(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        response_format: dict | None = None,
    ) -> str:
        """Generate a chat completion.

        Args:
            messages: List of message dicts (role, content).
            model: Override the default generation model.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in response.
            response_format: Optional response format specification.

        Returns:
            The generated text content.
        """
        kwargs: dict = {
            "model": model or self._generation_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = await self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""

        logger.debug(
            "chat_completion",
            model=kwargs["model"],
            input_messages=len(messages),
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )

        return content

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
