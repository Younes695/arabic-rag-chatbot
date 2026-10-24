"""Thin wrappers around the Gemini API: embeddings and answer generation.

The rest of the code depends only on the `Embedder` and `Generator` protocols,
so tests and the offline keyword-only mode run without an API key.
"""

from __future__ import annotations

import time
from typing import Callable, Protocol, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class Generator(Protocol):
    def generate(self, system: str, prompt: str) -> str: ...


def _with_retry(call: Callable[[], T], attempts: int = 3, base_delay: float = 2.0) -> T:
    """Retry rate-limit and transient server errors with exponential backoff."""
    from google.genai import errors

    for attempt in range(attempts):
        try:
            return call()
        except errors.APIError as exc:
            retryable = exc.code == 429 or (exc.code or 0) >= 500
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(base_delay * 2**attempt)
    raise AssertionError("unreachable")


class GeminiEmbedder:
    # The API accepts at most 100 texts per embedding request.
    batch_size = 100

    def __init__(self, client, model: str, dim: int):
        self.client = client
        self.model = model
        self.dim = dim

    def _embed(self, texts: Sequence[str], task_type: str) -> np.ndarray:
        from google.genai import types

        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            response = _with_retry(
                lambda: self.client.models.embed_content(model=self.model, contents=batch, config=config)
            )
            vectors.extend(e.values for e in response.embeddings)
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class GeminiGenerator:
    def __init__(self, client, model: str, temperature: float = 0.1):
        self.client = client
        self.model = model
        self.temperature = temperature

    def generate(self, system: str, prompt: str) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(system_instruction=system, temperature=self.temperature)
        response = _with_retry(
            lambda: self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        )
        return (response.text or "").strip()


def make_client(api_key: str, timeout_s: float = 60):
    """Gemini client with a per-request timeout, so a stalled request fails with an
    error the UI can show instead of hanging the app."""
    from google import genai
    from google.genai import types

    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
