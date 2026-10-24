import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragchat.text import tokenize  # noqa: E402


class FakeEmbedder:
    """Deterministic bag-of-words embedder: shared tokens -> similar vectors."""

    dim = 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[hash(tok) % self.dim] += 1.0
        return v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text):
        return self._vec(text)


class FakeGenerator:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        return self.reply


@pytest.fixture
def embedder():
    return FakeEmbedder()
