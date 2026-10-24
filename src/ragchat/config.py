"""Runtime settings, read from environment variables (and a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    chat_model: str
    embed_model: str
    embed_dim: int
    chunk_chars: int
    chunk_overlap: int
    top_k: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            chat_model=os.getenv("RAG_CHAT_MODEL", "gemini-3.5-flash-lite"),
            embed_model=os.getenv("RAG_EMBED_MODEL", "gemini-embedding-001"),
            embed_dim=int(os.getenv("RAG_EMBED_DIM", "768")),
            chunk_chars=int(os.getenv("RAG_CHUNK_CHARS", "900")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "150")),
            top_k=int(os.getenv("RAG_TOP_K", "6")),
        )
