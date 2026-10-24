"""Turn retrieved chunks into a grounded, cited answer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .index import Hit, Index
from .llm import Embedder, Generator

NOT_FOUND = "NOT_FOUND"

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered sources you are given.

Rules:
- Use only facts stated in the sources. Never use outside knowledge, never guess.
- After every sentence that states a fact, cite its source number in square brackets, e.g. [2] or [1][3].
- Answer in the same language as the question (Arabic question -> Arabic answer).
- Be concise. Quote exact numbers, dates and names as written in the sources.
- If the sources do not contain the answer, reply with exactly {NOT_FOUND} and nothing else.
- If the sources answer only part of the question, answer that part and say what is missing."""

NOT_FOUND_MESSAGE = {
    "ar": "مش لاقي إجابة السؤال ده في المستندات اللي اترفعت.",
    "en": "I couldn't find the answer to this question in the uploaded documents.",
}

_CITATION = re.compile(r"\[(\d+(?:\s*[,،]\s*\d+)*)\]")
_ARABIC_LETTER = re.compile("[؀-ۿ]")


@dataclass(frozen=True)
class Source:
    number: int
    source: str
    page: int
    text: str


@dataclass
class Answer:
    text: str
    found: bool
    sources: list[Source] = field(default_factory=list)  # only sources the answer cites
    retrieved: list[Hit] = field(default_factory=list)  # everything that was sent to the model

    @property
    def grounded(self) -> bool:
        """A found answer must cite at least one real source to be trusted."""
        return self.found and bool(self.sources)


def detect_language(text: str) -> str:
    return "ar" if _ARABIC_LETTER.search(text) else "en"


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = [
        f"[{n}] ({hit.chunk.source}, page {hit.chunk.page})\n{hit.chunk.text}"
        for n, hit in enumerate(hits, start=1)
    ]
    # Stated explicitly: with Arabic sources, smaller models drift into Arabic
    # even for English questions unless told the answer language.
    language = "Arabic" if detect_language(question) == "ar" else "English"
    return (
        "Sources:\n\n"
        + "\n\n".join(blocks)
        + f"\n\nWrite the answer in {language}.\nQuestion: {question}"
    )


def cited_numbers(answer_text: str, max_number: int) -> list[int]:
    """Source numbers cited in the answer, in first-seen order, ignoring invalid ones."""
    seen: list[int] = []
    for group in _CITATION.findall(answer_text):
        for raw in re.split(r"[,،]", group):
            n = int(raw.strip())
            if 1 <= n <= max_number and n not in seen:
                seen.append(n)
    return seen


def answer_question(
    question: str,
    index: Index,
    generator: Generator,
    embedder: Embedder | None = None,
    k: int = 6,
) -> Answer:
    language = detect_language(question)
    hits = index.search(question, k=k, embedder=embedder)
    if not hits:
        return Answer(NOT_FOUND_MESSAGE[language], found=False)

    raw = generator.generate(SYSTEM_PROMPT, build_prompt(question, hits))
    if not raw or raw.strip().strip(".").upper() == NOT_FOUND:
        return Answer(NOT_FOUND_MESSAGE[language], found=False, retrieved=hits)

    sources = [
        Source(n, hits[n - 1].chunk.source, hits[n - 1].chunk.page, hits[n - 1].chunk.text)
        for n in cited_numbers(raw, len(hits))
    ]
    return Answer(raw.replace(NOT_FOUND, "").strip(), found=True, sources=sources, retrieved=hits)
