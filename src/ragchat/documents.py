"""Load PDFs page by page and split pages into chunks that keep their source."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

from .pdftext import page_text
from .text import clean_page_text


@dataclass(frozen=True)
class Page:
    source: str  # file name shown to the user
    page: int  # 1-based page number
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_pdf(data: bytes, source: str) -> list[Page]:
    """Extract text from every page. Pages with no text layer are skipped."""
    pages = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for number, page in enumerate(doc, start=1):
            text = clean_page_text(page_text(page))
            if text:
                pages.append(Page(source=source, page=number, text=text))
    return pages


def load_pdf_file(path: str | Path) -> list[Page]:
    path = Path(path)
    return load_pdf(path.read_bytes(), path.name)


def split_page(page: Page, max_chars: int, overlap: int) -> list[Chunk]:
    """Split one page into overlapping chunks.

    Chunks never cross a page boundary, so every chunk cites exactly one page.
    Splits prefer paragraph breaks, then line breaks, then sentence ends, then spaces.
    """
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    text = page.text
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            end = _best_break(text, start, end)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        # Start the overlap on a word boundary instead of mid-word.
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start

    return [
        Chunk(id=f"{page.source}#p{page.page}-{i}", source=page.source, page=page.page, text=piece)
        for i, piece in enumerate(pieces)
    ]


def _best_break(text: str, start: int, end: int) -> int:
    floor = start + (end - start) // 2  # never cut a chunk to less than half its size
    for sep in ("\n\n", "\n", ". ", "۔", "? ", "؟", " "):
        cut = text.rfind(sep, floor, end)
        if cut != -1:
            return cut + len(sep)
    return end


def chunk_pages(pages: list[Page], max_chars: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(split_page(page, max_chars, overlap))
    return chunks
