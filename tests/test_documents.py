import pymupdf
import pytest

from ragchat.documents import Page, chunk_pages, load_pdf, split_page


def _pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_load_pdf_keeps_page_numbers_and_skips_empty_pages():
    pages = load_pdf(_pdf(["Payment terms are 30 days.", "", "Warranty is 12 months."]), "contract.pdf")
    assert [(p.source, p.page) for p in pages] == [("contract.pdf", 1), ("contract.pdf", 3)]
    assert "30 days" in pages[0].text
    assert "12 months" in pages[1].text


def test_short_page_is_one_chunk():
    chunks = split_page(Page("a.pdf", 4, "Short page."), max_chars=100, overlap=20)
    assert len(chunks) == 1
    assert (chunks[0].source, chunks[0].page, chunks[0].text) == ("a.pdf", 4, "Short page.")


def test_long_page_splits_with_overlap_and_no_lost_words():
    words = [f"w{i}" for i in range(400)]
    page = Page("a.pdf", 2, " ".join(words))
    chunks = split_page(page, max_chars=200, overlap=40)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.page == 2 for c in chunks)
    covered = set(" ".join(c.text for c in chunks).split())
    assert covered == set(words)
    # Consecutive chunks share text (the overlap).
    assert set(chunks[0].text.split()) & set(chunks[1].text.split())
    # No chunk starts mid-word.
    assert all(c.text.split()[0] in words for c in chunks)


def test_split_prefers_paragraph_breaks():
    first = "A" * 120
    second = "B" * 120
    chunks = split_page(Page("a.pdf", 1, f"{first}\n\n{second}"), max_chars=200, overlap=10)
    assert chunks[0].text == first


def test_chunk_ids_are_unique_across_pages():
    pages = [Page("a.pdf", n, "word " * 300) for n in (1, 2)]
    chunks = chunk_pages(pages, max_chars=300, overlap=50)
    assert len({c.id for c in chunks}) == len(chunks)


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        split_page(Page("a.pdf", 1, "text"), max_chars=100, overlap=100)
