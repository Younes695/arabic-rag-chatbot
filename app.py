"""Streamlit UI: upload PDFs, ask questions, get answers with file + page citations."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ragchat.answer import Answer, answer_question, detect_language  # noqa: E402
from ragchat.config import Settings  # noqa: E402
from ragchat.documents import chunk_pages, load_pdf  # noqa: E402
from ragchat.index import Index  # noqa: E402
from ragchat.llm import GeminiEmbedder, GeminiGenerator, make_client  # noqa: E402

st.set_page_config(page_title="Ask Your Documents", page_icon="📄", layout="wide")
settings = Settings.from_env()
SAMPLES = Path(__file__).resolve().parent / "samples"

# Each paragraph takes its direction from its own text, so Arabic reads right-to-left.
st.markdown(
    "<style>:is([data-testid='stMarkdownContainer'], [data-testid='stCaptionContainer']) :is(p, li, h1, h2, h3)"
    " {unicode-bidi: plaintext; text-align: start;}</style>",
    unsafe_allow_html=True,
)

UI = {
    "en": {"sources": "Sources", "page": "page", "unverified": "The answer cites no source; verify it before relying on it."},
    "ar": {"sources": "المصادر", "page": "صفحة", "unverified": "الإجابة دي مفيهاش مصدر، اتأكد منها قبل ما تعتمد عليها."},
}


@st.cache_resource(show_spinner=False)
def gemini(api_key: str):
    client = make_client(api_key)
    return (
        GeminiEmbedder(client, settings.embed_model, settings.embed_dim),
        GeminiGenerator(client, settings.chat_model),
    )


@st.cache_resource(show_spinner=False, max_entries=8)
def build_index(files: tuple[tuple[str, bytes], ...], api_key: str | None) -> tuple[Index, list[str]]:
    pages, skipped = [], []
    for name, data in files:
        file_pages = load_pdf(data, name)
        if file_pages:
            pages.extend(file_pages)
        else:
            skipped.append(name)
    chunks = chunk_pages(pages, settings.chunk_chars, settings.chunk_overlap)
    embedder = gemini(api_key)[0] if api_key else None
    return Index.build(chunks, embedder), skipped


def render_answer(answer: Answer, question: str) -> None:
    ui = UI[detect_language(question)]
    st.markdown(answer.text)
    if answer.found and not answer.grounded:
        st.warning(ui["unverified"])
    if answer.sources:
        with st.expander(f"{ui['sources']} ({len(answer.sources)})"):
            for s in answer.sources:
                # Page label first: an Arabic label then sets the line's direction, so
                # "صفحة 3" doesn't render reversed next to a Latin file name.
                st.markdown(f"**[{s.number}] {ui['page']} {s.page} · {s.source}**")
                st.caption(s.text)


def render_passages(question: str, index: Index) -> None:
    """Keyword-only mode (no API key): show the best matching passages."""
    hits = index.search(question, k=3)
    if not hits:
        st.markdown("No matching passages.")
    for hit in hits:
        st.markdown(f"**{hit.chunk.source}, page {hit.chunk.page}**")
        st.caption(hit.chunk.text)


with st.sidebar:
    st.header("📄 Documents")
    uploads = st.file_uploader("Upload PDFs (Arabic or English)", type="pdf", accept_multiple_files=True)
    use_samples = not uploads and st.toggle(
        "Use the sample documents",
        value=True,
        help="An Arabic HR policy and an English service agreement for a fictional company.",
    )
    api_key = settings.api_key or st.text_input(
        "Gemini API key", type="password", help="Free key from aistudio.google.com. Not stored."
    )
    if not api_key:
        st.info("No API key: keyword search only, without generated answers.")
    st.caption(f"Model: {settings.chat_model}")

st.title("Ask Your Documents")
st.caption("Answers come only from your PDFs, and every fact links to its file and page.")

if uploads:
    files = tuple(sorted((f.name, f.getvalue()) for f in uploads))
elif use_samples:
    files = tuple((p.name, p.read_bytes()) for p in sorted(SAMPLES.glob("*.pdf")))
    st.caption("Try: «كام يوم إجازة سنوية؟» or «What is the late payment interest?»")
else:
    st.info("Upload one or more PDFs from the sidebar to start.")
    st.stop()
with st.spinner("Reading and indexing documents..."):
    try:
        index, skipped = build_index(files, api_key or None)
    except Exception as exc:  # surface API/key problems in the UI instead of a stack trace
        st.error(f"Could not index the documents: {exc}")
        st.stop()

if skipped:
    st.warning("No text layer (scanned PDF?), skipped: " + ", ".join(skipped))
st.sidebar.success(f"{len(index.chunks)} passages from {len(files) - len(skipped)} file(s)")

# Reset the conversation whenever the document set changes.
doc_key = hashlib.sha256(b"".join(n.encode() + hashlib.sha256(d).digest() for n, d in files)).hexdigest()
if st.session_state.get("doc_key") != doc_key:
    st.session_state.doc_key = doc_key
    st.session_state.history = []

for question, answer in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        render_answer(answer, question)

if question := st.chat_input("Ask a question about your documents"):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not api_key:
            render_passages(question, index)
        else:
            embedder, generator = gemini(api_key)
            with st.spinner("Thinking..."):
                try:
                    answer = answer_question(question, index, generator, embedder, k=settings.top_k)
                except Exception as exc:
                    st.error(f"Gemini request failed: {exc}")
                    st.stop()
            render_answer(answer, question)
            st.session_state.history.append((question, answer))
