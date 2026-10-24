"""Evaluate retrieval and answers on the sample documents.

    python eval/run_eval.py --retrieval-only   # no API key needed (keyword search)
    python eval/run_eval.py                    # full run: hybrid search + Gemini answers

Metrics
- hit@k / MRR: the page holding the answer is among the top-k retrieved passages.
- answer accuracy: every expected keyword appears in the answer (normalized, so
  Arabic-Indic digits and letter variants match).
- citation accuracy: the answer cites the page that holds the answer.
- refusal accuracy: questions the documents can't answer get "not found".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragchat.answer import answer_question  # noqa: E402
from ragchat.config import Settings  # noqa: E402
from ragchat.documents import chunk_pages, load_pdf_file  # noqa: E402
from ragchat.index import Index  # noqa: E402
from ragchat.llm import GeminiEmbedder, GeminiGenerator, make_client  # noqa: E402
from ragchat.text import normalize_for_search  # noqa: E402


def load_questions() -> list[dict]:
    lines = (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def contains_all(answer: str, keywords: list[str]) -> bool:
    text = normalize_for_search(answer)
    return all(normalize_for_search(k) in text for k in keywords)


def pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true", help="skip Gemini answers")
    parser.add_argument("--keyword-only", action="store_true", help="BM25 search, no embeddings")
    parser.add_argument("--delay", type=float, default=4.0, help="seconds between Gemini calls")
    args = parser.parse_args()

    settings = Settings.from_env()
    use_api = not (args.retrieval_only and args.keyword_only)
    if use_api and not settings.api_key:
        sys.exit("GEMINI_API_KEY is not set. Use --retrieval-only --keyword-only to run offline.")

    client = make_client(settings.api_key) if use_api else None
    embedder = None if args.keyword_only else GeminiEmbedder(client, settings.embed_model, settings.embed_dim)
    generator = None if args.retrieval_only else GeminiGenerator(client, settings.chat_model)

    pages = [p for pdf in sorted((ROOT / "samples").glob("*.pdf")) for p in load_pdf_file(pdf)]
    index = Index.build(chunk_pages(pages, settings.chunk_chars, settings.chunk_overlap), embedder)
    k = settings.top_k

    rows = []
    for q in load_questions():
        answerable = q.get("answerable", True)
        hits = index.search(q["question"], k=k, embedder=embedder)
        rank = next(
            (i for i, h in enumerate(hits, 1) if (h.chunk.source, h.chunk.page) == (q.get("source"), q.get("page"))),
            None,
        )
        row = {"id": q["id"], "question": q["question"], "answerable": answerable, "rank": rank}
        if generator is not None:
            answer = answer_question(q["question"], index, generator, embedder, k=k)
            row.update(answer=answer.text, found=answer.found, cited=[[s.source, s.page] for s in answer.sources])
            if answerable:
                row["correct"] = answer.found and contains_all(answer.text, q["keywords"])
                row["cited_right_page"] = [q["source"], q["page"]] in row["cited"]
            else:
                row["refused"] = not answer.found
            time.sleep(args.delay)
        rows.append(row)
        print(f"{q['id']:5} rank={rank} {row.get('correct', row.get('refused', ''))}", flush=True)

    answerable_rows = [r for r in rows if r["answerable"]]
    unanswerable_rows = [r for r in rows if not r["answerable"]]
    mode = "keyword (BM25)" if args.keyword_only else "hybrid (BM25 + embeddings)"
    summary = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "search": mode,
        "chat_model": None if generator is None else settings.chat_model,
        "embed_model": None if embedder is None else settings.embed_model,
        "k": k,
        "hit@1": pct(sum(r["rank"] == 1 for r in answerable_rows), len(answerable_rows)),
        f"hit@{k}": pct(sum(r["rank"] is not None for r in answerable_rows), len(answerable_rows)),
        "MRR": round(sum(1 / r["rank"] for r in answerable_rows if r["rank"]) / len(answerable_rows), 3),
    }
    if generator is not None:
        summary["answer accuracy"] = pct(sum(r["correct"] for r in answerable_rows), len(answerable_rows))
        summary["citation accuracy"] = pct(sum(r["cited_right_page"] for r in answerable_rows), len(answerable_rows))
        summary["refusal accuracy"] = pct(sum(r["refused"] for r in unanswerable_rows), len(unanswerable_rows))

    out = ROOT / "eval" / "results"
    out.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / f"{stamp}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n| metric | value |\n|---|---|")
    for key, value in summary.items():
        print(f"| {key} | {value} |")


if __name__ == "__main__":
    main()
