from conftest import FakeGenerator

from ragchat.answer import NOT_FOUND, NOT_FOUND_MESSAGE, answer_question, build_prompt, cited_numbers
from ragchat.documents import Chunk
from ragchat.index import Index

CHUNKS = [
    Chunk("c0", "contract.pdf", 1, "The contract duration is 24 months from the signing date."),
    Chunk("c1", "contract.pdf", 2, "Payment is made within 30 days of each invoice."),
    Chunk("c2", "contract.pdf", 5, "The warranty period covers 12 months after handover."),
    Chunk("c3", "policy_ar.pdf", 3, "مدة الإجازة السنوية للموظف ٢١ يوما مدفوعة الأجر."),
    Chunk("c4", "policy_ar.pdf", 4, "يتم صرف المرتبات في آخر يوم عمل من كل شهر."),
]


def test_keyword_search_finds_english_and_arabic(embedder):
    index = Index.build(CHUNKS, embedder=None)
    assert index.search("How long is the warranty period?", k=1)[0].chunk.id == "c2"
    # Different letter forms and digits than the source text still match.
    assert index.search("كام يوم الاجازه السنويه؟", k=1)[0].chunk.id == "c3"


def test_hybrid_search_uses_vectors_when_available(embedder):
    index = Index.build(CHUNKS, embedder)
    assert index.has_vectors
    hits = index.search("payment invoice days", k=2, embedder=embedder)
    assert hits[0].chunk.id == "c1"
    assert hits[0].score >= hits[1].score


def test_search_on_empty_index_returns_nothing():
    assert Index.build([], embedder=None).search("anything", k=3) == []


def test_index_round_trips_through_disk(tmp_path, embedder):
    index = Index.build(CHUNKS, embedder)
    index.save(tmp_path)
    loaded = Index.load(tmp_path)
    assert loaded.chunks == CHUNKS
    assert loaded.has_vectors
    assert loaded.search("warranty", k=1, embedder=embedder)[0].chunk.id == "c2"


def test_prompt_numbers_sources_with_file_and_page(embedder):
    hits = Index.build(CHUNKS, embedder=None).search("warranty", k=1)
    prompt = build_prompt("How long is the warranty?", hits)
    assert "[1] (contract.pdf, page 5)" in prompt
    assert prompt.endswith("Write the answer in English.\nQuestion: How long is the warranty?")
    assert "Write the answer in Arabic." in build_prompt("مدة الضمان كام؟", hits)


def test_cited_numbers_handles_lists_arabic_commas_and_invalid_numbers():
    assert cited_numbers("A [2]. B [1, 3]. C [2][9] D [3،1]", max_number=3) == [2, 1, 3]


def test_answer_maps_citations_to_file_and_page():
    index = Index.build(CHUNKS, embedder=None)
    generator = FakeGenerator("The warranty is 12 months after handover [1].")
    answer = answer_question("How long is the warranty?", index, generator, k=3)

    assert answer.found and answer.grounded
    assert [(s.number, s.source, s.page) for s in answer.sources] == [(1, "contract.pdf", 5)]
    system, prompt = generator.calls[0]
    assert NOT_FOUND in system
    assert "12 months after handover" in prompt


def test_not_found_reply_becomes_message_in_question_language():
    index = Index.build(CHUNKS, embedder=None)
    answer = answer_question("مدة الإجازة كام يوم؟", index, FakeGenerator(NOT_FOUND))
    assert not answer.found
    assert answer.text == NOT_FOUND_MESSAGE["ar"]
    assert answer.retrieved  # the sources it looked at are still shown


def test_question_with_no_matches_skips_the_model():
    index = Index.build(CHUNKS, embedder=None)
    generator = FakeGenerator("should not be called")
    answer = answer_question("zebra giraffe", index, generator)
    assert not answer.found
    assert answer.text == NOT_FOUND_MESSAGE["en"]
    assert generator.calls == []


def test_answer_without_citations_is_not_grounded():
    index = Index.build(CHUNKS, embedder=None)
    answer = answer_question("warranty?", index, FakeGenerator("It is one year."))
    assert answer.found and not answer.grounded
