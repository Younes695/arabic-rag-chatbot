from ragchat.text import clean_page_text, normalize_for_search, tokenize


def test_arabic_letter_variants_normalize_to_one_form():
    assert normalize_for_search("أحمد") == normalize_for_search("احمد") == normalize_for_search("إحمد")
    assert normalize_for_search("مستشفى") == normalize_for_search("مستشفي")
    assert normalize_for_search("مدرسة") == normalize_for_search("مدرسه")


def test_diacritics_and_tatweel_are_removed():
    assert normalize_for_search("الْعَقْـــد") == normalize_for_search("العقد")


def test_eastern_arabic_digits_match_ascii():
    assert tokenize("سنة ٢٠٢٤") == tokenize("سنة 2024")


def test_definite_article_is_stripped_for_long_words():
    assert tokenize("الشركة") == tokenize("شركة")
    # Short words keep their letters: stripping "ال" from "الم" would leave one letter.
    assert tokenize("الم") == ["الم"]


def test_stopwords_are_dropped_in_both_languages():
    assert tokenize("the duration of the contract") == ["duration", "contract"]
    assert tokenize("ما هي مدة العقد") == ["هي", "مده", "عقد"]


def test_clean_page_text_collapses_spaces_and_blank_lines():
    raw = "Title  \n\n\n\n  first   line\t here \n"
    assert clean_page_text(raw) == "Title\n\nfirst line here"
