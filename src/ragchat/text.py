"""Text normalization and tokenization for Arabic and English keyword search."""

from __future__ import annotations

import re
import unicodedata

# Arabic diacritics (tashkeel), Quranic marks and tatweel carry no meaning for search.
_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_ALEF = re.compile("[آأإٱ]")  # آ أ إ ٱ -> ا
_TOKEN = re.compile(r"\w+", re.UNICODE)
_SPACES = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")

# Eastern Arabic and Persian digits -> ASCII, so "٢٠٢٤" matches "2024".
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_STOPWORDS = frozenset(
    """
    the a an of to in on for and or is are was were be by with as at from that this
    it its into than then there their what which who whom how when where why do does
    في من على إلى الى عن مع هذا هذه ذلك تلك التي الذي الذين ما ماذا هل او أو و ثم
    كان كانت يكون لا لم لن قد كل بعد قبل عند حتى اذا إذا ان أن إن
    """.split()
)


def clean_page_text(text: str) -> str:
    """Tidy raw PDF text without changing its meaning (used for display and chunking)."""
    text = unicodedata.normalize("NFKC", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANK_LINES.sub("\n\n", text).strip()


def normalize_for_search(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().translate(_DIGITS)
    text = _DIACRITICS.sub("", text)
    text = _ALEF.sub("ا", text)
    text = text.replace("ى", "ي")  # ى -> ي
    text = text.replace("ة", "ه")  # ة -> ه
    return text


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN.findall(normalize_for_search(text)):
        # Strip the Arabic definite article so "العقد" and "عقد" match.
        if len(tok) > 4 and tok.startswith("ال"):
            tok = tok[2:]
        if tok not in _STOPWORDS:
            tokens.append(tok)
    return tokens
