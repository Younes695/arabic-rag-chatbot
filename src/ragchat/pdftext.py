"""Bidi-aware PDF text extraction.

Plain text extraction follows the order glyphs were written into the PDF. For
Arabic documents exported from common editors (e.g. Word) that order is wrong in
two ways that break search and answers:

* digit runs inside right-to-left text come out reversed ("21 days" -> "12 days");
* the lam-alef ligature comes out as alef + lam ("الإجازة" -> "اإلجازة").

Both are fixed by rebuilding every line from character positions: sort glyphs
left-to-right by x (their visual order), then convert visual order to logical
order with a simplified bidi rule. Positions don't depend on how the PDF was
written, so this works for any producer, and left-to-right documents come out
exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf


@dataclass(frozen=True)
class Glyph:
    char: str
    x0: float
    x1: float
    baseline: float
    size: float


def _is_rtl(ch: str) -> bool:
    cp = ord(ch)
    if 0x0660 <= cp <= 0x0669 or 0x06F0 <= cp <= 0x06F9:  # Arabic-Indic digits are numbers
        return False
    return (
        0x0590 <= cp <= 0x08FF  # Hebrew, Arabic, Syriac, Arabic Supplement/Extended
        or 0xFB1D <= cp <= 0xFDFF  # Hebrew + Arabic presentation forms A
        or 0xFE70 <= cp <= 0xFEFF  # Arabic presentation forms B
    ) and ch.isalpha()


def _is_ltr(ch: str) -> bool:
    return ch.isalnum() and not _is_rtl(ch)


def is_rtl_text(text: str) -> bool:
    """Paragraph direction: RTL when it has more RTL letters than Latin letters.

    Only letters vote. Digits are direction-neutral, so a line like
    "USD 12,000" inside an Arabic paragraph doesn't flip it to LTR.
    """
    rtl = sum(map(_is_rtl, text))
    ltr = sum(1 for ch in text if ch.isalpha() and not _is_rtl(ch))
    return rtl > ltr


def visual_to_logical(visual: str, rtl: bool | None = None) -> str:
    """Convert one line from visual (left-to-right) order to logical order.

    In an RTL paragraph the whole line is reversed, then every run of LTR text
    (Latin words, numbers, and the punctuation/spaces between them, e.g.
    "USD 12,000" or "1.5%") is reversed back so it reads correctly. In an LTR
    paragraph only embedded Arabic runs are reversed. `rtl` is the paragraph
    direction; when omitted it is decided from this line alone.
    """
    if rtl is None:
        rtl = is_rtl_text(visual)
    if not rtl:
        return _reverse_runs(visual, _is_rtl)
    return _reverse_runs(visual[::-1], _is_ltr)


def _reverse_runs(text: str, strong) -> str:
    """Reverse each maximal run that starts and ends with a `strong` character."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        if not strong(text[i]):
            i += 1
            continue
        end = i
        j = i + 1
        while j < n and (strong(text[j]) or not (_is_rtl(text[j]) or _is_ltr(text[j]))):
            if strong(text[j]):
                end = j
            j += 1
        out[i : end + 1] = text[i : end + 1][::-1]
        i = end + 1
    return "".join(out)


_ALEFS = frozenset("اأإآ")  # ا أ إ آ
_LAM = "ل"


def _visual_key(glyphs: list[Glyph]) -> list[float]:
    """Left-to-right sort key for each glyph.

    A lam-alef ligature is drawn as one shape but mapped to two characters: the
    lam gets the full width and the alef a zero-width box on the lam's edge.
    Placing that alef just left of its lam makes it come out after the lam once
    the RTL line is reversed (logical "لإ", not "إل").
    """
    keys = [g.x0 for g in glyphs]
    lams = [g for g in glyphs if g.char == _LAM]
    for i, g in enumerate(glyphs):
        if g.char in _ALEFS and g.x1 - g.x0 < 0.5:
            # The alef box sits on its own lam's right edge, which can also be the
            # left edge of the previous letter (another lam in "اللا"): match x1 first.
            owner = min(lams, key=lambda lam: abs(lam.x1 - g.x0), default=None)
            if owner is None or abs(owner.x1 - g.x0) > 0.5:
                owner = next((lam for lam in lams if lam.x0 - 0.5 <= g.x0 <= lam.x1 + 0.5), None)
            if owner is not None:
                keys[i] = owner.x0 - 0.001
    return keys


# Gap between two glyphs, as a fraction of the font size, that counts as a word break.
WORD_GAP = 0.2


def _line_text(glyphs: list[Glyph], rtl: bool) -> str:
    """Rebuild one visual line of RTL-containing text in logical order.

    Space glyphs are ignored and word breaks come from the gaps between letters:
    producers place space glyphs inconsistently (on top of letters, or spanning
    several of them), while the letters' own positions are what the reader sees.
    """
    letters = [g for g in glyphs if not g.char.isspace()]
    keys = _visual_key(letters)
    parts: list[str] = []
    prev_x1: float | None = None
    for key, g in sorted(zip(keys, letters), key=lambda pair: pair[0]):
        # A moved ligature alef sits where its lam starts, so measure gaps from there.
        x0, x1 = key, key + (g.x1 - g.x0)
        if prev_x1 is not None and x0 - prev_x1 > WORD_GAP * g.size:
            parts.append(" ")
        parts.append(g.char)
        prev_x1 = max(x1, prev_x1) if prev_x1 is not None else x1
    return visual_to_logical("".join(parts), rtl).strip()


def _group_lines(glyphs: list[Glyph]) -> list[list[Glyph]]:
    """Group glyphs whose baselines are within half a font size into one visual line."""
    lines: list[list[Glyph]] = []
    for g in sorted(glyphs, key=lambda g: g.baseline):
        if lines and abs(lines[-1][0].baseline - g.baseline) <= 0.5 * g.size:
            lines[-1].append(g)
        else:
            lines.append([g])
    return lines


def page_text(page: pymupdf.Page) -> str:
    """Extract a page's text in logical reading order, block by block."""
    blocks_out: list[str] = []
    # Ligatures stay single characters here: expanding them would order their parts
    # left-to-right inside RTL words. NFKC normalization splits them later, correctly.
    raw = page.get_text("rawdict")
    for block in raw["blocks"]:
        lines_raw = block.get("lines", [])
        plain = "\n".join("".join(c["c"] for s in line["spans"] for c in s["chars"]) for line in lines_raw)
        if not any(map(_is_rtl, plain)):
            # No right-to-left text: keep PyMuPDF's own reading order untouched.
            text = "\n".join(line.strip() for line in plain.splitlines() if line.strip())
        else:
            glyphs = [
                Glyph(c["c"], c["bbox"][0], c["bbox"][2], c["origin"][1], span["size"])
                for line in lines_raw
                if abs(line["dir"][1]) < 1e-3  # horizontal text only
                for span in line["spans"]
                for c in span["chars"]
            ]
            # Direction is decided per block (paragraph), not per line: a wrapped
            # line holding only a number belongs to its Arabic paragraph.
            rtl = is_rtl_text(plain)
            lines = [_line_text(group, rtl) for group in _group_lines(glyphs)]
            text = "\n".join(line for line in lines if line)
        if text:
            blocks_out.append(text)
    return "\n\n".join(blocks_out)
