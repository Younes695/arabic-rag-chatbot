from pathlib import Path

import pymupdf

from ragchat.documents import load_pdf_file
from ragchat.pdftext import page_text, visual_to_logical

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def test_rtl_line_is_reversed_but_numbers_and_latin_keep_their_order():
    # Visual (left-to-right) order of the logical line "مدتها 21 يوم".
    assert visual_to_logical("موي 21 اهتدم") == "مدتها 21 يوم"
    assert visual_to_logical("ليم USD 12,000 غلبم") == "مبلغ USD 12,000 ميل"


def test_ltr_line_only_reverses_embedded_arabic():
    assert visual_to_logical("Delivery to رصم by June") == "Delivery to مصر by June"


def test_pure_latin_line_is_unchanged():
    assert visual_to_logical("Payment within 30 days.") == "Payment within 30 days."


def test_word_exported_arabic_pdf_keeps_numbers_and_lam_alef():
    # The sample is exported by Microsoft Word; plain extraction of this page reads
    # "21" as "12" and "الإجازات" as "اإلجازات".
    text = " ".join(p.text for p in load_pdf_file(SAMPLES / "hr_policy_ar.pdf"))
    assert "مدتها ٢١ يوم عمل" in text
    assert "إلى ٣٠ يوم عمل" in text
    assert "الإجازات السنوية" in text
    assert "اللائحة" in text
    assert "ولا يسري" in text
    assert "١٥٬٠٠٠ جنيه" in text
    assert "hr@nilebright.example" in text


def test_latin_only_pages_match_plain_extraction():
    with pymupdf.open(SAMPLES / "service_agreement_en.pdf") as doc:
        for page in doc:
            assert page_text(page).split() == page.get_text("text").split()
