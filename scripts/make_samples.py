"""Generate the two sample PDFs shipped with the demo and used by the evaluation.

Both documents describe a fictional company ("Nile Bright Solutions") and are
written for this project, so they can be published and redistributed freely.
Run on Windows with Word installed: python scripts/make_samples.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parents[1] / "samples"

AR_PAGES = [
    """
    <h1>لائحة شؤون العاملين</h1>
    <h2>شركة النيل المضيء للحلول التقنية (شركة افتراضية)</h2>
    <p>إصدار ٢٠٢٦، وتسري هذه اللائحة اعتبارا من أول يناير ٢٠٢٦ على جميع العاملين بالشركة في مقريها بالقاهرة والإسكندرية.</p>
    <h3>المادة (١): مواعيد العمل</h3>
    <p>أيام العمل الرسمية من الأحد إلى الخميس، وساعات العمل ثماني ساعات يوميا تبدأ من الساعة التاسعة صباحا وتنتهي الساعة الخامسة مساء، منها ساعة راحة.</p>
    <p>يسمح بالعمل من المنزل يومين في الأسبوع بحد أقصى بعد موافقة المدير المباشر، ولا يسري ذلك خلال فترة الاختبار.</p>
    <h3>المادة (٢): فترة الاختبار</h3>
    <p>يعين الموظف الجديد تحت الاختبار لمدة ثلاثة أشهر، ويجوز مدها مرة واحدة لمدة لا تزيد على ثلاثة أشهر أخرى بقرار من مدير الموارد البشرية.</p>
    """,
    """
    <h3>المادة (٣): الإجازات السنوية</h3>
    <p>يستحق الموظف إجازة سنوية مدفوعة الأجر مدتها ٢١ يوم عمل، وتزاد إلى ٣٠ يوم عمل لمن أمضى عشر سنوات في الخدمة أو تجاوز سن الخمسين.</p>
    <p>يجوز ترحيل سبعة أيام بحد أقصى من رصيد الإجازة السنوية إلى العام التالي، ويسقط ما زاد على ذلك.</p>
    <h3>المادة (٤): الإجازات المرضية</h3>
    <p>تمنح الإجازة المرضية بناء على تقرير طبي معتمد، ويصرف عنها الأجر كاملا لمدة شهر واحد في السنة، ثم ٧٥٪ من الأجر لمدة شهرين إضافيين.</p>
    <h3>المادة (٥): إجازة الوضع ورعاية الطفل</h3>
    <p>تستحق الموظفة إجازة وضع مدتها أربعة أشهر بأجر كامل، ويستحق الموظف إجازة أبوة مدفوعة الأجر مدتها خمسة أيام عمل.</p>
    """,
    """
    <h3>المادة (٦): الأجور والمكافآت</h3>
    <p>تصرف الأجور في الخامس والعشرين من كل شهر ميلادي بالتحويل البنكي، فإذا وافق يوم عطلة تصرف في آخر يوم عمل قبله.</p>
    <p>تصرف مكافأة سنوية تعادل أجر شهرين بحد أقصى، وفقا لنتيجة تقييم الأداء السنوي الذي يتم في شهر ديسمبر.</p>
    <h3>المادة (٧): التدريب</h3>
    <p>تتحمل الشركة تكلفة الدورات التدريبية والشهادات المهنية المعتمدة بحد أقصى ١٥٬٠٠٠ جنيه مصري للموظف في السنة، ويلتزم الموظف بالاستمرار في العمل سنة كاملة بعد انتهاء أي دورة تزيد تكلفتها على ٥٬٠٠٠ جنيه.</p>
    <h3>المادة (٨): إنهاء الخدمة</h3>
    <p>يلتزم الموظف الذي يرغب في الاستقالة بإخطار الشركة كتابة قبل شهرين من تاريخ ترك العمل، ويخفض الإخطار إلى شهر واحد خلال فترة الاختبار.</p>
    <p>للتواصل مع إدارة الموارد البشرية: البريد الإلكتروني hr@nilebright.example وتحويلة رقم ٢٠٤.</p>
    """,
]

EN_PAGES = [
    """
    <h1>Master Services Agreement</h1>
    <p><b>Between:</b> Nile Bright Solutions LLC ("Provider") and Delta Retail Group ("Client"). Both parties are fictional.</p>
    <p><b>Agreement No.:</b> MSA-2026-014. <b>Effective date:</b> 1 March 2026.</p>
    <h3>1. Scope of Services</h3>
    <p>The Provider will design, build and operate a customer-support chatbot for the Client's online store, including integration with the Client's order-tracking system and an admin dashboard.</p>
    <h3>2. Term</h3>
    <p>This Agreement starts on the effective date and runs for an initial term of 18 months. It renews automatically for successive 12-month periods unless either party gives written notice of non-renewal at least 60 days before the end of the current term.</p>
    """,
    """
    <h3>3. Fees and Payment</h3>
    <p>The Client will pay a one-time setup fee of USD 12,000 and a monthly operations fee of USD 2,500. The Provider invoices on the first business day of each month.</p>
    <p>Invoices are payable within 30 days. Late payments accrue interest of 1.5% per month on the overdue amount.</p>
    <h3>4. Service Levels</h3>
    <p>The Provider guarantees 99.5% monthly uptime for the chatbot. For each full 0.5% below this target, the Client receives a credit of 5% of that month's operations fee, up to a maximum credit of 25%.</p>
    <p>Critical incidents must receive a first response within 1 hour and be resolved within 8 hours.</p>
    """,
    """
    <h3>5. Data Protection</h3>
    <p>Customer conversation logs are stored in the Client's cloud region and kept for 90 days, after which they are deleted. The Provider must notify the Client of any personal-data breach within 48 hours of discovering it.</p>
    <h3>6. Liability</h3>
    <p>Each party's total liability under this Agreement is limited to the fees paid by the Client in the 12 months before the claim.</p>
    <h3>7. Termination</h3>
    <p>Either party may terminate for convenience with 90 days' written notice. Either party may terminate immediately if the other party materially breaches this Agreement and fails to cure the breach within 30 days of written notice.</p>
    <h3>8. Governing Law</h3>
    <p>This Agreement is governed by the laws of the Arab Republic of Egypt. Disputes go to arbitration at the Cairo Regional Centre for International Commercial Arbitration.</p>
    """,
]


PAGE_BREAK = '<br style="page-break-before:always">'


def write_pdf(pages: list[str], path: Path, rtl: bool) -> None:
    """Render HTML pages to PDF with Microsoft Word, like most real-world PDFs.

    Word output reproduces the Arabic extraction quirks the app must handle
    (reversed digits, split lam-alef ligatures), which makes the samples a
    realistic test. Requires Windows with Word installed.
    """
    body_attrs = 'dir="rtl" style="font-family:Arial"' if rtl else 'style="font-family:Calibri"'
    html = f'<html><head><meta charset="utf-8"></head><body {body_attrs}>{PAGE_BREAK.join(pages)}</body></html>'
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / (path.stem + ".html")
        src.write_text(html, encoding="utf-8")
        script = (
            "$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
            f"$d = $w.Documents.Open('{src}'); $d.SaveAs2('{path}', 17); $d.Close(0); $w.Quit()"
        )
        subprocess.run(["powershell.exe", "-NoProfile", "-Command", script], check=True)
        _clear_personal_metadata(path)


def _clear_personal_metadata(path: Path) -> None:
    """Word stamps the Office account's name as the PDF author; replace it."""
    with pymupdf.open(path) as doc:
        doc.set_metadata({"title": path.stem, "author": "ragchat sample document", "creator": "", "producer": ""})
        doc.del_xml_metadata()
        data = doc.tobytes(garbage=3, deflate=True)
    path.write_bytes(data)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    write_pdf(AR_PAGES, OUT / "hr_policy_ar.pdf", rtl=True)
    write_pdf(EN_PAGES, OUT / "service_agreement_en.pdf", rtl=False)
    for pdf in sorted(OUT.glob("*.pdf")):
        with pymupdf.open(pdf) as doc:
            if len(doc) != 3:
                raise RuntimeError(f"{pdf.name}: expected 3 pages, got {len(doc)}")
        print("wrote", pdf.name)


if __name__ == "__main__":
    main()
