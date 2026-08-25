"""Ingestion regressions. Each test pins a defect found in the supplied corpus."""
from pathlib import Path

import pytest

from millennium.ingest import clean_text, detect_type, load_document

ROOT = Path(__file__).resolve().parent.parent


def test_type_detection_uses_magic_bytes(tmp_path):
    """A .docx extension on a PDF must not fool the router."""
    fake = tmp_path / "resume.docx"
    fake.write_bytes(b"%PDF-1.7\n" + b"0" * 100)
    assert detect_type(fake) == "pdf"
    junk = tmp_path / "resume.pdf"
    junk.write_bytes(b"not a document at all")
    assert detect_type(junk) == "unknown"


def test_ligature_and_subsetting_repair():
    """Omar's PDF has Type1 subsetting damage: U+019F is 'ti', U+019E is 'tf'."""
    out, repairs = clean_text("QuanƟtaƟve Porƞolio Oﬃce Diﬀerence")
    assert out == "Quantitative Portfolio Office Difference"
    assert any("ligature" in r for r in repairs)


def test_hyphen_wrap_and_midsentence_rejoin():
    out, _ = clean_text("quantita-\ntive research\nof the portfolio")
    assert "quantitative research" in out


def test_invisible_characters_stripped():
    out, repairs = clean_text("hello​world‮reversed")
    assert "​" not in out and "‮" not in out
    assert any("zero-width" in r for r in repairs)


def test_omar_pdf_column_order_is_repaired():
    """Naive extraction interleaves the skills sidebar into the experience bullets."""
    d = load_document(ROOT / "Omar El-Hassan 202405.pdf")
    assert any("2-column" in w or "column" in w for w in d.warnings)
    t = d.text
    # The full-time role and its follow-on bullet must be adjacent, with the sidebar
    # contact line pushed out of the middle of the experience section.
    i_role = t.find("Quantitative Developer")
    i_intro = t.find("Introduction of a new model")
    i_email = t.find("o.elhassan15@gmail.com")
    assert -1 < i_role < i_intro, "experience block is still fragmented"
    assert i_email > i_intro, "sidebar contact line is still interleaved into experience"
    assert "QuanƟtaƟve" not in t


def test_single_column_pdf_is_not_split_by_a_centred_header():
    """A centred name block sits far right of the body and produces the same x-gap as a
    real column. Treating it as one moves the candidate's NAME to the bottom of the
    document. Regression: this is exactly what the first column detector did to Ryan's
    single-column CV while correctly fixing Omar's two-column one."""
    d = load_document(ROOT / "RYAN PATEL - Resume.pdf")
    assert not [w for w in d.warnings if "column layout" in w], \
        "single-column CV was wrongly treated as multi-column"
    head = d.text[:200]
    assert head.startswith("RYAN PATEL"), f"name is no longer first: {head[:60]!r}"
    assert "ryan.patel0403@gmail.com" in head, "contact block was displaced from the top"


def test_real_two_column_pdf_is_still_repaired():
    """The guard above must not disable the repair where it is genuinely needed."""
    d = load_document(ROOT / "Omar El-Hassan 202405.pdf")
    assert [w for w in d.warnings if "column layout" in w], \
        "genuine two-column layout is no longer detected"


def test_viktor_merged_cells_are_deduplicated():
    """Merged <w:tc> elements repeat once per grid column; content must not 4x."""
    d = load_document(ROOT / "Viktor Sharat.docx")
    phrase = "Tracked 38 companies within the healthcare fund portfolio"
    assert d.text.count(phrase) == 1, f"achievement repeated {d.text.count(phrase)}x"
    assert len(d.text) < 4200, "merged-cell duplication is still inflating the document"


def test_michael_body_order_preserves_sections():
    """Tables must be read in document order, not appended after all paragraphs."""
    d = load_document(ROOT / "Michael Rodriguez, CFA.docx")
    t = d.text
    assert 0 <= t.find("EDUCATION") < t.find("EXPERIENCE"), "section order is inverted"
    assert "Ann Arbor, MI May 2017" in t, "tab between columns was dropped"


def test_every_supplied_resume_extracts():
    for p in sorted(ROOT.iterdir()):
        if p.suffix.lower() not in (".pdf", ".docx") or p.name.startswith("~$"):
            continue
        d = load_document(p)
        assert len(d.text) > 800, f"{p.name} extracted only {len(d.text)} chars"
        assert d.extraction_quality > 0.5, f"{p.name} quality {d.extraction_quality}"
        assert d.file_sha256 and d.doc_id
