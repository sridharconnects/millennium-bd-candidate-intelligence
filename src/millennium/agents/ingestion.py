"""Agent 1 -- Ingestion. Bytes on disk -> a clean, trusted, de-duplicated Document."""
from __future__ import annotations

import re
from pathlib import Path

from .. import sanitize
from ..ingest import Document, detect_type, load_document
from .base import AgentResult, subagent

AGENT = "ingestion"


@subagent("ingest.detect_type", AGENT, "1.0")
def detect(path: Path) -> AgentResult:
    """Identify the file by magic bytes rather than trusting its extension."""
    t = detect_type(Path(path))
    supported = t in ("pdf", "docx")
    return AgentResult(name="", output={"file_type": t, "supported": supported},
                       status="ok" if supported else "failed",
                       confidence=1.0 if supported else 0.0,
                       errors=[] if supported else [f"unsupported file type: {t}"])


@subagent("ingest.extract", AGENT, "1.2")
def extract(path: Path, is_synthetic: bool = False) -> AgentResult:
    """Text extraction with layout repair (column order, ligatures, merged cells)."""
    doc = load_document(Path(path), is_synthetic=is_synthetic)
    status = "ok" if doc.text.strip() else "failed"
    if doc.warnings and status == "ok":
        status = "partial"
    return AgentResult(name="", output=doc, status=status,
                       confidence=doc.extraction_quality,
                       warnings=doc.repairs + doc.warnings,
                       errors=[] if doc.text.strip() else ["no extractable text (OCR required)"])


@subagent("ingest.language", AGENT, "1.0")
def language(text: str) -> AgentResult:
    """Coarse language detection via stopword profile -- no extra dependency needed.

    Deliberately coarse: the only decision it drives is whether to warn that a
    document is mostly non-English, and a heavyweight langdetect dependency is not
    worth carrying to Streamlit Cloud for that.
    """
    profiles = {
        "en": r"\b(the|and|of|for|with|from|to|in|a|an)\b",
        "fr": r"\b(le|la|les|de|des|du|et|pour|avec|dans|une)\b",
        "es": r"\b(el|la|los|las|de|del|y|para|con|en|una)\b",
        "pt": r"\b(o|a|os|as|de|do|da|e|para|com|em|uma)\b",
    }
    scores = {k: len(re.findall(v, text, re.I)) for k, v in profiles.items()}
    total = sum(scores.values()) or 1
    lang = max(scores, key=scores.get)
    share = scores[lang] / total
    warn = [] if lang == "en" else [f"document appears to be predominantly '{lang}' "
                                    f"({share:.0%} of stopword hits); extraction quality may drop"]
    # Non-English fragments inside an English CV are normal and not worth flagging.
    if lang == "en" and scores["en"] < 8:
        warn.append("very few English stopwords found -- text may be fragmentary or tabular")
    return AgentResult(name="", output={"language": lang, "confidence": round(share, 3),
                                        "scores": scores},
                       confidence=round(share, 3), warnings=warn)


@subagent("ingest.injection_scan", AGENT, "1.1")
def injection_scan(doc: Document, path: Path | None = None, enabled: bool = True) -> AgentResult:
    """Detect and neutralise prompt-injection payloads before any LLM sees the text."""
    if not enabled:
        return AgentResult(name="", status="skipped", output={"flags": [], "text": doc.text})
    res = sanitize.scan(doc.text, doc.doc_id)
    findings = list(res.findings)
    if path and Path(path).suffix.lower() == ".pdf":
        findings.extend(sanitize.scan_pdf_visual(path))
    flags = sorted({f["name"] for f in findings})
    high = [f for f in findings if f["severity"] == "high"]
    return AgentResult(
        name="", status="partial" if high else "ok",
        output={"flags": flags, "findings": findings, "text": res.clean_text,
                "neutralised": res.neutralised},
        confidence=1.0,
        warnings=([f"INJECTION DEFENCE: neutralised {res.neutralised} span(s); "
                   f"categories={flags}"] if findings else []),
    )


def _shingles(text: str, k: int = 5) -> set[int]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {hash(" ".join(words[i:i + k])) for i in range(max(0, len(words) - k + 1))}


@subagent("ingest.near_duplicate", AGENT, "1.1")
def near_duplicate(doc: Document, corpus: list[tuple[str, str, str]],
                   threshold: float = 0.55) -> AgentResult:
    """Flag re-submissions of the same candidate via shingle Jaccard similarity.

    This is a real recruiting problem, not a synthetic one: the same candidate arrives
    from three agencies with three slightly different formats, and a naive pool
    triple-counts them in every distribution chart. Exact text hash catches nothing
    because each agency reformats.

    `corpus` is [(doc_id, label, text)]. At this scale a linear pass is exact and
    instant; the MinHash/LSH swap-in is documented in README under scaling triggers.
    """
    mine = _shingles(doc.text)
    hits = []
    for other_id, label, text in corpus:
        if other_id == doc.doc_id or not text:
            continue
        theirs = _shingles(text)
        union = len(mine | theirs) or 1
        jac = len(mine & theirs) / union
        if jac >= threshold:
            hits.append({"doc_id": other_id, "label": label, "jaccard": round(jac, 3)})
    hits.sort(key=lambda h: -h["jaccard"])
    return AgentResult(
        name="", output={"duplicates": hits}, status="partial" if hits else "ok",
        warnings=[f"near-duplicate of {h['label']} (Jaccard {h['jaccard']}) -- "
                  f"likely the same candidate from a different source" for h in hits],
    )


@subagent("ingest.quality", AGENT, "1.0")
def quality(doc: Document) -> AgentResult:
    """Score how trustworthy the text layer is; caps downstream confidence."""
    q = doc.extraction_quality
    tier = "good" if q >= 0.8 else "acceptable" if q >= 0.55 else "poor"
    return AgentResult(name="", output={"extraction_quality": q, "tier": tier},
                       confidence=q,
                       warnings=[] if tier != "poor" else
                       [f"poor text layer (score {q}); routing to human review"])
