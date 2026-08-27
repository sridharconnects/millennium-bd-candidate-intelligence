"""Reusable UI pieces. Every one of them exists to make a claim checkable."""
from __future__ import annotations

import html

import streamlit as st

from millennium import taxonomy as tx
from millennium.schema import CandidateProfile, Evidence, Tracked
from . import theme


def loading_screen(subtitle: str) -> str:
    """A branded full-width loading state for the one-time cold-start cost (ONNX
    embedder load, index build over the full pool) -- replaces Streamlit's own tiny
    top-left spinner, which reads as the app stalling rather than as the product
    doing real, visible work. Rendered into a single `st.empty()` placeholder in
    app.py and cleared the instant the underlying cached call returns, so on every
    later rerun (a cache hit) it never appears at all."""
    return (
        '<div class="mm-loading">'
        '<div class="mm-loading-ring"><span class="mm-loading-mark">M</span></div>'
        '<div class="mm-loading-title">Millennium BD</div>'
        f'<div class="mm-loading-sub">{html.escape(subtitle)}</div>'
        '</div>')


def kpi(col, value, label: str, hint: str = "", colour: str | None = None,
        delta: str | None = None, delta_tone: str = "flat") -> None:
    """Metric card. Optional delta_tone: up | down | flat."""
    delta_html = ""
    if delta:
        tone = {"up": "delta-up", "down": "delta-down"}.get(delta_tone, "delta-flat")
        delta_html = f'<div class="delta {tone}">{html.escape(delta)}</div>'
    col.markdown(
        f'<div class="mm-kpi"><div class="v" style="color:{colour or theme.INK}">{value}</div>'
        f'<div class="l">{html.escape(label)}</div>'
        + (f'<div class="h">{html.escape(hint)}</div>' if hint else "")
        + delta_html
        + "</div>", unsafe_allow_html=True)


def metric_card(value, label: str, hint: str = "", colour: str | None = None,
                delta: str | None = None, delta_tone: str = "flat") -> None:
    """Full-width metric card (not column-bound)."""
    delta_html = ""
    if delta:
        tone = {"up": "delta-up", "down": "delta-down"}.get(delta_tone, "delta-flat")
        delta_html = f'<div class="delta {tone}">{html.escape(delta)}</div>'
    st.markdown(
        f'<div class="mm-kpi"><div class="v" style="color:{colour or theme.INK}">{value}</div>'
        f'<div class="l">{html.escape(label)}</div>'
        + (f'<div class="h">{html.escape(hint)}</div>' if hint else "")
        + delta_html
        + "</div>", unsafe_allow_html=True)


def llm_callout(title: str, body: str, *, stage: str = "") -> None:
    """Compact disclosure for any action that sends context to the LLM."""
    stage_html = f'<b>{html.escape(stage)}</b>' if stage else ""
    st.markdown(
        '<div class="mm-llm-callout">'
        '<span class="mm-llm-chip">Uses LLM</span>'
        f'<div><strong>{html.escape(title)}</strong>'
        f'<p>{html.escape(body)}</p></div>{stage_html}</div>',
        unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "", meta: str = "") -> None:
    """Toolbar-style section head used above results, analytics blocks, etc."""
    sub = (f'<div class="mm-toolbar-meta">{html.escape(subtitle)}</div>'
           if subtitle else "")
    right = (f'<div class="mm-toolbar-meta">{html.escape(meta)}</div>' if meta else "")
    st.markdown(
        f'<div class="mm-toolbar"><div><div class="mm-toolbar-title">'
        f'{html.escape(title)}</div>{sub}</div>{right}</div>',
        unsafe_allow_html=True)


def status_badge(text: str, tone: str = "plain") -> str:
    return theme.chip(text, tone)


def section_break(label: str, index: int = 0) -> None:
    """A labelled, coloured divider between major sections of a long page.

    Plain `st.divider()` repeated many times on one page (Workflow, Intake) reads as
    one undifferentiated wall -- there is nothing to tell a reader where section 2
    ends and section 3 begins short of reading every word. Cycling through the same
    categorical palette the charts use gives every section a distinct, consistent
    colour, so the page can be scanned by colour and label alone.
    """
    colour = theme.SERIES[index % len(theme.SERIES)]
    st.markdown(
        f'<div class="mm-section-break" style="--sc:{colour}">'
        f'<span class="tick"></span>'
        f'<span class="label">{html.escape(label)}</span>'
        f'<span class="rule"></span></div>', unsafe_allow_html=True)


def tracked_value(t: Tracked, label: str, show_status: bool = True,
                  show_method: bool = True) -> str:
    """Render a field so that 'unknown' is never mistaken for 'zero' or 'no'.

    The distinction the whole product turns on: an ABSTAINED field means the model
    proposed something and we threw it away because it could not be proven, which is
    a very different thing from the CV simply not mentioning it.

    `show_method` adds a second chip naming exactly what produced the value -- the LLM,
    a regex rule, a Python computation over verified fields, or a human correction.
    That is the direct, per-field answer to "how is this happening via LLM".
    """
    method = theme.method_chip(t.extraction_method) if show_method else ""
    if t.is_known:
        badge = theme.status_chip(t.validation_status,
                                  f"{t.validation_status} · {t.confidence:.0%}") if show_status else ""
        return (f'<div class="mm-row"><span class="mm-sub">{html.escape(label)}</span>'
                f'<span style="font-weight:600">{html.escape(str(t.display()))}</span>'
                f'{badge}{method}</div>')
    kind = "abstained" if t.validation_status == "abstained" else "missing"
    word = "abstained — unprovable" if kind == "abstained" else "not stated in document"
    return (f'<div class="mm-row"><span class="mm-sub">{html.escape(label)}</span>'
            f'<span style="color:{theme.MUTED}">—</span>{theme.status_chip(kind, word)}'
            f'{method if kind == "abstained" else ""}</div>')


def evidence_block(ev: Evidence, raw_text: str, pad: int = 190) -> str:
    """The evidence viewer: the exact span, highlighted inside its real surroundings."""
    s, e = max(0, ev.char_start), min(len(raw_text), ev.char_end)
    lo, hi = max(0, s - pad), min(len(raw_text), e + pad)
    before, hit, after = raw_text[lo:s], raw_text[s:e], raw_text[e:hi]
    loc = f"char {ev.char_start}–{ev.char_end}"
    if ev.page:
        loc = f"page {ev.page} · " + loc
    return (f'<div class="mm-ev">…{html.escape(before)}<mark>{html.escape(hit)}</mark>'
            f'{html.escape(after)}…<br><span class="mm-sub mm-mono">'
            f'{loc} · {ev.match_kind} match ({ev.match_score:.2f})</span></div>')


def evidence_for(t: Tracked, profile: CandidateProfile, caption: str = "") -> None:
    if not t.evidence:
        st.caption("No source span recorded for this field.")
        return
    if caption:
        st.caption(caption)
    for ev in t.evidence[:3]:
        # Hard invariant, re-checked at render time: a span may only ever be shown
        # under the candidate whose document it came from.
        if ev.doc_id != profile.doc_id:
            st.error("Evidence integrity failure: this span belongs to another document. "
                     "It has been withheld.")
            continue
        st.markdown(evidence_block(ev, profile.raw_text), unsafe_allow_html=True)


def score_bar(name: str, weight: float, score: float, contribution: float,
              maximum: float = 0.35) -> str:
    pct = min(100, contribution / max(maximum, 1e-6) * 100)
    colour = theme.ACCENT if score >= 0.66 else "#B45309" if score >= 0.33 else "#94A3B8"
    return (
        f'<div style="margin:5px 0"><div class="mm-row" style="justify-content:space-between">'
        f'<span style="font-size:0.8rem;font-weight:600">{html.escape(name)}</span>'
        f'<span class="mm-sub mm-mono">{weight:.2f} × {score:.2f} = {contribution:.3f}</span></div>'
        f'<div class="mm-bar-wrap"><div class="mm-bar" style="width:{pct:.1f}%;'
        f'background:{colour}"></div></div></div>')


def labels_row(profile: CandidateProfile, limit: int = 6) -> str:
    bits = []
    if profile.geo_region:
        bits.append(theme.chip(tx.REGION_DISPLAY.get(profile.geo_region.label,
                                                     profile.geo_region.label)))
    if profile.seniority and profile.seniority.label.startswith("L"):
        lvl = int(profile.seniority.label[1:])
        bits.append(theme.chip(f"{profile.seniority.label} · {tx.display("seniority", lvl)}"))
    if profile.quant_fundamental:
        bits.append(theme.chip(profile.quant_fundamental.label.title()))
    for c in profile.strategies[:limit]:
        tone = "verified" if c.confidence >= 0.7 and not c.low_support else "missing"
        bits.append(theme.chip(tx.display("strategy", c.label), tone))
    for c in profile.sectors[:limit]:
        tone = "verified" if c.confidence >= 0.7 and not c.low_support else "missing"
        bits.append(theme.chip(tx.display("sector", c.label), tone))
    return "".join(bits)


def candidate_card(p: CandidateProfile, blind: bool = False, score: float | None = None,
                   explain: str = "") -> str:
    cur = p.current_role()
    role = ""
    if cur:
        role = f"{cur.title_raw.display('—')} · {cur.employer_canonical or cur.employer_raw.display('—')}"
        if cur.employer_tier and cur.employer_tier != "unknown":
            role += f" ({tx.display("tier", cur.employer_tier)})"
    yrs = (f"{p.years_experience.value:.1f}y experience"
           if p.years_experience.is_known else "experience unknown")
    flags = []
    if p.quality.needs_human_review:
        flags.append(theme.chip("needs review", "abstained"))
    if p.provenance and p.provenance.injection_flags:
        flags.append(theme.chip("injection neutralised", "conflicted"))
    if p.provenance and p.provenance.near_duplicate_of:
        flags.append(theme.chip("near-duplicate", "conflicted"))
    if p.provenance and p.provenance.is_synthetic:
        flags.append(theme.chip("SYNTHETIC", "human_corrected"))
    sc = (f'<span class="mm-cand-score">{score:.3f}</span>' if score is not None else "")
    name = p.display_name(blind)
    initials = "".join(part[0] for part in name.replace("(", " ").split() if part[:1].isalpha())[:2].upper() or "·"
    return (
        f'<div class="mm-card mm-cand">'
        f'<div class="mm-cand-avatar">{html.escape(initials)}</div>'
        f'<div class="mm-cand-body">'
        f'<div class="mm-row" style="justify-content:space-between;align-items:center">'
        f'<span class="mm-name">{html.escape(name)}</span>{sc}</div>'
        f'<div class="mm-sub">{html.escape(role)}</div>'
        f'<div class="mm-sub">{html.escape(yrs)}'
        + (f' · {html.escape(explain)}' if explain else "") + "</div>"
        f'<div style="margin-top:6px">{labels_row(p)}{"".join(flags)}</div>'
        f'</div></div>')


def footer(metrics: dict) -> None:
    bits = " · ".join(f"{k} {v}" for k, v in metrics.items())
    st.markdown(f'<div class="mm-foot">{html.escape(bits)}</div>', unsafe_allow_html=True)


def empty_state(title: str, body: str, icon: str = "⌕") -> None:
    st.markdown(
        f'<div class="mm-empty">'
        f'<div class="mm-empty-icon">{html.escape(icon)}</div>'
        f'<h4>{html.escape(title)}</h4>'
        f'<div>{html.escape(body)}</div></div>',
        unsafe_allow_html=True)


def page_kicker(title: str, subtitle: str = "") -> None:
    """In-page section title when the top bar already named the route."""
    sub = f'<div class="mm-page-sub" style="margin-bottom:8px">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="mm-page-head" style="margin-bottom:4px">'
        f'<div class="mm-page-title">{html.escape(title)}</div>{sub}</div>',
        unsafe_allow_html=True)


def synthetic_banner(n: int) -> None:
    if n:
        st.markdown(
            f'<div class="mm-synth">⚠ Synthetic corpus active — {n} of the candidates '
            f'shown are LLM-generated records used solely for scalability benchmarking. '
            f'They are excluded from every accuracy metric.</div>', unsafe_allow_html=True)


def provenance_banner(p: CandidateProfile) -> None:
    """One line, always visible, answering 'how was this profile actually produced'.

    Three distinct states, because they mean different things to a recruiter deciding
    how much to trust what they're looking at:
      * a genuine LLM parse (teal) -- the required path, working as intended;
      * the rule baseline (amber) -- NOT the required path, a fallback/comparison run;
      * LLM requested but unavailable (amber) -- e.g. DEMO_MODE with no cached response
        for this specific file, so the fields that needed the model abstained.
    """
    pv = p.provenance
    if pv is None:
        st.markdown('<div class="mm-warn">No provenance recorded for this profile.</div>',
                    unsafe_allow_html=True)
        return
    extractor = pv.extractor or ""
    degraded = any("pipeline degraded" in f.lower() and "parse." in f.lower()
                  for f in p.quality.validation_flags)

    if "parse:rules" in extractor:
        st.markdown(
            '<div class="mm-warn">'
            f'{theme.method_chip("rule")} '
            '<b>Parsed via the deterministic rule baseline</b> — regex and taxonomy '
            'matching, not the LLM API. This is the published rule-vs-LLM comparison '
            'path (see DECISIONS.md), not the case study\'s required parsing path.'
            '</div>', unsafe_allow_html=True)
    elif degraded:
        st.markdown(
            '<div class="mm-warn">'
            f'{theme.method_chip("llm")} '
            '<b>LLM parsing was requested but unavailable for this document</b> — most '
            'likely <code>DEMO_MODE</code> with no cached response for this exact file. '
            'Fields that needed the model <u>abstained rather than guessing</u>. Run '
            '<code>python scripts/run_pipeline.py</code> with '
            '<code>ANTHROPIC_API_KEY</code> set to populate it for real.'
            '</div>', unsafe_allow_html=True)
    else:
        model = html.escape(pv.llm_model or "unknown model")
        cost = f"${pv.cost_usd:.4f}" if pv.cost_usd else "$0.0000 (replayed from cache)"
        st.markdown(
            '<div class="mm-banner">'
            f'{theme.method_chip("llm")} '
            f'<b>Parsed via the Anthropic API</b> — model <code>{model}</code> · '
            f'cost {cost}. Every field below carries its own tag showing exactly what '
            f'produced it — the model, a rule, a Python computation, or a reviewer.'
            '</div>', unsafe_allow_html=True)


def resume_preview(p: CandidateProfile, height: int = 340, expanded: bool = True) -> None:
    """Extracted source text (post layout-repair), for the Source tab.

    Distinct from `original_document_view` below on purpose: this shows what the
    *parser* read after column-order and ligature repair -- the exact text every
    evidence span points into. The original-document view shows what a *human* would
    open -- the real PDF or Word file, unmodified.
    """
    src = p.provenance.source_file if p.provenance else "source document"
    with st.expander(f"📄 Extracted text — {html.escape(src)} "
                     f"({len(p.raw_text):,} characters, after layout repair)",
                     expanded=expanded):
        if not p.raw_text:
            st.caption("No source text is attached to this record in the current view.")
            return
        st.caption("Exactly what the parser read — column order and OCR-adjacent "
                   "repairs applied, nothing else changed. This is the text every "
                   "evidence span points into.")
        st.text_area("Extracted text", p.raw_text, height=height,
                     label_visibility="collapsed", key=f"resume_src_{p.candidate_id}")


def _resolve_original_file(p: CandidateProfile):
    """Locate the real, unmodified source file on disk, if it still exists.

    The 10 documents supplied with the case study live in the project root next to
    app.py, and `provenance.source_file` is exactly their filename -- so for those
    ten, and for anything else staged the same way, resolution is a straight lookup.
    Records with no recoverable original (synthetic benchmark rows; a record from a
    session where the uploaded original was never persisted) get None, handled by the
    caller as a clearly-labelled unavailable state rather than a crash.
    """
    from pathlib import Path

    from millennium.config import SETTINGS
    if not p.provenance or not p.provenance.source_file:
        return None
    candidate = SETTINGS.paths.root / p.provenance.source_file
    return candidate if candidate.exists() and candidate.is_file() else None


@st.cache_data(show_spinner=False)
def _docx_to_html(file_bytes: bytes) -> str:
    """DOCX has no native browser renderer; convert to HTML once and cache it.

    `mammoth` is deliberately the choice here over a LibreOffice/`docx2pdf` round
    trip: pure Python, no system dependency, and it survives a free Streamlit Cloud
    deploy the same way the rest of this stack was chosen to.
    """
    import io

    import mammoth
    result = mammoth.convert_to_html(io.BytesIO(file_bytes))
    return result.value


def original_document_view(p: CandidateProfile, height: int = 640) -> None:
    """The real, unmodified source file — not extracted text, not a re-render.

    This is deliberately the counterpart to the agent-work panel next to it: the left
    side is everything the pipeline produced; this side is exactly what a recruiter
    would see if they opened the attachment themselves, so the two can be compared
    directly.
    """
    path = _resolve_original_file(p)
    if path is None:
        st.markdown(
            '<div class="mm-warn">Original file not available in this session — '
            'this record has no recoverable source document (a synthetic benchmark '
            'row, or a prior upload whose original was not retained).</div>',
            unsafe_allow_html=True)
        return

    suffix = path.suffix.lower()
    data = path.read_bytes()

    if suffix == ".pdf":
        st.pdf(data, height=height)
    elif suffix == ".docx":
        try:
            body_html = _docx_to_html(data)
        except Exception as e:  # noqa: BLE001 -- a broken render must not crash the page
            st.markdown(
                f'<div class="mm-warn">Could not render this Word document '
                f'({type(e).__name__}) — use the download button below instead.</div>',
                unsafe_allow_html=True)
        else:
            # NOTE: st.components.v1.html is deprecated in favour of st.iframe, not
            # removed until 2026-06-01. st.iframe's `src` takes a URL or a local Path,
            # not a raw HTML string, so switching means writing the mammoth output to
            # a temp file first -- a real change, not a rename. Deferred: the warning
            # is silent to the end user and there's a full year of runway on it.
            import streamlit.components.v1 as components
            components.html(
                f"""<div style="font-family:Calibri,Arial,sans-serif;font-size:14px;
                    line-height:1.5;color:#1a1a1a;background:#fff;padding:28px 34px;
                    max-width:760px;margin:0 auto;box-shadow:0 1px 4px rgba(0,0,0,.08);
                    border-radius:4px;">{body_html}</div>""",
                height=height, scrolling=True)
    else:
        st.caption(f"No in-browser viewer for {suffix} files.")

    st.download_button(f"Download original ({path.name})", data, file_name=path.name,
                       width="stretch", key=f"dl_orig_{p.candidate_id}")
