"""Intake — upload resumes and watch the pipeline work.

Two reasons this page exists beyond the obvious one:

* A recruiting tool you cannot add a resume to is a demo, not a product. Agencies send
  CVs continuously; the pool is never static.
* It is the only place the pipeline trace is visible. Every subagent reports status,
  confidence, latency and cost under a uniform contract, and rendering that as a
  timeline is what turns "there are seven agents" from a claim into something a
  reviewer can watch happen — including a document degrading gracefully rather than
  taking the batch down.

Uploads are untrusted. Type is checked by magic bytes rather than extension, size and
page count are capped, filenames are randomised before anything touches disk, and the
injection scanner runs before a single byte reaches the model.
"""
from __future__ import annotations

import io
import json
import os
import html
import re
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from millennium import taxonomy as tx
from millennium.config import SETTINGS
from millennium.ingest import detect_type
from millennium.llm import LLMUnavailable
from millennium.orchestrator import Pipeline
from millennium.schema import (CandidateProfile, Classification, ProvenanceRecord,
                               SensitiveAttributes, SkillEntry, Tracked, stable_id)
from . import components as C
from . import theme

ORCH_PYTHON = "Python pipeline (default)"
ORCH_LANGCHAIN = "LangChain agent (experimental)"

MAX_BYTES = 8 * 1024 * 1024
MAX_PAGES = 20
ALLOWED = {"pdf", "docx"}


def _stage(path: Path, data: bytes) -> tuple[Path | None, list[str]]:
    """Validate an upload and stage it under a randomised name.

    The original filename is never used on disk: it is attacker-controlled and is a
    path-traversal and overwrite vector. It is retained only as a display label.
    """
    problems: list[str] = []
    if len(data) > MAX_BYTES:
        problems.append(f"{path.name}: {len(data)/1e6:.1f} MB exceeds the {MAX_BYTES/1e6:.0f} MB cap")
        return None, problems

    tmp_dir = SETTINGS.paths.artifacts / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    safe = tmp_dir / f"upload_{uuid.uuid4().hex[:12]}{path.suffix.lower()}"
    safe.write_bytes(data)

    kind = detect_type(safe)
    if kind not in ALLOWED:
        problems.append(f"{path.name}: rejected — magic bytes say '{kind}', not PDF or DOCX "
                        f"(the extension was not trusted)")
        safe.unlink(missing_ok=True)
        return None, problems

    if kind == "pdf":
        try:
            import fitz
            with fitz.open(str(safe)) as d:
                if d.page_count > MAX_PAGES:
                    problems.append(f"{path.name}: {d.page_count} pages exceeds the "
                                    f"{MAX_PAGES}-page cap")
                    safe.unlink(missing_ok=True)
                    return None, problems
        except Exception as e:  # noqa: BLE001
            problems.append(f"{path.name}: unreadable PDF ({type(e).__name__})")
            safe.unlink(missing_ok=True)
            return None, problems
    return safe, problems


def _trace_table(result) -> pd.DataFrame:
    rows = []
    for r in result.trace:
        for f in r.flatten():
            rows.append({
                "subagent": f.name,
                "status": f.status,
                "conf": round(f.confidence, 2),
                "ms": f.latency_ms,
                "cached": "yes" if f.cached else "",
                "cost": f"${f.cost_usd:.5f}" if f.cost_usd else "",
                "warnings": len(f.warnings),
                "errors": len(f.errors),
            })
    return pd.DataFrame(rows)


def _merge_into_pool(new_profiles: list[CandidateProfile]) -> int:
    """Add profiles to the session's working pool, deduped by candidate_id.

    The working pool is session-scoped, not written to `data/exports/` -- it survives
    navigation and reruns but not a browser refresh or 'Reset demo', which mirrors
    how every other piece of session state in this app already behaves (shortlist,
    saved filters). `app.py` folds `manual_profiles` into `pool` and into the search
    index's cache key, so an addition here shows up in Search, Analytics, and
    Requisition on the very next rerun -- not just on this page.
    """
    existing = {p.candidate_id for p in st.session_state.manual_profiles}
    added = [p for p in new_profiles if p.candidate_id not in existing]
    st.session_state.manual_profiles += added
    return len(added)


def _tracked(value, source_label: str) -> Tracked:
    """A field supplied directly at import time -- not extracted, so never claimed as
    'verified' (that status is reserved for a value whose span was located in a
    source document). `unverified` here means exactly what it says: present, and
    unproven by this pipeline."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return Tracked.missing()
    return Tracked(value=value, normalized_value=value, confidence=0.55,
                  extraction_method="human", validation_status="unverified",
                  notes=[f"supplied at import from {source_label}, not span-verified"])


def _row_to_profile(row: dict, i: int, source_label: str) -> CandidateProfile:
    """Map one CSV row or JSON record (an arbitrary, user-supplied flat schema) into a
    valid CandidateProfile. Deliberately minimal rather than a lossy best-effort
    reconstruction of the full parsed shape: every field this function fills in is
    honestly labelled 'human' / 'unverified' provenance, never 'llm' or 'verified',
    because none of it passed through span verification."""
    def get(*keys):
        for k in keys:
            if row.get(k) not in (None, ""):
                return row[k]
        return None

    name = get("name", "full_name", "candidate_name") or f"Imported candidate {i + 1}"
    doc_id = f"import-{uuid.uuid4().hex[:12]}"
    cid = stable_id(doc_id, str(name), SETTINGS.schema_version)

    skills_raw = get("skills") or []
    if isinstance(skills_raw, str):
        skill_names = [s.strip() for s in re.split(r"[;,]", skills_raw) if s.strip()]
    elif isinstance(skills_raw, list):
        skill_names = [str(s).strip() for s in skills_raw if str(s).strip()]
    else:
        skill_names = []

    years_raw = get("years_experience", "years")
    try:
        years_val = float(years_raw) if years_raw not in (None, "") else None
    except (TypeError, ValueError):
        years_val = None

    p = CandidateProfile(
        candidate_id=cid, doc_id=doc_id,
        sensitive=SensitiveAttributes(full_name=_tracked(name, source_label)),
        headline=_tracked(get("headline", "title"), source_label),
        summary=_tracked(get("summary", "notes"), source_label),
        location_current=_tracked(get("region", "location", "geography"), source_label),
        years_experience=_tracked(years_val, source_label),
        skills=[SkillEntry(canonical=s, depth="mentioned") for s in skill_names],
        provenance=ProvenanceRecord(
            source_file=source_label, file_sha256="", text_sha256="",
            file_type="import", extractor="manual-import/structured",
            schema_version=SETTINGS.schema_version, taxonomy_version=tx.TAXONOMY_VERSION,
            is_synthetic=False),
    )
    strategy = get("strategy", "strategies")
    if strategy:
        p.strategies = [Classification(label=str(strategy), confidence=0.5,
                                       rationale="supplied at import, not classified")]
    sector = get("sector", "sectors")
    if sector:
        p.sectors = [Classification(label=str(sector), confidence=0.5,
                                    rationale="supplied at import, not classified")]
    seniority = get("seniority", "seniority_level")
    if seniority:
        p.seniority = Classification(label=str(seniority), confidence=0.5,
                                     rationale="supplied at import, not classified")
    known = sum(1 for t in (p.headline, p.summary, p.location_current, p.years_experience)
               if t.is_known)
    p.quality.completeness = min(1.0, known / 4 + (0.1 if p.skills else 0.0))
    p.quality.validation_flags = [f"imported from {source_label} — no field below is "
                                  f"LLM-extracted or span-verified"]
    return p


def _parse_import_file(upload) -> tuple[list[CandidateProfile], list[str]]:
    """-> (profiles, problems). Two JSON shapes are accepted: this app's own full
    export (`{"candidates": [...]}` of complete CandidateProfile dicts -- e.g.
    re-importing a previous export or another run's synthetic corpus) and a flat
    list of simple records, handled the same way a CSV row is. A record that fails
    full validation falls back to the flat mapper rather than being dropped."""
    name = upload.name
    problems: list[str] = []
    if name.lower().endswith(".json"):
        try:
            payload = json.loads(upload.getvalue().decode("utf8"))
        except Exception as e:  # noqa: BLE001
            return [], [f"{name}: invalid JSON ({type(e).__name__}: {e})"]
        records = payload.get("candidates", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            return [], [f"{name}: expected a JSON array or an object with a "
                       f"'candidates' array"]
        profiles = []
        for i, rec in enumerate(records):
            if not isinstance(rec, dict):
                problems.append(f"{name}: record {i + 1} is not an object — skipped")
                continue
            try:
                profiles.append(CandidateProfile.model_validate(rec))
            except Exception:
                try:
                    profiles.append(_row_to_profile(rec, i, name))
                except Exception as e:  # noqa: BLE001
                    problems.append(f"{name}: record {i + 1} could not be read "
                                    f"({type(e).__name__}) — skipped")
        return profiles, problems

    if name.lower().endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(upload.getvalue()))
        except Exception as e:  # noqa: BLE001
            return [], [f"{name}: could not read as CSV ({type(e).__name__}: {e})"]
        df = df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in df.columns})
        profiles = []
        for i, row in enumerate(df.to_dict(orient="records")):
            try:
                profiles.append(_row_to_profile(row, i, name))
            except Exception as e:  # noqa: BLE001
                problems.append(f"{name}: row {i + 1} could not be read "
                                f"({type(e).__name__}) — skipped")
        return profiles, problems

    return [], [f"{name}: unsupported file type — use .csv or .json"]


def _render_bulk_import() -> None:
    st.caption(
        "For structured data you already have — not a resume PDF/Word file. JSON "
        "accepts this app's own full export shape (`candidates.json`) or a flat "
        "array of simple records; CSV expects one row per candidate with columns "
        "like `name, headline, summary, region, strategy, sector, seniority, "
        "years_experience, skills`. Every field brought in this way is labelled "
        "**human / unverified** — none of it has been LLM-extracted or "
        "span-verified, and that is shown wherever the record appears.")
    up = st.file_uploader("Structured data", type=["csv", "json"],
                          accept_multiple_files=True, label_visibility="collapsed",
                          key="bulk_import_uploader")
    if not up:
        return
    all_profiles: list[CandidateProfile] = []
    all_problems: list[str] = []
    for f in up:
        profs, probs = _parse_import_file(f)
        all_profiles += profs
        all_problems += probs
    for prob in all_problems:
        st.markdown(f'<div class="mm-warn">{html.escape(prob)}</div>',
                    unsafe_allow_html=True)
    if not all_profiles:
        return
    prev = pd.DataFrame([{
        "name": p.sensitive.full_name.display(),
        "headline": p.headline.display(),
        "region": p.location_current.display(),
        "years_exp": p.years_experience.display(),
        "skills": ", ".join(s.canonical for s in p.skills[:6]),
    } for p in all_profiles])
    st.dataframe(prev, width="stretch", hide_index=True)
    if st.button(f"➕ Add {len(all_profiles)} imported candidate(s) to the working pool",
                type="primary", key="add_import_to_pool"):
        n = _merge_into_pool(all_profiles)
        st.success(f"Added {n} candidate(s) — {len(all_profiles) - n} were already "
                  f"in the working pool. They now appear in Search, Analytics, and "
                  f"Requisition.")
        st.rerun()


def render_intake(profiles, synth, pool, index, index_manifest, manifest, store,
                  client, bench, evals):
    C.page_kicker("Intake",
                  "Parse a resume through the live pipeline, or import structured CSV/JSON.")

    upload_col, side_col = st.columns([0.62, 0.38], gap="medium",
                                      vertical_alignment="top")
    with upload_col:
        with st.container(border=True, key="intake_upload_panel"):
            st.markdown('<div class="mm-panel-heading"><span>Upload resumes</span>'
                        '<b>PDF / Word</b></div>', unsafe_allow_html=True)
            st.caption("Files are checked, renamed, scanned for prompt injection, then "
                       "parsed into structured candidate profiles.")
            C.llm_callout(
                "Resume parsing",
                "Uses the LLM to extract structured candidate fields from uploaded "
                "resumes after file safety checks and prompt-injection scanning.",
                stage="ingest")

            if SETTINGS.flags.demo_mode:
                st.info("DEMO_MODE is on. Cached resumes replay without cost; new "
                        "resumes need DEMO_MODE=0 and ANTHROPIC_API_KEY in Streamlit "
                        "secrets to parse live.")

            orch = st.radio(
                "Orchestration", [ORCH_PYTHON, ORCH_LANGCHAIN], horizontal=True,
                help=f"{ORCH_PYTHON}: reliable production path. {ORCH_LANGCHAIN}: "
                     "optional live tool-calling demonstration; requires API key.")
            is_langchain = orch == ORCH_LANGCHAIN
            if is_langchain:
                st.markdown(
                    "**LangChain orchestration mode** uses the same resume parser "
                    "through a live tool-calling agent. Upload one file and run it "
                    "only after the deployment secret is set.")
            up = st.file_uploader("Drop resumes here", type=["pdf", "docx"],
                                  accept_multiple_files=not is_langchain)
            if is_langchain and up and not isinstance(up, list):
                up = [up]

            demo_col, run_col = st.columns([0.5, 0.5])
            use_fixture = (not is_langchain) and demo_col.checkbox(
                "Include injection test resume",
                help="Adds the test file used to demonstrate prompt-injection scanning.")
            key_missing = is_langchain and not os.getenv("ANTHROPIC_API_KEY")
            go = run_col.button("Run parsing pipeline", type="primary", width="stretch",
                                disabled=not (up or use_fixture) or key_missing)
            if key_missing:
                st.caption("ANTHROPIC_API_KEY is not set. Add it in Streamlit Cloud "
                           "Secrets to run the live LangChain mode.")

    with side_col:
        with st.container(border=True, key="intake_import_panel"):
            st.markdown('<div class="mm-panel-heading"><span>Bulk import</span>'
                        '<b>CSV / JSON</b></div>', unsafe_allow_html=True)
            _render_bulk_import()
        with st.expander("How resume parsing works", expanded=False):
            _explain_pipeline()

    if not go:
        return

    paths: list[tuple[str, Path]] = []
    problems: list[str] = []
    for f in (up or []):
        staged, probs = _stage(Path(f.name), f.getvalue())
        problems += probs
        if staged:
            paths.append((f.name, staged))
    if use_fixture:
        fx = SETTINGS.paths.root / "tests" / "fixtures" / "injected_resume.pdf"
        if fx.exists():
            paths.append(("injected_resume.pdf (test fixture)", fx))
        else:
            problems.append("run scripts/make_injected_fixture.py to build the fixture")

    for p in problems:
        st.markdown(f'<div class="mm-danger">{html.escape(p)}</div>', unsafe_allow_html=True)
    if not paths:
        return

    if is_langchain:
        _run_langchain(paths[0])
        return

    pipe = Pipeline(client=client, max_workers=min(4, len(paths)))
    bar = st.progress(0.0, text="starting…")
    results = []
    t0 = time.perf_counter()
    for i, (label, path) in enumerate(paths, 1):
        bar.progress(i / len(paths), text=f"[{i}/{len(paths)}] {label}")
        try:
            results.append((label, pipe.process(path)))
        except Exception as e:  # noqa: BLE001 -- the UI must never die on one bad file
            st.error(f"{label}: {type(e).__name__}: {e}")
    bar.empty()
    elapsed = time.perf_counter() - t0

    k = st.columns(5)
    ok = sum(1 for _l, r in results if r.status != "failed")
    C.kpi(k[0], f"{ok}/{len(results)}", "parsed")
    C.kpi(k[1], f"{elapsed:.1f}s", "elapsed",
          f"{len(results)/max(elapsed,1e-6)*60:.0f} docs/min")
    C.kpi(k[2], sum(len(r.trace) for _l, r in results), "subagent calls")
    C.kpi(k[3], f"${sum(r.cost_usd for _l, r in results):.4f}", "LLM cost")
    flagged = sum(1 for _l, r in results
                  if r.profile and r.profile.provenance
                  and r.profile.provenance.injection_flags)
    C.kpi(k[4], flagged, "injections caught",
          colour="#B91C1C" if flagged else theme.ACCENT)

    C.section_break("Parse results", 2)
    for label, res in results:
        st.markdown(f"#### {html.escape(label)}")
        if res.status == "failed":
            # Colour-coded by what actually happened -- an LLM-unavailable-in-DEMO_MODE
            # miss is expected and low-severity; a genuine crash is not, and the two
            # looked identical (both a red "danger" box) before this classified them.
            st.markdown(theme.flag_card(str(res.error), prefix="Degraded, not crashed — "),
                       unsafe_allow_html=True)
            st.caption("The batch continued; this document yields abstained fields and "
                      "a recorded error rather than taking the run down.")
        p = res.profile
        if p is not None:
            if p.provenance and p.provenance.injection_flags:
                st.markdown(
                    f'<div class="mm-danger"><b>Prompt injection detected and '
                    f'neutralised.</b> Categories: '
                    f'{html.escape(", ".join(p.provenance.injection_flags))}. The payload '
                    f'was stripped before the model saw it, the call carried no tools, and '
                    f'every surviving field was independently span-verified — so the '
                    f'profile below is unaffected.</div>', unsafe_allow_html=True)
            st.markdown(C.candidate_card(p, st.session_state.blind), unsafe_allow_html=True)
            if p.quality.review_reasons:
                st.markdown('<div class="mm-warn"><b>Routed to human review:</b> '
                            + html.escape("; ".join(p.quality.review_reasons))
                            + "</div>", unsafe_allow_html=True)

        with st.expander(f"Pipeline trace · {sum(s for s in res.stage_ms.values())} ms across "
                         f"{len(res.trace)} subagents"):
            if res.stage_ms:
                st.markdown("**Stage timings**")
                sm = pd.DataFrame([{"stage": k, "ms": v} for k, v in res.stage_ms.items()])
                st.dataframe(sm, width="stretch", hide_index=True)
            df = _trace_table(res)
            if not df.empty:
                st.dataframe(
                    df, width="stretch", hide_index=True,
                    column_config={"status": st.column_config.TextColumn(width="small"),
                                   "ms": st.column_config.NumberColumn(format="%d ms")})
            warns = [w for r in res.trace for f in r.flatten() for w in f.warnings]
            if warns:
                st.markdown("**Warnings** — every repair and every refusal is logged, "
                            "colour-coded by kind")
                st.markdown(theme.flag_list(warns[:24]), unsafe_allow_html=True)

    parsed_ok = [res.profile for _l, res in results if res.profile is not None]
    if parsed_ok:
        st.divider()
        if st.button(f"➕ Add {len(parsed_ok)} parsed candidate(s) to the working pool",
                    type="primary", key="add_parsed_to_pool"):
            n = _merge_into_pool(parsed_ok)
            st.success(f"Added {n} candidate(s) to the working pool — they now appear "
                      f"in Search, Analytics, and Requisition for this session. "
                      f"({SETTINGS.paths.exports} on disk is untouched; re-run "
                      f"`scripts/run_pipeline.py` to make an addition permanent.)")
            st.rerun()


_TOOL_ICON = {
    "ingest_document": "📄", "scan_for_injection": "🛡",
    "segment_and_scan_contacts": "🔎", "extract_with_llm": "🤖",
    "ground_and_merge": "📍", "adjudicate_conflicts": "⚖",
    "classify_and_validate": "🏷", "finalize_profile": "✅",
}


def _run_langchain(item: tuple[str, Path]) -> None:
    """Run the LangChain `AgentExecutor` on one document and render its real trace.

    This is the direct, literal answer to "how are the multiple agents
    communicating": every row below is one tool call the orchestrating agent actually
    made, in the order it actually made them, with the actual JSON it actually sent
    and received — not a diagram of how it would work.
    """
    label, path = item
    try:
        from millennium.langchain_agents import LangChainPipeline, LangChainUnavailable
    except ImportError:
        st.error("langchain / langchain-anthropic are not installed. "
                 "pip install langchain langchain-core langchain-anthropic")
        return

    from millennium.llm import LLMClient
    pipe = LangChainPipeline(client=LLMClient(demo_mode=False))
    with st.spinner(f"Agent orchestrating {label}… this makes live API calls, "
                    f"typically 15–30s"):
        try:
            profile, report = pipe.process(path)
        except LangChainUnavailable as e:
            st.error(str(e))
            return

    k = st.columns(5)
    C.kpi(k[0], report["status"], "status",
          colour=theme.ACCENT if report["status"] == "ok" else "#B45309")
    C.kpi(k[1], report["n_tool_calls"], "tool calls")
    C.kpi(k[2], f"{report['elapsed_ms']/1000:.1f}s", "elapsed")
    C.kpi(k[3], report["tokens_in"] + report["tokens_out"], "agent tokens",
          "routing overhead, not extraction")
    C.kpi(k[4], f"${report['cost_usd']:.4f}", "extraction cost",
          "$0 = replayed from the disk cache")

    if report["status"] != "ok":
        st.markdown(theme.flag_card(report.get("error", "no profile was produced"),
                                    prefix="Agent run did not complete — "),
                   unsafe_allow_html=True)

    st.divider()
    st.markdown("##### Agent communication trace")
    st.caption("Each row is one real tool call LangChain's `AgentExecutor` made, in "
               "the order it chose. The agent reasons only over the summaries shown "
               "here — never over resume text, which no tool returns.")

    steps = report.get("langchain_steps", [])
    internal = {t["tool"]: t for t in report.get("internal_trace", [])}
    if not steps:
        st.caption("No tool calls were recorded (the agent may have hit its "
                  "iteration limit before calling any tool).")
    for i, s in enumerate(steps, 1):
        tool = s["tool"]
        icon = _TOOL_ICON.get(tool, "•")
        note = internal.get(tool, {}).get("summary", "")
        ok = internal.get(tool, {}).get("ok", True)
        border = theme.ACCENT if ok else "#B45309"
        st.markdown(
            f'<div class="mm-card" style="border-left:3px solid {border}">'
            f'<div class="mm-row" style="justify-content:space-between">'
            f'<span class="mm-name">{i}. {icon} {html.escape(tool)}</span>'
            f'<span class="mm-sub mm-mono">step {i}/{len(steps)}</span></div>'
            f'<div class="mm-sub">{html.escape(note)}</div></div>',
            unsafe_allow_html=True)
        with st.expander(f"raw input/output — {tool}", expanded=False):
            st.json({"input": s["input"], "output": s["output"]})

    if report.get("final_message"):
        st.markdown("##### Agent's own summary")
        msg = report["final_message"]
        text = msg if isinstance(msg, str) else "\n".join(
            b.get("text", "") for b in msg if isinstance(b, dict))
        st.markdown(f'<div class="mm-banner">{html.escape(text)}</div>',
                   unsafe_allow_html=True)

    if profile is not None:
        st.divider()
        st.markdown("##### Resulting profile")
        st.markdown(C.candidate_card(profile, st.session_state.blind),
                   unsafe_allow_html=True)
        st.caption(f"completeness {profile.quality.completeness:.0%} · "
                  f"{len(profile.employment)} role(s) · "
                  f"extractor: {profile.provenance.extractor if profile.provenance else '—'}")


def _explain_pipeline() -> None:
    stages = [
        ("1 · Ingest", "Magic-byte typing, layout-repairing extraction (column order, "
                       "ligatures, merged table cells), language detection, SHA-256 "
                       "hashing, near-duplicate check against the existing pool."),
        ("2 · Sanitize", "Ten prompt-injection pattern families plus a render-layer scan "
                         "for white-on-white and sub-3pt text. Payloads are neutralised "
                         "and logged, never silently dropped."),
        ("3 · Parse", "Three targeted LLM passes — identity, employment, profile — each "
                      "issued with no tools, each returning a verbatim quote for every "
                      "value. A fourth adjudication pass runs only if the rule layer and "
                      "the model disagree."),
        ("4 · Ground", "Every quote is located in the source text. If it cannot be, the "
                       "value is discarded and the field is marked abstained. This is the "
                       "step that makes hallucination self-limiting."),
        ("5 · Classify", "Closed-taxonomy labelling: strategy, sector, geography, "
                         "seniority (tier-adjusted), quant/fundamental profile, feeder "
                         "path — each with the trigger that fired."),
        ("6 · Validate", "Experience derived in Python from verified dates as a union of "
                         "intervals, timeline contradictions, gaps, duplicates, contact "
                         "plausibility, and review routing."),
        ("7 · Finalize", "Run manifest, provenance record, persistence, export."),
    ]
    for title, body in stages:
        st.markdown(f'<div class="mm-card"><div class="mm-name">{title}</div>'
                    f'<div class="mm-sub">{html.escape(body)}</div></div>',
                    unsafe_allow_html=True)
