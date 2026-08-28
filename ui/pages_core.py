"""Search / Candidate / Requisition / Shortlist — the recruiter's daily workspace."""
from __future__ import annotations

import html
import json
import re
import time
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from millennium import taxonomy as tx
from millennium.config import SETTINGS, ScoreWeights
from millennium.index import build_chunks
from millennium.llm import LLMUnavailable
from millennium.prompts import requisition_prompt
from millennium.retrieval import (ParsedQuery, apply_filters, retrieve, similar_candidates,
                                  understand_query)
from millennium.scoring import gap_analysis, minimal_edit, rank, weight_sensitivity
from . import components as C
from . import llm_panels
from . import theme

# Example searches, shown as suggestion pills under the search bar. Keys are short,
# readable labels (what a recruiter sees on the chip); values are the full
# plain-English query that actually runs. The old version rendered the full query
# text as five equal-width buttons, which truncated into meaningless slivers the
# moment the viewport narrowed (e.g. with the chat panel open).
EXAMPLES = {
    "Healthcare L/S · APAC": "healthcare equity long/short in APAC, no banking background",
    "Quant dev · C++ · Europe": "quant developer, C++ derivatives pricing, Europe",
    "CFA + 5y credit research": "must have CFA and 5+ years credit or fixed income research",
    "Sell-side TMT → buy-side": "sell-side TMT analyst ready to move buy-side, US",
    "Systematic factor · Python": "systematic factor research with Python and backtesting",
}


def _apply_example() -> None:
    """on_change for the example pills: copy the full query into the search box and
    clear the pill again, so the same example can be re-picked later and a chip never
    stays highlighted while the user has since typed something else. Writing the
    widget's own key is legal here because callbacks run before the script re-renders
    the widget."""
    # dict-style access, not .get(): the AppTest harness used in tests implements
    # `in`/`[]` on session_state but not `.get()` (see the nav_radio note in app.py).
    pick = st.session_state["search_examples"] if "search_examples" in st.session_state else None
    if pick:
        st.session_state.query = EXAMPLES[pick]
        st.session_state.search_examples = None


def _clear_query() -> None:
    """on_click for the results header's ✕ — legal write to the widget key because
    callbacks run before the next script render."""
    st.session_state.query = ""


def _step_candidate(delta: int) -> None:
    """on_click for Prev / Next on the candidate page.

    The selectbox (`cand_switch_box`) owns its own widget state. Writing only
    `selected` and rerunning does nothing: on the next run Streamlit restores the
    selectbox's previous value and overwrites the profile. Dropping the widget
    key here lets the next run reconstruct the selectbox from the new `selected`.
    """
    ids = st.session_state["_cand_ids"] if "_cand_ids" in st.session_state else []
    cur = st.session_state.selected
    if cur not in ids:
        return
    i = ids.index(cur) + delta
    if 0 <= i < len(ids):
        st.session_state.selected = ids[i]
        if "cand_switch_box" in st.session_state:
            del st.session_state["cand_switch_box"]


def _sync_candidate_from_switch() -> None:
    """on_change for the candidate selectbox — keep `selected` in lockstep."""
    st.session_state.selected = st.session_state.cand_switch_box


def _flag_on(key: str) -> bool:
    return key in st.session_state and bool(st.session_state[key])


def _toggle_studio(which: str) -> None:
    """Open one Search-hero studio, closing the other so they never stack."""
    key = f"{which}_studio_open"
    currently = _flag_on(key)
    st.session_state["match_studio_open"] = False
    st.session_state["import_studio_open"] = False
    st.session_state[key] = not currently


def _close_studio(which: str) -> None:
    st.session_state[f"{which}_studio_open"] = False


def _remove_from_pool(cids: list[str]) -> int:
    """Drop profiles from this session's working pool.

    Imported records leave `manual_profiles`. Corpus records are hidden via
    `hidden_ids` (Reset demo restores them). Anyone shortlisted is removed from
    that list too. This is the recruiter-workspace delete, not GDPR erasure —
    that still lives on Review and actually wipes SQLite / FTS / vectors.
    """
    want = list(dict.fromkeys(c for c in cids if c))
    if not want:
        return 0
    if "hidden_ids" not in st.session_state:
        st.session_state.hidden_ids = []
    hidden = set(st.session_state.hidden_ids)
    st.session_state.manual_profiles = [
        p for p in st.session_state.manual_profiles if p.candidate_id not in want]
    sl = st.session_state.shortlist
    for cid in want:
        if cid in sl:
            sl.pop(cid)
        hidden.add(cid)
        if "selected" in st.session_state and st.session_state.selected == cid:
            st.session_state.selected = None
    st.session_state.hidden_ids = list(hidden)
    return len(want)


# What each retrieval mode actually does with a query, in recruiter-readable steps.
# Rendered by _mode_popover next to the mode selector, alongside MEASURED accuracy
# from the eval artifact -- the explanation is grounded in numbers this repo
# computed, not marketing copy.
_MODE_STEPS = {
    "hybrid": (
        "Two independent searches run and their rankings are fused",
        ["Your query is embedded into a 384-dimension vector (bge-small, runs "
         "locally) and compared against every profile chunk — this catches "
         "**paraphrases**: \"healthcare\" finds a CV that only says "
         "\"pharma coverage\".",
         "The same query also runs against a BM25 keyword index (SQLite FTS5) — "
         "this catches **exact terms**: CFA, C++, a fund name.",
         "The two ranked lists are merged with Reciprocal Rank Fusion: each chunk "
         "scores `Σ 1 / (60 + rank)` across both lists, so a candidate ranked well "
         "by *both* searches beats one ranked #1 by only one of them.",
         "A candidate's **best** chunk wins — a long CV with many mediocre matches "
         "cannot outrank a short one with a perfect match."]),
    "dense": (
        "Semantic (embedding) search only",
        ["Your query is embedded into a 384-dimension vector (bge-small, runs "
         "locally) and compared against every profile chunk by cosine similarity.",
         "Strong on **paraphrases and concepts** — \"healthcare\" finds \"pharma "
         "coverage\" — but can miss **exact identifiers** like CFA or C++ that "
         "carry little semantic weight.",
         "A candidate's best chunk wins."]),
    "lexical": (
        "Keyword (BM25) search only",
        ["Your query runs against a BM25 full-text index (SQLite FTS5) — the same "
         "family of scoring search engines used for decades.",
         "Strong on **exact terms, acronyms and names** — CFA, C++, J.P. Morgan — "
         "but blind to **synonyms**: \"healthcare\" will not find a CV that only "
         "says \"pharma\".",
         "A candidate's best chunk wins."]),
}


def _mode_popover(mode: str, evals: dict) -> None:
    """The ⓘ next to the retrieval-mode selector: exactly what the selected mode is
    doing with the query right now, plus the measured ablation for all three modes
    so 'hybrid is the default' is a demonstrated claim, not an asserted one."""
    with st.popover("ⓘ", help=f"How “{mode}” retrieval ranks candidates",
                    width="stretch"):
        title, steps = _MODE_STEPS[mode]
        st.markdown(f"**{mode} — {title}**")
        for i, s in enumerate(steps, 1):
            st.markdown(f"{i}. {s}")
        rows = [a for a in (evals or {}).get("ablation", [])
                if "hashing" not in a.get("mode", "")]
        if rows:
            st.markdown("**Measured on this corpus** (16 hand-graded queries — "
                        "`scripts/run_eval.py`):")
            lines = ["| mode | nDCG@10 | MRR | p50 latency |", "|---|---:|---:|---:|"]
            for a in rows:
                is_current = a["mode"].startswith(mode)
                name = f"**{a['mode']}**" if is_current else a["mode"]
                fmt = (lambda v: f"**{v}**") if is_current else (lambda v: v)
                ndcg = f"{a['ndcg@10']:.3f}"
                mrr = f"{a['mrr']:.3f}"
                latency = f"{a['latency_ms']:.1f} ms"
                lines.append(f"| {name} | {fmt(ndcg)} | {fmt(mrr)} | {fmt(latency)} |")
            st.markdown("\n".join(lines))
            st.caption("Higher nDCG@10 / MRR = better ranking. Every result below "
                       "shows its own provenance (\"semantic #2 + keyword #1\"), so "
                       "you can see which search found each candidate.")


def _byid(pool) -> dict:
    return {p.candidate_id: p for p in pool}


def _facets(pool) -> dict:
    f = {"region": set(), "country": set(), "strategy": set(), "sector": set(),
         "skill": set(), "seniority": set(), "employer": set(), "tier": set(),
         "cert": set(), "degree": set(), "language": set(), "feeder": set(),
         "approach": set()}
    for p in pool:
        if p.geo_region:
            f["region"].add(tx.display("region", p.geo_region.label))
        if p.geography:
            f["country"].add(p.geography.label)
        if p.seniority and p.seniority.label.startswith("L"):
            f["seniority"].add(p.seniority.label)
        if p.quant_fundamental:
            f["approach"].add(p.quant_fundamental.label)
        if p.feeder_path:
            f["feeder"].add(tx.display("feeder", p.feeder_path.label))
        f["strategy"] |= {tx.display("strategy", c.label) for c in p.strategies}
        f["sector"] |= {tx.display("sector", c.label) for c in p.sectors}
        f["skill"] |= {s.canonical for s in p.skills}
        f["cert"] |= {tx.display("certification", c.canonical) for c in p.certifications if c.canonical}
        f["language"] |= {l.language for l in p.languages}
        for e in p.employment:
            if e.employer_canonical:
                f["employer"].add(e.employer_canonical)
            if e.employer_tier and e.employer_tier != "unknown":
                f["tier"].add(tx.display("tier", e.employer_tier))
        f["degree"] |= {e.degree_level for e in p.education if e.degree_level}
    return {k: sorted(v) for k, v in f.items()}


def _manual_filter(pool, F: dict):
    """Filter-rail gating. Empty selection means 'no opinion', never 'exclude all'."""
    out = []
    for p in pool:
        region = tx.display("region", p.geo_region.label, "") if p.geo_region else ""
        strat = {tx.display("strategy", c.label) for c in p.strategies}
        sect = {tx.display("sector", c.label) for c in p.sectors}
        skills = {s.canonical for s in p.skills}
        certs = {tx.display("certification", c.canonical) for c in p.certifications if c.canonical}
        langs = {l.language for l in p.languages}
        emps = {e.employer_canonical for e in p.employment if e.employer_canonical}
        tiers = {tx.display("tier", e.employer_tier) for e in p.employment
                 if e.employer_tier and e.employer_tier != "unknown"}
        degs = {e.degree_level for e in p.education if e.degree_level}
        sen = p.seniority.label if p.seniority else ""
        y = p.years_experience.value

        if F["region"] and region not in F["region"]:
            continue
        if F["strategy"] and not (strat & set(F["strategy"])):
            continue
        if F["sector"] and not (sect & set(F["sector"])):
            continue
        if F["skill"] and not (skills & set(F["skill"])):
            continue
        if F["cert"] and not (certs & set(F["cert"])):
            continue
        if F["language"] and not (langs & set(F["language"])):
            continue
        if F["employer"] and not (emps & set(F["employer"])):
            continue
        if F["tier"] and not (tiers & set(F["tier"])):
            continue
        if F["degree"] and not (degs & set(F["degree"])):
            continue
        if F["seniority"] and sen not in F["seniority"]:
            continue
        if F["approach"] and (not p.quant_fundamental
                              or p.quant_fundamental.label not in F["approach"]):
            continue
        if F["feeder"]:
            feeder = (tx.display("feeder", p.feeder_path.label)
                      if p.feeder_path else None)
            if feeder not in F["feeder"]:
                continue
        lo, hi = F["years"]
        if y is not None and not (lo <= y <= hi):
            continue
        # Unknown experience is only excluded when the recruiter opts in, because
        # 'we could not derive it' is not the same as 'they have none'.
        if y is None and not F["include_unknown_years"]:
            continue
        if F["review_only"] and not p.quality.needs_human_review:
            continue
        if p.quality.completeness < F["min_completeness"]:
            continue
        out.append(p)
    return out


def _render_filter_bar(facets: dict, store, *, n_shown: int | None = None,
                       n_pool: int | None = None, latency_ms: float | None = None) -> dict:
    """Results toolbar: filters sit above the table, with active chips + view meta."""
    F: dict = {}
    active_bits: list[str] = []

    meta = ""
    if n_shown is not None and n_pool is not None and latency_ms is not None:
        meta = f"{n_shown} of {n_pool} candidates · {latency_ms:.0f} ms"

    st.markdown(
        f'<div class="mm-results-bar">'
        f'<div class="mm-results-bar-label">Results'
        + (f'<span class="mm-results-bar-meta">{html.escape(meta)}</span>' if meta else "")
        + '</div></div>',
        unsafe_allow_html=True)

    with st.container(horizontal=True, gap="small", key="filter_studio"):
        with st.popover("Filters", icon=":material/tune:"):
            st.caption("Narrow the working pool. Filters compose with the query.")
            c1, c2 = st.columns(2)
            with c1:
                F["region"] = st.multiselect("Region", facets["region"], key="f_region")
                F["strategy"] = st.multiselect("Strategy", facets["strategy"], key="f_strategy")
                F["sector"] = st.multiselect("Sector", facets["sector"], key="f_sector")
            with c2:
                F["skill"] = st.multiselect("Skills", facets["skill"], key="f_skill")
                F["seniority"] = st.multiselect(
                    "Seniority", facets["seniority"], key="f_seniority",
                    format_func=lambda s: f"{s} · {tx.display('seniority', s)}")
                F["years"] = st.slider(
                    "Years of experience", 0.0, 30.0, (0.0, 30.0), 0.5, key="f_years")
            F["include_unknown_years"] = st.checkbox(
                "Include candidates whose experience is unknown", value=True,
                key="f_include_unknown_years",
                help="Unknown ≠ zero. Some CVs state tenure as a duration with no dates.")
            with st.expander("More filters"):
                F["approach"] = st.multiselect("Approach", facets["approach"], key="f_approach")
                F["feeder"] = st.multiselect("Feeder path", facets["feeder"], key="f_feeder")
                F["employer"] = st.multiselect("Employer", facets["employer"], key="f_employer")
                F["tier"] = st.multiselect("Employer tier", facets["tier"], key="f_tier")
                F["cert"] = st.multiselect("Certification", facets["cert"], key="f_cert")
                F["degree"] = st.multiselect("Degree level", facets["degree"], key="f_degree")
                F["language"] = st.multiselect("Language", facets["language"], key="f_language")
                F["min_completeness"] = st.slider(
                    "Minimum profile completeness", 0.0, 1.0, 0.0, 0.05,
                    key="f_min_completeness")
                F["review_only"] = st.checkbox("Only records flagged for review",
                                                key="f_review_only")
            F.setdefault("min_completeness", 0.0)
            F.setdefault("review_only", False)
            F.setdefault("feeder", [])

        with st.popover("Saved", icon=":material/bookmark:"):
            saved = store.list_searches()
            if saved:
                pick = st.selectbox("Load", ["—"] + [s["name"] for s in saved],
                                    label_visibility="collapsed", key="load_saved_pick")
                if pick != "—":
                    rec = next(s for s in saved if s["name"] == pick)
                    lc, dc = st.columns(2)
                    if lc.button("Load", width="stretch", key="load_search"):
                        st.session_state.query = rec["query"]
                        st.session_state.retrieval_mode = rec.get("mode", "hybrid")
                        st.session_state.pending_filters = rec["filters"]
                        st.rerun()
                    if dc.button("Delete", width="stretch", key="del_search"):
                        store.delete_search(pick)
                        st.rerun()
            else:
                st.caption("No saved searches yet.")
            nm = st.text_input("Name", placeholder="e.g. APAC healthcare L/S",
                               label_visibility="collapsed", key="save_search_name")
            if st.button("Save current search", width="stretch",
                         disabled=not nm.strip(), key="save_search_btn"):
                snap = {
                    "region": st.session_state["f_region"] if "f_region" in st.session_state else [],
                    "strategy": st.session_state["f_strategy"] if "f_strategy" in st.session_state else [],
                    "sector": st.session_state["f_sector"] if "f_sector" in st.session_state else [],
                    "skill": st.session_state["f_skill"] if "f_skill" in st.session_state else [],
                    "seniority": st.session_state["f_seniority"] if "f_seniority" in st.session_state else [],
                    "years": st.session_state["f_years"] if "f_years" in st.session_state else (0.0, 30.0),
                    "approach": st.session_state["f_approach"] if "f_approach" in st.session_state else [],
                    "feeder": st.session_state["f_feeder"] if "f_feeder" in st.session_state else [],
                    "employer": st.session_state["f_employer"] if "f_employer" in st.session_state else [],
                    "tier": st.session_state["f_tier"] if "f_tier" in st.session_state else [],
                    "cert": st.session_state["f_cert"] if "f_cert" in st.session_state else [],
                    "degree": st.session_state["f_degree"] if "f_degree" in st.session_state else [],
                    "language": st.session_state["f_language"] if "f_language" in st.session_state else [],
                    "include_unknown_years": st.session_state["f_include_unknown_years"]
                    if "f_include_unknown_years" in st.session_state else True,
                    "min_completeness": st.session_state["f_min_completeness"]
                    if "f_min_completeness" in st.session_state else 0.0,
                    "review_only": st.session_state["f_review_only"]
                    if "f_review_only" in st.session_state else False,
                }
                store.save_search(nm.strip(), st.session_state.query, snap, "hybrid")
                st.success(f"Saved '{nm.strip()}'")

    for key, label in (("f_region", "Region"), ("f_strategy", "Strategy"),
                       ("f_sector", "Sector"), ("f_skill", "Skill"),
                       ("f_seniority", "Level")):
        vals = st.session_state[key] if key in st.session_state else []
        for v in vals or []:
            active_bits.append(f"{label}: {v}")
    years = st.session_state["f_years"] if "f_years" in st.session_state else (0.0, 30.0)
    if years != (0.0, 30.0):
        active_bits.append(f"Years: {years[0]:.0f}–{years[1]:.0f}")
    if active_bits:
        chips = "".join(
            f'<span class="mm-filter-chip">{html.escape(b)}</span>' for b in active_bits[:8])
        extra = (f'<span class="mm-sub">+{len(active_bits) - 8} more</span>'
                 if len(active_bits) > 8 else "")
        st.markdown(
            f'<div class="mm-filter-chip-row">{chips}{extra}</div>',
            unsafe_allow_html=True)

    F.setdefault("region", st.session_state["f_region"] if "f_region" in st.session_state else [])
    F.setdefault("strategy", st.session_state["f_strategy"] if "f_strategy" in st.session_state else [])
    F.setdefault("sector", st.session_state["f_sector"] if "f_sector" in st.session_state else [])
    F.setdefault("skill", st.session_state["f_skill"] if "f_skill" in st.session_state else [])
    F.setdefault("seniority", st.session_state["f_seniority"] if "f_seniority" in st.session_state else [])
    F.setdefault("years", years)
    F.setdefault("include_unknown_years", st.session_state["f_include_unknown_years"]
                 if "f_include_unknown_years" in st.session_state else True)
    F.setdefault("approach", st.session_state["f_approach"] if "f_approach" in st.session_state else [])
    F.setdefault("feeder", st.session_state["f_feeder"] if "f_feeder" in st.session_state else [])
    F.setdefault("employer", st.session_state["f_employer"] if "f_employer" in st.session_state else [])
    F.setdefault("tier", st.session_state["f_tier"] if "f_tier" in st.session_state else [])
    F.setdefault("cert", st.session_state["f_cert"] if "f_cert" in st.session_state else [])
    F.setdefault("degree", st.session_state["f_degree"] if "f_degree" in st.session_state else [])
    F.setdefault("language", st.session_state["f_language"] if "f_language" in st.session_state else [])
    F.setdefault("min_completeness", st.session_state["f_min_completeness"]
                 if "f_min_completeness" in st.session_state else 0.0)
    F.setdefault("review_only", st.session_state["f_review_only"]
                 if "f_review_only" in st.session_state else False)
    return F


# ============================================================================ SEARCH
def render_search(profiles, synth, pool, index, index_manifest, manifest, store,
                  client, bench, evals):
    C.synthetic_banner(len(synth) if st.session_state.include_synthetic else 0)

    # Command card (query + tools) and a separate results panel so Filters sit
    # directly above the table — Linear/Notion density, not sidebar clutter.
    facets = _facets(pool)

    # A saved search restores filters, not just the query text. Widget state must be
    # written before the widgets are constructed, and only values still present in
    # the current facets are restored.
    pending = st.session_state.pop("pending_filters", None)
    if pending:
        facet_key = {"region": "region", "strategy": "strategy", "sector": "sector",
                     "skill": "skill", "seniority": "seniority", "approach": "approach",
                     "feeder": "feeder", "employer": "employer", "tier": "tier",
                     "cert": "cert", "degree": "degree", "language": "language"}
        for fk, ffk in facet_key.items():
            vals = [v for v in (pending.get(fk) or []) if v in facets.get(ffk, [])]
            st.session_state[f"f_{fk}"] = vals
        if isinstance(pending.get("years"), (list, tuple)) and len(pending["years"]) == 2:
            st.session_state["f_years"] = tuple(float(x) for x in pending["years"])
        if "include_unknown_years" in pending:
            st.session_state["f_include_unknown_years"] = bool(pending["include_unknown_years"])
        if "min_completeness" in pending:
            st.session_state["f_min_completeness"] = float(pending["min_completeness"])
        if "review_only" in pending:
            st.session_state["f_review_only"] = bool(pending["review_only"])

    hero = st.container(border=False, key="search_hero")
    with hero:
        from . import cc_surfaces

        show_n = st.session_state["f_show_n"] if "f_show_n" in st.session_state else 50
        q0 = st.session_state["query"] if "query" in st.session_state else ""
        cc_surfaces.command_bar(
            query=q0,
            mode="hybrid",
            show=show_n,
            examples=[{"label": k, "query": v} for k, v in EXAMPLES.items()],
            meta=f"{len(pool)} in pool",
            key="cc_command",
        )
        query = st.session_state["query"] if "query" in st.session_state else ""
        mode = "hybrid"
        show_n = st.session_state["f_show_n"] if "f_show_n" in st.session_state else 50
        n_sl = len(st.session_state.shortlist)

        # Filter widgets live in the results panel; read session keys here so Match
        # studio and retrieval see the same active filters.
        F = {
            "region": st.session_state["f_region"] if "f_region" in st.session_state else [],
            "strategy": st.session_state["f_strategy"] if "f_strategy" in st.session_state else [],
            "sector": st.session_state["f_sector"] if "f_sector" in st.session_state else [],
            "skill": st.session_state["f_skill"] if "f_skill" in st.session_state else [],
            "seniority": st.session_state["f_seniority"] if "f_seniority" in st.session_state else [],
            "years": st.session_state["f_years"] if "f_years" in st.session_state else (0.0, 30.0),
            "include_unknown_years": st.session_state["f_include_unknown_years"]
            if "f_include_unknown_years" in st.session_state else True,
            "approach": st.session_state["f_approach"] if "f_approach" in st.session_state else [],
            "feeder": st.session_state["f_feeder"] if "f_feeder" in st.session_state else [],
            "employer": st.session_state["f_employer"] if "f_employer" in st.session_state else [],
            "tier": st.session_state["f_tier"] if "f_tier" in st.session_state else [],
            "cert": st.session_state["f_cert"] if "f_cert" in st.session_state else [],
            "degree": st.session_state["f_degree"] if "f_degree" in st.session_state else [],
            "language": st.session_state["f_language"] if "f_language" in st.session_state else [],
            "min_completeness": st.session_state["f_min_completeness"]
            if "f_min_completeness" in st.session_state else 0.0,
            "review_only": st.session_state["f_review_only"]
            if "f_review_only" in st.session_state else False,
        }
        show_cap = len(pool) if show_n == "All" else int(show_n)
        t0 = time.perf_counter()
        filtered = _manual_filter(pool, F)
        parsed = None
        hits = []
        if query.strip():
            parsed, _presult = understand_query(query, client)
            r = retrieve(index, parsed.semantic_text or query, mode,
                         top_k=max(show_cap, len(pool)))
            hits = r.output or []
            allowed = {p.candidate_id for p in filtered}
            gated, excluded, caveats = apply_filters(filtered, parsed)
            gset = {p.candidate_id for p in gated}
            ordered = [h for h in hits if h.candidate_id in allowed and h.candidate_id in gset]
            byid = _byid(pool)
            results = [(byid[h.candidate_id], h.score, h.explain, h.matched_chunks)
                       for h in ordered if h.candidate_id in byid]
            seen = {p.candidate_id for p, *_ in results}
            results += [(p, 0.0, "matched filters only", []) for p in gated
                        if p.candidate_id not in seen]
        else:
            excluded, caveats = [], {}
            results = [(p, 0.0, "", []) for p in sorted(
                filtered, key=lambda p: -p.quality.completeness)]
        latency = (time.perf_counter() - t0) * 1000
        st.session_state.last_latency_ms = latency
        st.query_params["q"] = query

        # Two UI-layer shortcuts on top of the already-computed `results`, never fed
        # back into retrieval/scoring:
        #  - a bare seniority code ("L3", "l4+") jumps straight to that level, since
        #    neither query parser has vocabulary for taxonomy codes typed on their own;
        #  - a name substring floats a known person to the top -- typing a name you
        #    already know ("pull up Chen's profile") is navigation, not screening, and
        #    CandidateProfile.searchable_text() never gives the scorer/embedder/BM25
        #    index a name to begin with, so this only reorders what is already there.
        # Both are skipped in blind-review mode, where identity and level are meant
        # to stay out of view.
        if query.strip() and not st.session_state.blind:
            qlow = query.strip().lower()
            boost_ids: set[str] = set()
            boost_label = ""
            m = re.fullmatch(r"l([1-7])(\+)?", qlow.replace(" ", ""))
            if m:
                lvl, at_least = int(m.group(1)), bool(m.group(2))
                for p, *_ in results:
                    if not p.seniority or not p.seniority.label.startswith("L"):
                        continue
                    try:
                        plvl = int(p.seniority.label[1:])
                    except ValueError:
                        continue
                    if plvl >= lvl if at_least else plvl == lvl:
                        boost_ids.add(p.candidate_id)
                boost_label = f"L{lvl}{'+' if at_least else ''} level match"
            else:
                boost_ids = {p.candidate_id for p, *_ in results
                             if qlow in p.display_name(False).lower()}
                boost_label = "name match"
            if boost_ids:
                boosted = [
                    (p, score, (f"{boost_label} — {explain}" if explain else boost_label), chunks)
                    for p, score, explain, chunks in results if p.candidate_id in boost_ids]
                rest = [t for t in results if t[0].candidate_id not in boost_ids]
                results = boosted + rest

        if "import_flash" in st.session_state:
            st.success(st.session_state.import_flash)
            del st.session_state["import_flash"]
        _resume_match_studio(results, pool, index, client, query)
        _pool_import_studio(client)
        if parsed:
            st.markdown(
                f'<div class="mm-banner"><b>Interpreted as</b> — {html.escape(parsed.interpretation)}'
                f' <span class="mm-mono">[{parsed.method} parser]</span></div>',
                unsafe_allow_html=True)

    shown = min(show_cap, len(results))
    results_col, insight_col = st.columns([0.70, 0.30], gap="medium",
                                          vertical_alignment="top")
    with results_col:
        _render_results_panel(facets, store, shown, results, pool, latency, query,
                              caveats, excluded)
    with insight_col:
        _render_search_action_cards(n_sl, vertical=True)
        query_llm_state = "enabled" if SETTINGS.flags.enable_llm_query_parse else "disabled"
        C.llm_callout(
            "Search intent understanding",
            f"Plain-English queries can use the LLM to extract must-haves, "
            f"preferences, and exclusions. Retrieval, filtering, and ranking stay "
            f"local; LLM query parsing is currently {query_llm_state}.",
            stage="query")
        _render_pool_glance(results, shown, latency, pool)


def _render_results_panel(facets, store, shown: int, results, pool, latency: float,
                          query: str, caveats: dict, excluded: list) -> None:
    results_panel = st.container(border=True, key="results_panel")
    with results_panel:
        _render_filter_bar(facets, store, n_shown=shown, n_pool=len(pool),
                           latency_ms=latency)
        hd_m, hd_r = st.columns([0.35, 0.65], vertical_alignment="center")
        with hd_m:
            if query.strip():
                st.button("Clear search", key="clear_query", width="stretch",
                          icon=":material/close:", on_click=_clear_query,
                          help="Remove the query and show the whole pool again "
                               "(filters stay applied)")
        with hd_r:
            view = st.segmented_control(
                "View", ["List", "Table", "Cards"], default="List",
                key="view", label_visibility="collapsed",
                help="List for a dense product scan; table for sorting; cards for flags.")
        if len(results) > shown:
            st.caption(f"showing {shown} of {len(results)} — raise \"Show\" in the "
                       f"command bar to see more")
        if not results:
            C.empty_state(
                "No candidates match",
                "Try removing a filter, or search in plain English instead — "
                "must-have terms gate, preferences only score.",
                icon="⌀")
        if view == "Table" and results:
            picked = _results_table(results[:shown])
            _results_actions(picked)
        if view == "List" and results:
            _render_result_list(results[:shown], shown, latency)
        for p, score, explain, chunks in (results[:shown] if view == "Cards" else []):
            st.markdown(C.candidate_card(p, st.session_state.blind,
                                         score if score else None, explain),
                        unsafe_allow_html=True)
            b = st.columns([0.22, 0.26, 0.22, 0.30])
            if b[0].button("Open", key=f"o_{p.candidate_id}",
                           icon=":material/open_in_new:"):
                st.session_state.selected = p.candidate_id
                st.session_state.page = "Candidate"
                st.rerun()
            listed = p.candidate_id in st.session_state.shortlist
            if b[1].button("Listed" if listed else "Shortlist",
                           key=f"s_{p.candidate_id}", icon=":material/star:"):
                if listed:
                    st.session_state.shortlist.pop(p.candidate_id)
                else:
                    st.session_state.shortlist[p.candidate_id] = {
                        "note": "", "tags": "", "source": "human", "basis": ""}
                st.rerun()
            if b[2].button("Delete", key=f"d_{p.candidate_id}",
                           icon=":material/delete:"):
                _remove_from_pool([p.candidate_id])
                st.rerun()
            with b[3].popover("Export", icon=":material/download:"):
                _profile_downloads_compact(p, prefix=f"card_{p.candidate_id[:8]}")
            if chunks:
                with st.expander(f"Why this matched · {explain}"):
                    for ch in chunks:
                        rk = " · ".join(filter(None, [
                            f"semantic #{ch['dense_rank']}" if ch.get("dense_rank") else "",
                            f"keyword #{ch['lexical_rank']}" if ch.get("lexical_rank") else ""]))
                        st.markdown(
                            f'<div class="mm-ev"><b>{html.escape(ch["label"])}</b> '
                            f'<span class="mm-sub">({ch["kind"]})</span><br>'
                            f'{html.escape(ch["text"])}<br>'
                            f'<span class="mm-sub mm-mono">{rk}</span></div>',
                            unsafe_allow_html=True)
        if results:
            _render_result_exports(results[:shown])

        if caveats:
            with st.expander(f"⚠ {len(caveats)} kept with an unverified must-have"):
                st.caption("Their CV does not state the requirement either way. Kept, "
                           "because unknown is not the same as unqualified.")
                byid_c = _byid(pool)
                for cid, notes in list(caveats.items())[:12]:
                    nm = byid_c[cid].display_name(st.session_state.blind) if cid in byid_c else cid
                    st.markdown(f"**{html.escape(nm)}** — "
                                + "; ".join(html.escape(n) for n in notes))
        if excluded:
            with st.expander(f"⊘ {len(excluded)} gated out by must-have requirements"):
                st.caption("Shown, never silently dropped — one over-strict requirement "
                           "is the usual reason a pool looks empty.")
                byid = _byid(pool)
                for e in excluded[:15]:
                    nm = byid[e["candidate_id"]].display_name(st.session_state.blind) \
                        if e["candidate_id"] in byid else e["candidate_id"]
                    st.markdown(f"**{html.escape(nm)}** — " +
                                "; ".join(html.escape(r) for r in e["reasons"]))


def _render_result_list(results, shown: int, latency: float) -> None:
    from . import cc_surfaces

    def _list_action(payload: dict) -> None:
        cid = payload.get("id") if isinstance(payload, dict) else None
        kind = payload.get("type") if isinstance(payload, dict) else None
        if not cid:
            return
        if kind == "open":
            st.session_state.selected = cid
            st.session_state.page = "Candidate"
        elif kind == "shortlist":
            if cid in st.session_state.shortlist:
                st.session_state.shortlist.pop(cid)
            else:
                st.session_state.shortlist[cid] = {
                    "note": "", "tags": "", "source": "human", "basis": ""}

    rows = []
    for p, score, explain, _chunks in results:
        cur = p.current_role()
        tags = []
        if p.geo_region:
            tags.append({"label": tx.display("region", p.geo_region.label,
                                             p.geo_region.label), "tone": "muted"})
        if p.seniority:
            tags.append({"label": p.seniority.label, "tone": ""})
        if p.strategies:
            tags.append({"label": tx.display("strategy", p.strategies[0].label,
                                             p.strategies[0].label), "tone": ""})
        if p.quality.needs_human_review:
            tags.append({"label": "review", "tone": "warn"})
        name = p.display_name(st.session_state.blind)
        initials = "".join(w[0] for w in name.split()[:2] if w) or "?"
        subtitle = " · ".join(filter(None, [
            (cur.title_raw.display("") if cur else ""),
            (cur.employer_canonical or cur.employer_raw.display("")) if cur else "",
        ]))
        rows.append({
            "id": p.candidate_id,
            "name": name,
            "initials": initials.upper(),
            "subtitle": subtitle or explain or "Profile in pool",
            "tags": tags[:4],
            "score": f"{score:.3f}" if score else "",
            "listed": p.candidate_id in st.session_state.shortlist,
        })
    cc_surfaces.candidate_list(
        rows, title=f"{shown} candidates", meta=f"{latency:.0f} ms",
        key="cc_candidate_list", on_action=_list_action)


def _render_result_exports(results) -> None:
    from millennium.export import flat_row

    if not results:
        return
    with st.expander("Export results and profiles", expanded=False):
        visible = [p for p, *_ in results]
        include_pii = not st.session_state.blind
        exclude_fields = {"raw_text"} if include_pii else {"raw_text", "sensitive"}
        payload = {
            "count": len(visible),
            "blind_mode": st.session_state.blind,
            "candidates": [
                json.loads(p.model_dump_json(exclude=exclude_fields)) for p in visible
            ],
        }
        c1, c2 = st.columns(2)
        c1.download_button(
            "Visible results JSON",
            json.dumps(payload, indent=1, ensure_ascii=False),
            "visible_search_results.json",
            "application/json",
            icon=":material/data_object:",
            key="search_visible_json",
            width="stretch")
        c2.download_button(
            "Visible results CSV",
            pd.DataFrame([flat_row(p, include_pii=include_pii) for p in visible])
            .to_csv(index=False),
            "visible_search_results.csv",
            "text/csv",
            icon=":material/table:",
            key="search_visible_csv",
            width="stretch")
        choices = {
            f"{p.display_name(st.session_state.blind)} · {p.candidate_id[:6]}": p
            for p in visible
        }
        pick = st.selectbox(
            "Single profile export",
            list(choices),
            key="search_export_profile_pick")
        picked = choices.get(pick, visible[0])
        _profile_downloads_compact(picked, prefix="search_pick")


def _render_pool_glance(results, shown: int, latency: float, pool) -> None:
    with st.container(border=True, key="pool_glance"):
        C.section_header(
            "Pool at a glance",
            subtitle="Distribution of candidates currently in view.")
        m1, m2 = st.columns(2)
        C.metric_card(shown, "visible", f"of {len(results)} matched")
        with m1:
            C.metric_card(len(st.session_state.shortlist), "shortlisted")
        with m2:
            C.metric_card(review_n := sum(1 for p, *_ in results if p.quality.needs_human_review),
                          "review", colour=theme.WARNING if review_n else theme.SUCCESS)
        l1, l2 = st.columns(2)
        with l1:
            C.metric_card(f"{latency:.0f}", "ms")
        with l2:
            C.metric_card(len(pool), "pool")
        _mini_charts(results)
        _right_rail_focus(results[:shown])


def _render_search_action_cards(n_sl: int, *, vertical: bool = False) -> None:
    cards = [st.container(), st.container(), st.container()] if vertical else st.columns(
        [0.34, 0.33, 0.33], gap="medium")
    match_active = _flag_on("match_studio_open")
    import_active = _flag_on("import_studio_open")

    with cards[0], st.container(border=True, key="search_action_match"):
        st.markdown(
            '<div class="mm-action-card-head">'
            '<span class="mm-action-icon">person_search</span>'
            '<div><b>Match to JD</b><small>Paste a mandate, score candidates, shortlist top fits.</small></div>'
            '</div>'
            f'<div class="mm-action-card-state">{"Open now" if match_active else "Recommended next step"}</div>',
            unsafe_allow_html=True)
        st.button("Match to JD", icon=":material/person_search:",
                  key="open_match_studio",
                  type="primary" if match_active else "secondary",
                  on_click=_toggle_studio, args=("match",), width="stretch",
                  help="Score resumes against a job description and shortlist the top matches.")

    with cards[1], st.container(border=True, key="search_action_import"):
        st.markdown(
            '<div class="mm-action-card-head">'
            '<span class="mm-action-icon">upload_file</span>'
            '<div><b>Import profiles</b><small>Add PDFs, Word resumes, JSON, or CSV into this pool.</small></div>'
            '</div>'
            f'<div class="mm-action-card-state">{"Open now" if import_active else "Fast pool expansion"}</div>',
            unsafe_allow_html=True)
        st.button("Import", icon=":material/upload_file:",
                  key="open_import_studio",
                  type="primary" if import_active else "secondary",
                  on_click=_toggle_studio, args=("import",), width="stretch",
                  help="Add PDF, Word, JSON, or CSV profiles to the pool.")

    with cards[2], st.container(border=True, key="search_action_shortlist"):
        st.markdown(
            '<div class="mm-action-card-head">'
            '<span class="mm-action-icon">star</span>'
            '<div><b>Shortlist</b><small>Review the slate, add notes, export profiles, draft outreach.</small></div>'
            '</div>'
            f'<div class="mm-action-card-state">{n_sl} selected candidate{"s" if n_sl != 1 else ""}</div>',
            unsafe_allow_html=True)
        if st.button(f"Open Shortlist ({n_sl})" if n_sl else "Shortlist",
                     icon=":material/star:", key="jump_shortlist_from_search",
                     disabled=not n_sl, width="stretch",
                     help="Open the Shortlist tab."):
            st.session_state.page = "Shortlist"
            st.rerun()


def _right_rail_focus(results) -> None:
    if not results:
        return

    st.markdown('<div class="mm-rail-sec">Focus queue</div>', unsafe_allow_html=True)
    for p, score, explain, _chunks in results[:4]:
        cur = p.current_role()
        role = " · ".join(filter(None, [
            cur.title_raw.display("") if cur else "",
            (cur.employer_canonical or cur.employer_raw.display("")) if cur else "",
        ]))
        badges = []
        if score:
            badges.append(f"{score:.3f}")
        if p.quality.needs_human_review:
            badges.append("review")
        if p.candidate_id in st.session_state.shortlist:
            badges.append("listed")
        badge_html = "".join(
            f'<span class="mm-rail-pill">{html.escape(b)}</span>' for b in badges[:3])
        st.markdown(
            f'<div class="mm-rail-person">'
            f'<div class="mm-row" style="justify-content:space-between">'
            f'<span class="mm-name">{html.escape(p.display_name(st.session_state.blind))}</span>'
            f'<span>{badge_html}</span></div>'
            f'<div class="mm-sub">{html.escape(role or explain or "Profile in pool")}</div>'
            f'</div>',
            unsafe_allow_html=True)
        if st.button("Open profile", key=f"rail_open_{p.candidate_id}",
                     icon=":material/open_in_new:", width="stretch"):
            st.session_state.selected = p.candidate_id
            st.session_state.page = "Candidate"
            st.rerun()

    a, b = st.columns(2)
    if a.button("Analytics", icon=":material/monitoring:", key="rail_go_analytics",
                width="stretch"):
        st.session_state.page = "Analytics"
        st.rerun()
    if b.button("Review", icon=":material/flag:", key="rail_go_review",
                width="stretch"):
        st.session_state.page = "Review"
        st.rerun()


def _results_table(results) -> list[str]:
    """Dense, sortable results grid. Tick rows to shortlist, open, or delete.

    Cards are good for scanning labels; a table is what a recruiter actually works in
    when comparing twenty people on the same six dimensions. Sorting is native (click
    a header). Row ticks feed the action bar under the table — Open / Shortlist /
    Delete — rather than navigating away on a single click, so bulk actions stay
    possible.
    """
    rows = []
    for p, score, explain, _chunks in results:
        cur = p.current_role()
        rows.append({
            "Candidate": p.display_name(st.session_state.blind),
            "Score": round(score, 4) if score else None,
            "Region": tx.display("region", p.geo_region.label, "—") if p.geo_region else "—",
            "Yrs": p.years_experience.value if p.years_experience.is_known else None,
            "Level": p.seniority.label if p.seniority else "—",
            "Current role": (cur.title_raw.display("—") if cur else "—"),
            "Employer": (cur.employer_canonical or cur.employer_raw.display("—")) if cur else "—",
            "Tier": tx.display("tier", cur.employer_tier) if cur else "—",
            "Strategies": ", ".join(tx.display("strategy", c.label) for c in p.strategies[:2]),
            "Sectors": ", ".join(tx.display("sector", c.label) for c in p.sectors[:2]),
            "Approach": p.quant_fundamental.label if p.quant_fundamental else "—",
            "Complete": p.quality.completeness,
            "Review": "⚑" if p.quality.needs_human_review else "",
            "Match": explain,
            "_id": p.candidate_id,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    # Browsing without a query: every Score is None and every Match is empty, and a
    # column of "None" reads as "search is broken". The columns only exist when a
    # query actually ranked the rows.
    if df["Score"].isna().all():
        df = df.drop(columns=["Score"])
        if (df["Match"] == "").all():
            df = df.drop(columns=["Match"])
    sel = st.dataframe(
        df.drop(columns=["_id"]), width="stretch", hide_index=True, height=460,
        on_select="rerun", selection_mode="multi-row", key="results_table",
        column_config={
            "Score": st.column_config.NumberColumn(format="%.4f", width="small"),
            "Yrs": st.column_config.NumberColumn(
                format="%.1f", width="small",
                help="Blank means the total could not be derived from verified dates — "
                     "unknown, not zero."),
            "Complete": st.column_config.ProgressColumn(
                format="%.0f%%", min_value=0.0, max_value=1.0, width="small"),
            "Review": st.column_config.TextColumn(width="small",
                                                  help="flagged for human review"),
            "Match": st.column_config.TextColumn(width="medium"),
        })
    picked = (sel.selection.rows or []) if hasattr(sel, "selection") else []
    if not picked:
        st.caption("Click a column header to sort · tick rows to shortlist, open, or "
                   "delete · a blank Yrs cell means unknown, never zero")
        return []
    return [str(df.iloc[i]["_id"]) for i in picked]


def _mini_charts(results) -> None:
    """Compact distribution bars for the 'Pool at a glance' panel.

    Hand-rolled HTML rows instead of three separate Plotly figures: each Plotly chart
    carried its own title block, margins and inconsistent bar heights, which made the
    panel read as three stacked mini-dashboards rather than one summary. These rows
    share one typographic system with the rest of the app, cost nothing to render,
    and reflow cleanly at any panel width (the panel is user-resizable).
    """
    if not results:
        st.caption("No candidates in view.")
        return
    rows = []
    for p, *_ in results:
        rows.append({
            "region": tx.display("region", p.geo_region.label, "Unknown") if p.geo_region else "Unknown",
            "sector": tx.display("sector", p.sectors[0].label, "Unknown") if p.sectors else "Unknown",
            "seniority": (f"{p.seniority.label} · {tx.display('seniority', p.seniority.label)}"
                          if p.seniority else "Unknown"),
        })
    df = pd.DataFrame(rows)
    total = len(rows)
    cols = st.columns(3, gap="large")
    for slot, (col, title) in zip(cols, (("region", "Region"),
                                         ("sector", "Primary sector"),
                                         ("seniority", "Seniority"))):
        vc = df[col].value_counts()
        top = int(vc.max())
        parts = [f'<div class="mm-glance-sec">{html.escape(title)}</div>']
        for label, n in vc.items():
            pct_of_max = 100.0 * n / top
            share = 100.0 * n / total
            parts.append(
                f'<div class="mm-glance-row" title="{html.escape(str(label))} — '
                f'{n} of {total} ({share:.0f}%)">'
                f'<span class="lbl">{html.escape(str(label))}</span>'
                f'<span class="track"><span class="fill" style="width:{pct_of_max:.1f}%">'
                f'</span></span>'
                f'<span class="n">{n}</span></div>')
        slot.markdown("".join(parts), unsafe_allow_html=True)


def _profile_exports(p) -> None:
    """Four download buttons — JSON / CSV / PDF / Word — for this one profile."""
    from millennium.export import (
        profile_csv_bytes, profile_docx_bytes, profile_json_bytes,
        profile_pdf_bytes, profile_filename,
    )

    include_pii = not st.session_state.blind
    cid = p.candidate_id[:8]
    with st.container(key="profile_exports"):
        st.markdown(
            '<div class="mm-sub" style="margin-bottom:4px">Download this profile</div>',
            unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        # Stable container keys (not the per-candidate widget keys) so theme.py can
        # colour each format without a CSS wildcard on the changing id suffix.
        with c1, st.container(key="dl_json"):
            st.download_button(
                "JSON", profile_json_bytes(p, include_pii=include_pii),
                file_name=profile_filename(p, "json", st.session_state.blind),
                mime="application/json",
                icon=":material/data_object:",
                key=f"dl_json_{cid}", width="stretch",
                help="Full structured record, every field with evidence")
        with c2, st.container(key="dl_csv"):
            st.download_button(
                "CSV", profile_csv_bytes(p, include_pii=include_pii),
                file_name=profile_filename(p, "csv", st.session_state.blind),
                mime="text/csv",
                icon=":material/table:",
                key=f"dl_csv_{cid}", width="stretch",
                help="One flattened row for Excel. Name/email/phone omitted "
                     "in blind-review mode.")
        with c3, st.container(key="dl_pdf"):
            st.download_button(
                "PDF", profile_pdf_bytes(p, include_pii=include_pii),
                file_name=profile_filename(p, "pdf", st.session_state.blind),
                mime="application/pdf",
                icon=":material/picture_as_pdf:",
                key=f"dl_pdf_{cid}", width="stretch",
                help="Printable one-pager of the extracted profile")
        with c4, st.container(key="dl_docx"):
            st.download_button(
                "Word", profile_docx_bytes(p, include_pii=include_pii),
                file_name=profile_filename(p, "docx", st.session_state.blind),
                mime="application/vnd.openxmlformats-officedocument."
                     "wordprocessingml.document",
                icon=":material/description:",
                key=f"dl_docx_{cid}", width="stretch",
                help="Editable .docx of the extracted profile")


def _profile_downloads_compact(p, *, prefix: str = "dlc") -> None:
    from millennium.export import (
        profile_csv_bytes, profile_docx_bytes, profile_json_bytes,
        profile_pdf_bytes, profile_filename,
    )

    include_pii = not st.session_state.blind
    cid = p.candidate_id[:8]
    st.markdown('<div class="mm-console-title">Exports</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1, st.container(key=f"{prefix}_json"):
        st.download_button(
            "JSON", profile_json_bytes(p, include_pii=include_pii),
            file_name=profile_filename(p, "json", st.session_state.blind),
            mime="application/json", icon=":material/data_object:",
            key=f"{prefix}_json_{cid}", width="stretch")
    with c2, st.container(key=f"{prefix}_csv"):
        st.download_button(
            "CSV", profile_csv_bytes(p, include_pii=include_pii),
            file_name=profile_filename(p, "csv", st.session_state.blind),
            mime="text/csv", icon=":material/table:",
            key=f"{prefix}_csv_{cid}", width="stretch")
    c3, c4 = st.columns(2)
    with c3, st.container(key=f"{prefix}_pdf"):
        st.download_button(
            "PDF", profile_pdf_bytes(p, include_pii=include_pii),
            file_name=profile_filename(p, "pdf", st.session_state.blind),
            mime="application/pdf", icon=":material/picture_as_pdf:",
            key=f"{prefix}_pdf_{cid}", width="stretch")
    with c4, st.container(key=f"{prefix}_docx"):
        st.download_button(
            "Word", profile_docx_bytes(p, include_pii=include_pii),
            file_name=profile_filename(p, "docx", st.session_state.blind),
            mime="application/vnd.openxmlformats-officedocument."
                 "wordprocessingml.document",
            icon=":material/description:", key=f"{prefix}_docx_{cid}",
            width="stretch")


def _render_ai_json_panel(data: dict, title: str) -> None:
    if data.get("_notice"):
        st.info(data["_notice"])
    source = "Uses LLM: generated" if data.get("_llm_generated") else "LLM feature: fallback"
    st.markdown(
        f'<div class="mm-ai-panel-head"><span>{html.escape(title)}</span>'
        f'<b>{html.escape(source)}</b></div>',
        unsafe_allow_html=True)
    for key, value in data.items():
        if key.startswith("_"):
            continue
        label = key.replace("_", " ").title()
        if isinstance(value, list):
            st.markdown(f"**{label}**")
            if value and isinstance(value[0], dict):
                for item in value[:8]:
                    bits = [f"**{html.escape(str(k).replace('_', ' ').title())}:** "
                            f"{html.escape(str(v))}" for k, v in item.items() if v not in ("", [], None)]
                    st.markdown("- " + " · ".join(bits) if bits else "-")
            else:
                for item in value[:8]:
                    st.markdown(f"- {html.escape(str(item))}")
        elif value not in ("", None, []):
            st.markdown(f"**{label}**  \n{html.escape(str(value))}")


def _render_candidate_ai_brief(client, p, pool) -> None:
    key = f"candidate_ai_brief_{p.candidate_id}"
    C.llm_callout(
        "Candidate brief",
        "Creates recruiter-facing strengths, watchouts, and next actions from the "
        "verified profile. It runs only after you click.",
        stage="candidate_brief")
    if st.button("Generate LLM brief", icon=":material/auto_awesome:",
                 key=f"cand_ai_brief_btn_{p.candidate_id}", width="stretch"):
        st.session_state[key] = llm_panels.candidate_brief(
            client, p, pool, blind=st.session_state.blind)
    if st.session_state.get(key):
        _render_ai_json_panel(st.session_state[key], "LLM recruiter brief")


def _profile_insights(p, pool) -> None:
    """Compact charts on the Profile tab so opening a record shows *work done*,
    not only fields: skill depth, quality coverage, and how this person sits in
    the pool on years of experience."""
    import plotly.graph_objects as go

    n_core = sum(1 for s in p.skills if s.depth == "core")
    n_applied = sum(1 for s in p.skills if s.depth == "applied")
    n_ment = sum(1 for s in p.skills if s.depth == "mentioned")
    strat_n = len(p.strategies)
    sect_n = len(p.sectors)
    roles_n = len(p.employment)
    pool_years = [q.years_experience.value for q in pool
                  if q.years_experience.is_known]
    y = p.years_experience.value if p.years_experience.is_known else None
    median_y = sorted(pool_years)[len(pool_years) // 2] if pool_years else None

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="mm-insight-h">Skill depth</div>'
                    '<div class="mm-sub">How skills are used, not just listed</div>',
                    unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=[n_core, n_applied, n_ment],
            y=["Core", "Applied", "Mentioned"],
            orientation="h", marker_color=[theme.ACCENT, "#1D4ED8", "#94A3B8"],
            text=[n_core, n_applied, n_ment], textposition="outside",
            cliponaxis=False))
        fig.update_layout(height=150, margin=dict(l=0, r=28, t=8, b=0),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(visible=False), yaxis=dict(title=None),
                          font=dict(size=11), showlegend=False)
        st.plotly_chart(theme.polish_fig(fig), width="stretch", config={"displayModeBar": False},
                        key="insight_skills")
    with c2:
        st.markdown('<div class="mm-insight-h">Extraction quality</div>'
                    '<div class="mm-sub">What the pipeline actually proved</div>',
                    unsafe_allow_html=True)
        comp = p.quality.completeness
        ev = p.quality.evidence_coverage
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[comp, ev], y=["Complete", "Evidenced"],
                             orientation="h",
                             marker_color=[theme.ACCENT, "#0E7490"],
                             text=[f"{comp:.0%}", f"{ev:.0%}"],
                             textposition="outside", cliponaxis=False))
        fig.update_layout(height=150, margin=dict(l=0, r=48, t=8, b=0),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(range=[0, 1.15], visible=False),
                          yaxis=dict(title=None), font=dict(size=11), showlegend=False)
        st.plotly_chart(theme.polish_fig(fig), width="stretch", config={"displayModeBar": False},
                        key="insight_quality")
    with c3:
        st.markdown('<div class="mm-insight-h">In this pool</div>'
                    '<div class="mm-sub">Roles, coverage, experience vs median</div>',
                    unsafe_allow_html=True)
        vs = ""
        if y is not None and median_y is not None:
            delta = y - median_y
            vs = (f'{y:.1f}y exp · {delta:+.1f} vs pool median {median_y:.1f}y')
        elif y is not None:
            vs = f"{y:.1f} years experience"
        else:
            vs = "Years of experience unknown"
        st.markdown(
            f'<div class="mm-insight-stats">'
            f'<div><b>{roles_n}</b><span>roles extracted</span></div>'
            f'<div><b>{strat_n}</b><span>strategies</span></div>'
            f'<div><b>{sect_n}</b><span>sectors</span></div>'
            f'</div>'
            f'<div class="mm-sub" style="margin-top:8px">{html.escape(vs)}</div>',
            unsafe_allow_html=True)
        if y is not None and pool_years:
            cap = max(max(pool_years), y, 1)
            pct = 100.0 * y / cap
            st.markdown(
                f'<div class="mm-glance-row" title="Experience vs longest in pool">'
                f'<span class="lbl">Exp</span>'
                f'<span class="track"><span class="fill" style="width:{pct:.1f}%">'
                f'</span></span>'
                f'<span class="n">{y:.0f}y</span></div>',
                unsafe_allow_html=True)


def _render_candidate_command_center(p, ids: list[str], byid: dict, i: int,
                                     pool, client) -> None:
    name = p.display_name(st.session_state.blind)
    initials = "".join(part[0] for part in name.replace("(", " ").split()
                       if part[:1].isalpha())[:2].upper() or "·"
    cur = p.current_role()
    current = ""
    if cur:
        current = " · ".join(filter(None, [
            cur.title_raw.display(""),
            cur.employer_canonical or cur.employer_raw.display(""),
        ]))
    review = p.quality.needs_human_review
    status = "Human review" if review else "Ready for recruiter review"
    status_tone = "warn" if review else "ok"
    listed = p.candidate_id in st.session_state.shortlist

    with st.container(border=True, key="candidate_command"):
        profile_col, console_col = st.columns([0.38, 0.62], gap="medium",
                                              vertical_alignment="top")
        with profile_col:
            st.markdown(
                f'<div class="mm-profile-command-head">'
                f'<div class="mm-cand-avatar mm-hud-avatar">{html.escape(initials)}</div>'
                f'<div class="mm-profile-command-body">'
                f'<div class="mm-profile-name-line">'
                f'<div class="mm-hud-name">{html.escape(name)}</div>'
                f'<div class="mm-profile-status {status_tone}">{html.escape(status)}</div>'
                f'</div>'
                f'<div class="mm-profile-role">{html.escape(current or p.headline.display(""))}</div>'
                f'</div></div>',
                unsafe_allow_html=True)
        with console_col:
            st.markdown(
                f'<div class="mm-console-title">Profile controls · {i + 1} of {len(ids)}</div>',
                unsafe_allow_html=True)
            nav_col, short_col, export_col = st.columns(
                [0.62, 0.20, 0.18], gap="small", vertical_alignment="bottom")
            with nav_col:
                n1, n2, n3 = st.columns([0.16, 0.68, 0.16], vertical_alignment="bottom")
                with n1:
                    st.button("Prev", icon=":material/chevron_left:", key="cand_prev",
                              width="stretch", disabled=i == 0,
                              on_click=_step_candidate, args=(-1,),
                              help="Open the previous candidate")
                with n2:
                    st.selectbox(
                        "Switch candidate", ids, index=i, key="cand_switch_box",
                        on_change=_sync_candidate_from_switch,
                        label_visibility="collapsed",
                        format_func=lambda cid: byid[cid].display_name(st.session_state.blind),
                        help="Jump to another parsed profile")
                with n3:
                    st.button("Next", icon=":material/chevron_right:", key="cand_next",
                              width="stretch", disabled=i >= len(ids) - 1,
                              on_click=_step_candidate, args=(1,),
                              help="Open the next candidate")
            with short_col:
                if listed:
                    if st.button("Remove", icon=":material/star:", key="cand_shortlist",
                                 width="stretch",
                                 help="Remove this person from the shortlist"):
                        st.session_state.shortlist.pop(p.candidate_id)
                        st.rerun()
                    if st.button("Open list", icon=":material/open_in_new:",
                                 key="cand_go_sl", width="stretch"):
                        st.session_state.page = "Shortlist"
                        st.rerun()
                else:
                    if st.button("Shortlist", icon=":material/star:", type="primary",
                                 key="cand_shortlist", width="stretch",
                                 help="Add this person to the shortlist"):
                        st.session_state.shortlist[p.candidate_id] = {
                            "note": "", "tags": "", "source": "human",
                            "basis": "Manually shortlisted from the candidate profile"}
                        st.rerun()
            with export_col:
                with st.popover("Export", icon=":material/download:"):
                    _profile_downloads_compact(p, prefix=f"cand_{p.candidate_id[:8]}")
                    st.divider()
                    if st.button("Delete profile", icon=":material/delete:", key="cand_remove",
                                 width="stretch",
                                 help="Remove this profile from this session"):
                        _remove_from_pool([st.session_state.selected])
                        st.session_state.page = "Search"
                        st.rerun()


def _render_profile_tab(p, pool, client) -> None:
    source = p.provenance.source_file if p.provenance else "source document"
    y = f"{p.years_experience.value:.1f}" if p.years_experience.is_known else "—"
    detail_col, source_col = st.columns([0.58, 0.42], gap="large",
                                        vertical_alignment="top")
    with detail_col:
        with st.container(border=True, key="profile_detail_panel"):
            st.markdown(
                '<div class="mm-profile-panel-title">Agent-extracted profile</div>'
                '<div class="mm-profile-panel-sub">Verified fields, work history, '
                'skills, and education in the order a reviewer needs them.</div>',
                unsafe_allow_html=True)
            st.markdown(
                '<div class="mm-profile-body-snapshot">'
                f'<span>Source: {html.escape(source)}</span>'
                f'<span>{html.escape(y)}y exp</span>'
                f'<span>{p.quality.completeness:.0%} complete</span>'
                f'<span>{p.quality.evidence_coverage:.0%} evidenced</span>'
                f'<span>{p.quality.abstention_count} abstained</span>'
                '</div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="mm-profile-labels mm-profile-body-labels">'
                f'{C.labels_row(p, 10)}</div>',
                unsafe_allow_html=True)
            if p.summary.is_known:
                st.markdown(
                    f'<div class="mm-candidate-summary">'
                    f'{html.escape(str(p.summary.display("")))}</div>',
                    unsafe_allow_html=True)

            a, b = st.columns([0.52, 0.48], gap="medium")
            with a:
                st.markdown("**Identity & contact**")
                for t, lbl in ((p.sensitive.full_name, "Name"),
                               (p.sensitive.email, "Email"),
                               (p.sensitive.phone, "Phone"),
                               (p.location_current, "Location"),
                               (p.work_authorization, "Work authorisation")):
                    if st.session_state.blind and lbl in ("Name", "Email", "Phone"):
                        st.markdown(
                            f'<div class="mm-row"><span class="mm-sub">{lbl}</span>'
                            f'<span style="color:{theme.MUTED}">masked '
                            f'(blind review)</span></div>',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(C.tracked_value(t, lbl), unsafe_allow_html=True)
                st.markdown(C.tracked_value(p.years_experience, "Total experience"),
                            unsafe_allow_html=True)
                st.markdown(C.tracked_value(p.years_relevant_experience,
                                            "Investment-relevant"),
                            unsafe_allow_html=True)
                if p.certifications:
                    st.markdown("**Certifications**")
                    st.markdown("".join(
                        theme.chip(tx.display("certification", c.canonical)
                                   + (f" · {c.status}" if c.status else ""), "verified")
                        for c in p.certifications if c.canonical), unsafe_allow_html=True)
                if p.languages:
                    st.markdown("**Languages**")
                    st.markdown("".join(
                        theme.chip(f"{l.language}"
                                   + (f" · {l.proficiency}" if l.proficiency else ""))
                        for l in p.languages), unsafe_allow_html=True)
            with b:
                st.markdown("**Skills**")
                st.caption("Grouped by how strongly the document supports each skill.")
                for depth, tone in (("core", "verified"), ("applied", "derived"),
                                    ("mentioned", "missing")):
                    group = [s for s in p.skills if s.depth == depth]
                    if group:
                        st.markdown(f'<span class="mm-sub">{depth}</span><br>'
                                    + "".join(theme.chip(s.canonical, tone)
                                              for s in group),
                                    unsafe_allow_html=True)

            st.divider()
            st.markdown("**Employment**")
            _render_employment_flow(p)
            st.markdown("**Education**")
            edu_rows = [{"Institution": e.institution.display("—"),
                         "Degree": e.degree_raw.display("—"),
                         "Level": e.degree_level or "—",
                         "Field": e.field_of_study.display("—"),
                         "Year": e.graduation_year.display("—"),
                         "Result": e.gpa_raw.display("—")}
                        for e in p.education]
            if edu_rows:
                st.dataframe(pd.DataFrame(edu_rows), width="stretch", hide_index=True)
            else:
                st.caption("No education entries were extracted.")

    with source_col:
        with st.container(border=True, key="profile_source_panel"):
            st.markdown(
                '<div class="mm-profile-panel-title">Original document</div>'
                '<div class="mm-profile-panel-sub">The uploaded file sits beside '
                'the extracted record for direct inspection.</div>',
                unsafe_allow_html=True)
            C.original_document_view(p, height=520)

        with st.container(border=True, key="profile_trust_panel"):
            st.markdown(
                '<div class="mm-profile-panel-title">Trust & review</div>'
                '<div class="mm-profile-panel-sub">How the profile was produced and '
                'what needs attention.</div>',
                unsafe_allow_html=True)
            C.provenance_banner(p)
            if p.quality.validation_flags:
                st.markdown(theme.flag_list(p.quality.validation_flags[:3]),
                            unsafe_allow_html=True)
                if len(p.quality.validation_flags) > 3:
                    with st.expander(f"{len(p.quality.validation_flags) - 3} more flag(s)"):
                        st.markdown(theme.flag_list(p.quality.validation_flags[3:]),
                                    unsafe_allow_html=True)
            else:
                st.success("No validation flags on this record.")
            with st.expander("Candidate brief · LLM", expanded=False):
                _render_candidate_ai_brief(client, p, pool)

    C.section_break("Profile intelligence", 1)
    _profile_insights(p, pool)


def _render_employment_flow(p) -> None:
    flow_parts: list[str] = ['<div class="mm-flow">']
    for e in p.employment:
        dates = f"{e.dates.start.normalized_value or '?'} → {e.dates.end.normalized_value or '?'}"
        if not e.dates.start.is_known and e.dates.duration_months.is_known:
            dates = (f"{e.dates.duration_months.value} months stated · "
                     f"absolute dates unknown")
        tier = tx.display("tier", e.employer_tier, "")
        intern_cls = " is-intern" if e.is_internship else ""
        meta = " · ".join(x for x in [
            e.employer_canonical or e.employer_raw.display("—"),
            tier if tier and tier != "Unknown" else "",
            e.location.display("") if e.location.is_known else "",
            f"L{e.seniority_level}" if e.seniority_level else "",
            "internship" if e.is_internship else "",
        ] if x)
        highlights = ""
        if e.highlights:
            items = "".join(f"<li>{html.escape(str(h.value))}</li>"
                            for h in e.highlights)
            highlights = (
                f'<details open><summary>{len(e.highlights)} grounded '
                f'highlight(s)</summary><ul>{items}</ul></details>')
        flow_parts.append(
            f'<div class="mm-flow-item{intern_cls}">'
            f'<div class="mm-row" style="justify-content:space-between">'
            f'<span class="mm-name">{html.escape(e.title_raw.display("—"))}</span>'
            f'<span class="mm-sub mm-mono">{html.escape(dates)}</span></div>'
            f'<div class="mm-flow-meta">{html.escape(meta)}</div>'
            f'{highlights}</div>')
    flow_parts.append("</div>")
    if p.employment:
        st.markdown("".join(flow_parts), unsafe_allow_html=True)
    else:
        st.caption("No employment entries were extracted.")


# ========================================================================= CANDIDATE
def render_candidate(profiles, synth, pool, index, index_manifest, manifest, store,
                     client, bench, evals):
    byid = _byid(pool)
    ids = list(byid)
    if st.session_state.selected not in byid:
        st.session_state.selected = ids[0] if ids else None
    if not st.session_state.selected:
        st.info("No candidate selected.")
        return
    # Prev/Next callbacks read this list; keep it current every render.
    st.session_state["_cand_ids"] = ids
    # Sync the selectbox widget key FROM `selected` before it is instantiated, so
    # arriving from Search (or a Prev/Next click) is not overwritten by stale
    # widget state. Same pattern as `nav_radio` in app.py.
    if ("cand_switch_box" not in st.session_state
            or st.session_state["cand_switch_box"] != st.session_state.selected):
        st.session_state.cand_switch_box = st.session_state.selected
    p = byid[st.session_state.selected]
    i = ids.index(st.session_state.selected)

    if p.provenance and p.provenance.is_synthetic:
        st.markdown('<div class="mm-synth">SYNTHETIC RECORD — generated for benchmarking, '
                    'not a real candidate.</div>', unsafe_allow_html=True)
    if p.provenance and p.provenance.injection_flags:
        st.markdown(
            f'<div class="mm-danger"><b>Prompt injection detected and neutralised.</b> '
            f'Categories: {html.escape(", ".join(p.provenance.injection_flags))}. '
            f'The payload was stripped before the document reached the model, the call '
            f'was issued with no tools, and every surviving field was independently '
            f'span-verified. The profile below is unaffected.</div>',
            unsafe_allow_html=True)
    if p.provenance and p.provenance.near_duplicate_of:
        st.markdown(f'<div class="mm-warn"><b>Near-duplicate</b> of '
                    f'{html.escape(", ".join(p.provenance.near_duplicate_of))} — likely the '
                    f'same person submitted through a different source.</div>',
                    unsafe_allow_html=True)

    _render_candidate_command_center(p, ids, byid, i, pool, client)

    tabs = st.tabs(["Profile", "Evidence Board", "Career Map", "Lookalikes",
                    "Lineage", "Source Text"])

    with tabs[0]:
        _render_profile_tab(p, pool, client)

    with tabs[1]:
        st.caption("Click any field to see the exact text it came from. This is the "
                   "whole point of the system: nothing here is asserted without a span.")
        fields = {"Name": p.sensitive.full_name, "Email": p.sensitive.email,
                  "Phone": p.sensitive.phone, "Headline": p.headline,
                  "Summary": p.summary, "Location": p.location_current}
        for e in p.employment[:6]:
            fields[f"Employer — {e.employer_raw.display('?')}"] = e.employer_raw
            fields[f"Title — {e.title_raw.display('?')}"] = e.title_raw
        for e in p.education[:4]:
            fields[f"Institution — {e.institution.display('?')}"] = e.institution
        pick = st.selectbox("Field", list(fields))
        t = fields[pick]
        st.markdown(C.tracked_value(t, pick), unsafe_allow_html=True)
        if t.notes:
            for n in t.notes:
                st.caption(f"· {n}")
        C.evidence_for(t, p)
        st.divider()
        abst = [(k, v) for k, v in fields.items() if not v.is_known
                and v.validation_status == "abstained"]
        if abst:
            st.markdown("**Abstained fields** — a value was proposed and discarded")
            for k, v in abst:
                st.markdown(f'{theme.status_chip("abstained")} **{html.escape(k)}** — '
                            f'{html.escape(v.notes[0] if v.notes else "")}',
                            unsafe_allow_html=True)

    with tabs[2]:
        _timeline(p)

    with tabs[3]:
        sim = similar_candidates(index, p, 5).output or []
        if not sim:
            st.caption("No similar candidates found.")
        for cid, score in sim:
            if cid in byid:
                st.markdown(C.candidate_card(byid[cid], st.session_state.blind, score),
                            unsafe_allow_html=True)

    with tabs[4]:
        pv = p.provenance
        if pv:
            st.json({"source_file": pv.source_file, "file_type": pv.file_type,
                     "page_count": pv.page_count, "file_sha256": pv.file_sha256[:24] + "…",
                     "text_sha256": pv.text_sha256[:24] + "…", "extractor": pv.extractor,
                     "ingested_at": pv.ingested_at, "llm_model": pv.llm_model,
                     "schema_version": pv.schema_version,
                     "taxonomy_version": pv.taxonomy_version,
                     "pipeline_run_id": pv.pipeline_run_id, "cost_usd": pv.cost_usd,
                     "injection_flags": pv.injection_flags,
                     "near_duplicate_of": pv.near_duplicate_of,
                     "is_synthetic": pv.is_synthetic})
        if p.quality.validation_flags:
            st.markdown("**Validation flags**")
            for f in p.quality.validation_flags:
                st.markdown(f"- {html.escape(f)}")
        if p.quality.review_reasons:
            st.markdown("**Routed to review because**")
            for r in p.quality.review_reasons:
                st.markdown(f"- {html.escape(r)}")

    with tabs[5]:
        if p.raw_text:
            st.caption(f"Extracted text after layout repair · {len(p.raw_text)} characters")
            st.text_area("Source", p.raw_text, height=440, label_visibility="collapsed")
        else:
            C.empty_state(
                "Source text not available in this session",
                "The extracted profile above is intact, but the original document text "
                "was not re-attached for this run — the per-run state file this session "
                "expected has moved, been cleaned up, or belongs to an older process. "
                "Restart the app or re-run `scripts/run_pipeline.py` to regenerate it.",
                icon="⌀")


def _timeline(p) -> None:
    import plotly.express as px
    rows = []
    for e in p.employment:
        s = e.dates.start.normalized_value
        en = e.dates.end.normalized_value
        if not s:
            continue
        s_full = f"{s}-01" if len(s) == 7 else f"{s}-06-01"
        if en == "present" or not en:
            e_full = pd.Timestamp.today().strftime("%Y-%m-%d")
        else:
            e_full = f"{en}-28" if len(en) == 7 else f"{en}-12-31"
        rows.append({"Role": f"{e.title_raw.display('?')} · {e.employer_canonical or '?'}",
                     "Start": s_full, "End": e_full,
                     "Type": "Internship" if e.is_internship else "Full-time"})
    if not rows:
        st.markdown('<div class="mm-warn">No dated employment entries. This CV states '
                    'tenure as durations only, so no timeline can be drawn without '
                    'inventing dates — which the pipeline refuses to do.</div>',
                    unsafe_allow_html=True)
        return
    df = pd.DataFrame(rows)
    fig = px.timeline(df, x_start="Start", x_end="End", y="Role", color="Type",
                      color_discrete_map={"Full-time": theme.ACCENT, "Internship": "#94A3B8"},
                      height=90 + 40 * len(df))
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(size=11),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(theme.polish_fig(fig), width="stretch", config={"displayModeBar": False})
    if p.employment_gaps:
        st.markdown("**Employment gaps**")
        for g in p.employment_gaps:
            st.markdown(f'<div class="mm-warn">{g["months"]}-month gap between '
                        f'<b>{html.escape(str(g["after"]))}</b> and '
                        f'<b>{html.escape(str(g["before"]))}</b> ({g["from"]} → {g["to"]}). '
                        f'Surfaced as context for a conversation, not as a negative signal.'
                        f'</div>', unsafe_allow_html=True)


# ======================================================================= REQUISITION
SAMPLE_JD = """Investment Analyst — Healthcare Long/Short (New York)

Millennium is hiring a junior analyst for a fundamental healthcare long/short pod.

Requirements:
- 3-10 years of experience in healthcare equity research or healthcare investment banking
- Demonstrated financial modelling ability (three-statement, DCF)
- Must be based in, or willing to relocate to, the United States
- Bachelor's degree required

Preferred:
- CFA charterholder or candidate
- Prior buy-side experience at a multi-manager platform
- Exposure to medtech, diagnostics or pharmaceutical services
- Python for data analysis
"""


JD_TEMPLATES = {
    "Healthcare L/S": SAMPLE_JD,
    "Quant / C++": (
        "VP-level quantitative developer for a systematic equity book.\n\n"
        "Must-have: 5+ years C++ and Python, factor research. kdb+/q a plus.\n"
        "Europe or hybrid London. No sell-side-only backgrounds."
    ),
    "Credit research": (
        "Credit research analyst, 4–8 years, CFA preferred.\n"
        "HY / distressed experience and financial modelling. New York."
    ),
}


REQ_WEIGHT_PRESETS = {
    "Balanced": ScoreWeights().model_dump(),
    "Skill first": {"skills": 0.42, "strategy": 0.16, "sector": 0.14,
                    "semantic": 0.12, "geography": 0.06, "experience": 0.06,
                    "data_quality": 0.04},
    "Mandate fit": {"skills": 0.24, "strategy": 0.22, "sector": 0.18,
                    "semantic": 0.12, "geography": 0.12, "experience": 0.08,
                    "data_quality": 0.04},
    "Research mode": {"skills": 0.22, "strategy": 0.15, "sector": 0.13,
                      "semantic": 0.25, "geography": 0.07, "experience": 0.08,
                      "data_quality": 0.10},
}


_REQ_DISPLAY_KIND = {
    "strategies": "strategy", "sectors": "sector", "skills": "skill",
    "certifications": "certification", "degree_levels": "degree",
    "geo_regions": "region", "countries": "region", "employer_tiers": "tier",
    "feeder_paths": "feeder", "languages": "language",
}


def _seed_requisition() -> None:
    from millennium.retrieval import parse_query_rules
    if "req_jd_text" not in st.session_state:
        active = st.session_state.requisition if "requisition" in st.session_state else None
        raw = active["raw"] if active else SAMPLE_JD
        st.session_state.req_jd_text = raw or SAMPLE_JD
    if "requisition" not in st.session_state or not st.session_state.requisition:
        jd = st.session_state.req_jd_text
        st.session_state.requisition = {
            "parsed": parse_query_rules(jd).output, "raw": jd, "method": "rule",
            "requirements": []}


def _req_count(block: dict) -> int:
    n = 0
    for value in (block or {}).values():
        if isinstance(value, list):
            n += len([x for x in value if x])
        elif value not in (None, "", []):
            n += 1
    return n


def _req_format_value(key: str, value) -> str:
    if key == "min_years":
        return f"{float(value):g}+ years"
    if key == "max_years":
        return f"up to {float(value):g} years"
    if key == "min_seniority":
        return f"L{value}+ seniority"
    kind = _REQ_DISPLAY_KIND.get(key, key)
    return tx.display(kind, value, str(value))


def _req_chip_block(title: str, block: dict, tone: str = "plain") -> str:
    chips = []
    for key, value in (block or {}).items():
        values = value if isinstance(value, list) else [value]
        for v in values:
            if v in (None, "", []):
                continue
            chips.append(theme.chip(_req_format_value(key, v), tone))
    if not chips:
        chips.append(theme.chip("none", "missing"))
    return (f'<div class="mm-req-chip-block"><b>{html.escape(title)}</b>'
            f'<div>{"".join(chips)}</div></div>')


def _req_top_components(result, limit: int = 3) -> str:
    parts = []
    for c in sorted(result.components, key=lambda x: -x.contribution)[:limit]:
        parts.append(
            f'<div><span>{html.escape(c.name.replace("_", " "))}</span>'
            f'<b>{c.contribution:.3f}</b></div>')
    return "".join(parts)


def _render_req_hero(req, ranked, excluded, latency_ms: float) -> None:
    pq = req["parsed"]
    top_score = ranked[0].total if ranked else 0.0
    st.markdown(
        '<div class="mm-req-hero">'
        '<div><div class="mm-page-title">Requisition command center</div>'
        '<div class="mm-page-sub">Build the mandate, control hard requirements, '
        'inspect why people rank, and move the right candidates into shortlist.</div></div>'
        f'<div class="mm-req-hero-badge">{html.escape(str(req.get("method", "rule")).upper())}'
        '</div></div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="mm-req-metrics">'
        f'<div><b>{len(ranked)}</b><span>ranked matches</span><small>kept after gates</small></div>'
        f'<div><b>{len(excluded)}</b><span>gated out</span><small>visible with reasons</small></div>'
        f'<div><b>{top_score:.3f}</b><span>top score</span><small>best composite fit</small></div>'
        f'<div><b>{_req_count(pq.must_have)}</b><span>must-haves</span><small>hard constraints</small></div>'
        f'<div><b>{_req_count(pq.preferences)}</b><span>preferences</span><small>ranking signals</small></div>'
        f'<div><b>{latency_ms:.0f} ms</b><span>latency</span><small>current run</small></div>'
        '</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="mm-req-stepbar">'
        '<div><b>1</b><span>Paste JD</span><small>role, mandate, recruiter brief</small></div>'
        '<div><b>2</b><span>Parse needs</span><small>LLM or deterministic rules</small></div>'
        '<div><b>3</b><span>Check gates</span><small>must-haves stay visible</small></div>'
        '<div><b>4</b><span>Build slate</span><small>shortlist ranked matches</small></div>'
        '</div>',
        unsafe_allow_html=True)


def _render_req_brief_builder(client, store) -> None:
    st.markdown('<div class="mm-panel-heading"><span>Role brief</span>'
                '<b>paste, parse, reuse</b></div>', unsafe_allow_html=True)

    starter = st.selectbox("Starter mandate", ["Custom"] + list(JD_TEMPLATES),
                           key="req_starter_template")
    c1, c2 = st.columns(2)
    if c1.button("Use starter", icon=":material/content_copy:", width="stretch",
                 disabled=starter == "Custom", key="req_use_starter"):
        from millennium.retrieval import parse_query_rules
        st.session_state.req_jd_text = JD_TEMPLATES[starter]
        st.session_state.requisition = {
            "parsed": parse_query_rules(st.session_state.req_jd_text).output,
            "raw": st.session_state.req_jd_text, "method": "rule",
            "requirements": []}
        st.rerun()
    if c2.button("Reset sample", icon=":material/restart_alt:", width="stretch",
                 key="req_reset_sample"):
        from millennium.retrieval import parse_query_rules
        st.session_state.req_jd_text = SAMPLE_JD
        st.session_state.requisition = {
            "parsed": parse_query_rules(SAMPLE_JD).output, "raw": SAMPLE_JD,
            "method": "rule", "requirements": []}
        st.rerun()

    templates = store.list_templates()
    with st.expander("Saved role templates", expanded=bool(templates)):
        if templates:
            pick = st.selectbox("Saved template", ["-"] + [t["name"] for t in templates],
                                key="req_saved_template")
            cols = st.columns(2)
            if cols[0].button("Load", width="stretch", disabled=pick == "-",
                              key="req_load_saved"):
                rec = next(t for t in templates if t["name"] == pick)
                if rec.get("weights"):
                    st.session_state.weights = rec["weights"]
                    for k, v in rec["weights"].items():
                        st.session_state[f"w_{k}"] = float(v)
                st.session_state.req_jd_text = rec.get("jd", "") or SAMPLE_JD
                st.session_state.requisition = {
                    "parsed": _pq_from_dict(rec.get("parsed_query") or {}),
                    "raw": st.session_state.req_jd_text, "method": "template",
                    "requirements": rec.get("requirements") or []}
                st.rerun()
            if cols[1].button("Delete", width="stretch", disabled=pick == "-",
                              key="req_delete_saved"):
                store.delete_template(pick)
                st.rerun()
        else:
            st.caption("No templates saved yet.")

    st.markdown(
        '<div class="mm-jd-paste-card">'
        '<div><span class="mm-action-icon">contract_edit</span></div>'
        '<div><b>Paste a JD or recruiter mandate</b>'
        '<p>Use plain text. The parser will turn it into must-haves, preferences, '
        'gates, and ranking signals.</p>'
        '<small><span>LLM parse</span><span>Hard gates</span><span>Ranked slate</span></small>'
        '</div></div>',
        unsafe_allow_html=True)
    jd = st.text_area(
        "Paste job description / mandate",
        height=380,
        key="req_jd_text",
        placeholder="Paste the JD, recruiter brief, or mandate here...")
    C.llm_callout(
        "Requisition parser",
        "The primary parser sends the role brief to the LLM to structure "
        "must-haves, preferences, exclusions, and requirement rows. The rules-only "
        "button below does not use the LLM.",
        stage="requisition")
    p1, p2 = st.columns(2)
    if p1.button("Parse with LLM", type="primary", icon=":material/auto_awesome:",
                 width="stretch", key="req_parse_llm"):
        st.session_state.requisition = _parse_req(client, jd)
        st.rerun()
    if p2.button("Use rules only", icon=":material/rule:", width="stretch",
                 key="req_parse_rules"):
        from millennium.retrieval import parse_query_rules
        st.session_state.requisition = {
            "parsed": parse_query_rules(jd).output, "raw": jd, "method": "rule",
            "requirements": []}
        st.rerun()


def _render_req_weight_controls() -> ScoreWeights:
    st.markdown('<div class="mm-panel-heading compact"><span>Scoring lens</span>'
                '<b>weights normalize</b></div>', unsafe_allow_html=True)
    lens = st.segmented_control("Scoring lens", list(REQ_WEIGHT_PRESETS), default="Balanced",
                                key="req_weight_lens")
    if st.button("Apply lens", icon=":material/tune:", width="stretch",
                 key="req_apply_weight_lens"):
        st.session_state.weights = dict(REQ_WEIGHT_PRESETS[lens])
        for k, v in st.session_state.weights.items():
            st.session_state[f"w_{k}"] = float(v)
        st.rerun()

    w = dict(st.session_state.weights)
    for k, default in ScoreWeights().model_dump().items():
        w.setdefault(k, default)
    cols = st.columns(2)
    for i, k in enumerate(ScoreWeights().model_dump()):
        w[k] = cols[i % 2].slider(k, 0.0, 0.6, float(w[k]), 0.01,
                                  key=f"w_{k}")
    st.session_state.weights = w
    total = sum(w.values()) or 1.0
    st.markdown(
        f'<div class="mm-req-note">Raw total {total:.2f}. Scores are normalized to '
        '1.00 so one slider can move without manual rebalancing.</div>',
        unsafe_allow_html=True)
    return ScoreWeights(**w)


def _render_req_requirement_editor(req) -> ParsedQuery:
    pq: ParsedQuery = req["parsed"]
    st.markdown('<div class="mm-panel-heading compact"><span>Requirement control</span>'
                '<b>hard vs soft</b></div>', unsafe_allow_html=True)
    if req.get("requirements"):
        edited = st.data_editor(
            pd.DataFrame(req["requirements"]), width="stretch", hide_index=True,
            column_config={"must_have": st.column_config.CheckboxColumn("Must have"),
                           "text": st.column_config.TextColumn("Requirement", width="large"),
                           "quote": st.column_config.TextColumn("Source quote", width="medium")},
            key="req_editor")
        req["requirements"] = edited.to_dict("records")
        pq = _apply_requirement_edits(pq, edited)
        req["parsed"] = pq
    else:
        st.markdown(
            _req_chip_block("Must have", pq.must_have, "abstained") +
            _req_chip_block("Preference", pq.preferences, "verified") +
            _req_chip_block("Exclude", pq.exclusions, "conflicted"),
            unsafe_allow_html=True)
        st.caption("Use LLM parsing to edit individual requirement rows. The rule parser "
                   "still exposes extracted hard and soft constraints here.")
    return pq


def _score_requisition(pool, index, pq: ParsedQuery, weights: ScoreWeights):
    sem = _semantic_scores(index, pq)
    t0 = time.perf_counter()
    out = rank(pool, pq, weights, sem).output
    latency_ms = (time.perf_counter() - t0) * 1000
    st.session_state.last_latency_ms = latency_ms
    return out["ranked"], out["excluded"], sem, latency_ms


def _shortlist_ranked(ranked, pq: ParsedQuery, top_n: int, min_score: float,
                      tags: str = "jd-match") -> tuple[int, int]:
    sl = st.session_state.shortlist
    added = 0
    skipped = 0
    candidates = [r for r in ranked if r.total >= min_score][:top_n]
    for i, r in enumerate(candidates, 1):
        if r.candidate_id in sl:
            skipped += 1
            continue
        sl[r.candidate_id] = {
            "note": "", "tags": tags, "source": "ai",
            "basis": f"Ranked #{i} of {len(ranked)} for this requisition "
                     f"(score {r.total:.3f}) - {pq.interpretation[:140]}"}
        added += 1
    return added, skipped


def _render_req_strategy_panel(pq: ParsedQuery, ranked, excluded, store, sem,
                               weights: ScoreWeights, pool) -> tuple[int, float]:
    st.markdown('<div class="mm-panel-heading"><span>Pool strategy</span>'
                '<b>fit, actions, scenarios</b></div>', unsafe_allow_html=True)
    top_score = ranked[0].total if ranked else 0.0
    caveats = sum(1 for r in ranked if r.exclusion_reasons)
    st.markdown(
        '<div class="mm-req-fit-grid">'
        f'<div><b>{top_score:.3f}</b><span>top fit</span></div>'
        f'<div><b>{caveats}</b><span>needs proof</span></div>'
        f'<div><b>{len(excluded)}</b><span>hard gated</span></div>'
        f'<div><b>{len(st.session_state.shortlist)}</b><span>shortlist</span></div>'
        '</div>',
        unsafe_allow_html=True)
    st.progress(min(1.0, max(0.0, top_score)), text="Top candidate fit")

    if not ranked:
        st.warning("No one survived the current hard requirements. Relax one must-have "
                   "or use the gated-out list below to see which rule is too strict.")
    elif len(excluded) > len(ranked):
        st.info("The pool is being squeezed by hard requirements. Check gated-out "
                "reasons before assuming the market is thin.")
    elif caveats:
        st.info("Some strong matches have unverified must-haves. Treat those as quick "
                "research tasks, not automatic rejections.")
    else:
        st.success("The mandate has a clean ranked pool with visible scoring support.")

    top_n = st.number_input("Shortlist top N", min_value=1,
                            max_value=max(1, min(50, len(ranked) or 1)),
                            value=max(1, min(5, len(ranked) or 1)),
                            key="req_top_n")
    min_score = st.slider("Minimum shortlist score", 0.0, 1.0, 0.0, 0.05,
                          key="req_min_shortlist_score")
    if st.button("Add ranked slate", type="primary", icon=":material/star:",
                 width="stretch", disabled=not ranked, key="req_add_ranked_slate"):
        added, skipped = _shortlist_ranked(ranked, pq, int(top_n), float(min_score))
        st.success(f"Added {added} candidate(s); skipped {skipped} already listed.")

    tname = st.text_input("Save current mandate as", placeholder="Template name",
                          key="req_template_name")
    if st.button("Save role template", icon=":material/bookmark_add:", width="stretch",
                 disabled=not tname.strip(), key="req_save_template"):
        store.save_template(
            tname.strip(), st.session_state.requisition.get("raw", ""),
            st.session_state.weights,
            {"semantic_text": pq.semantic_text,
             "must_have": pq.must_have,
             "preferences": pq.preferences,
             "exclusions": pq.exclusions,
             "interpretation": pq.interpretation},
            st.session_state.requisition.get("requirements", []))
        st.success(f"Saved template '{tname.strip()}'.")

    with st.expander("Parsed mandate summary", expanded=True):
        st.markdown(
            _req_chip_block("Must have", pq.must_have, "abstained") +
            _req_chip_block("Preference", pq.preferences, "verified") +
            _req_chip_block("Exclude", pq.exclusions, "conflicted"),
            unsafe_allow_html=True)

    with st.expander("Weight sensitivity"):
        st.caption("Runs the match repeatedly with small weight changes to show whether "
                   "the top slate is stable.")
        if st.button("Run sensitivity sweep", icon=":material/network_check:",
                     key="req_run_sensitivity"):
            s = weight_sensitivity(pool, pq, weights, sem).output
            df = pd.DataFrame(s["stability"])
            if not df.empty:
                byid = _byid(pool)
                df["candidate"] = df["candidate_id"].map(
                    lambda c: byid[c].display_name(st.session_state.blind) if c in byid else c)
                st.dataframe(df[["candidate", "base_rank", "max_rank_shift",
                                 "mean_abs_shift", "verdict"]],
                             width="stretch", hide_index=True)
    return int(top_n), float(min_score)


def _render_req_match_card(i: int, r, p, pq: ParsedQuery, weights: ScoreWeights,
                           sem: dict, pool) -> None:
    with st.container(border=True, key=f"req_match_card_{r.candidate_id}"):
        st.markdown(
            f'<div class="mm-req-card-head"><div class="mm-req-rank">#{i}</div>'
            f'<div class="mm-req-score"><b>{r.total:.3f}</b><span>match score</span></div>'
            f'<div class="mm-req-components">{_req_top_components(r)}</div></div>',
            unsafe_allow_html=True)
        left, right = st.columns([0.62, 0.38])
        with left:
            st.markdown(C.candidate_card(p, st.session_state.blind, r.total),
                        unsafe_allow_html=True)
            if r.exclusion_reasons:
                st.markdown(
                    '<div class="mm-warn"><b>Unverified must-have.</b> '
                    + html.escape("; ".join(x.replace("unverified: ", "")
                                            for x in r.exclusion_reasons))
                    + ' - kept visible because unknown is a research task, not a rejection.</div>',
                    unsafe_allow_html=True)
        with right:
            mx = max((c.contribution for c in r.components), default=0.35)
            st.markdown("".join(C.score_bar(c.name, c.weight, c.score,
                                            c.contribution, mx)
                                for c in r.components), unsafe_allow_html=True)

        actions = st.columns([0.22, 0.26, 0.26, 0.26])
        if actions[0].button("Open", key=f"req_open_{r.candidate_id}",
                             icon=":material/open_in_new:", width="stretch"):
            st.session_state.selected = r.candidate_id
            st.session_state.page = "Candidate"
            st.rerun()
        listed = r.candidate_id in st.session_state.shortlist
        if actions[1].button("Listed" if listed else "Shortlist",
                             key=f"req_shortlist_{r.candidate_id}",
                             icon=":material/star:", width="stretch"):
            if listed:
                st.session_state.shortlist.pop(r.candidate_id)
            else:
                st.session_state.shortlist[r.candidate_id] = {
                    "note": "", "tags": "jd-match", "source": "ai",
                    "basis": f"Ranked #{i} for this requisition (score {r.total:.3f})"}
            st.rerun()
        if actions[2].button("Why not higher", key=f"req_why_{r.candidate_id}",
                             icon=":material/troubleshoot:", width="stretch"):
            res = minimal_edit(pool, pq, weights, r.candidate_id, sem).output
            st.session_state[f"req_why_result_{r.candidate_id}"] = res
        if actions[3].button("Open Shortlist", key=f"req_open_shortlist_{r.candidate_id}",
                             icon=":material/list_alt:", width="stretch"):
            st.session_state.page = "Shortlist"
            st.rerun()

        if f"req_why_result_{r.candidate_id}" in st.session_state:
            res = st.session_state[f"req_why_result_{r.candidate_id}"]
            if res["minimal"]:
                e = res["minimal"]
                st.success(f"Rank {e['from_rank']} to {e['new_rank']} if you {e['description']}.")
            elif res["edits"]:
                e = res["edits"][0]
                st.info(f"Closest single change: {e['description']} gives rank {e['new_rank']}.")
            else:
                st.info("No single requirement change moves this candidate up.")

        with st.expander("Gaps, unknowns, and evidence signals"):
            g = gap_analysis(r).output
            gc = st.columns(3)
            gc[0].markdown("**Has**\n\n" + ("\n".join(f"- {x}" for x in g["has"]) or "- none"))
            gc[1].markdown("**Lacks**\n\n" + ("\n".join(f"- {x}" for x in g["lacks"]) or "- none"))
            gc[2].markdown("**Unknown**\n\n" + ("\n".join(f"- {x}" for x in g["unknown"]) or "- none"))
            for c in r.components:
                if c.note:
                    st.caption(f"{c.name}: {c.note}")


def _render_req_results(ranked, excluded, pq: ParsedQuery, weights: ScoreWeights,
                        sem: dict, pool, min_score: float) -> None:
    byid = _byid(pool)
    st.markdown('<div class="mm-results-bar"><div><span class="mm-results-bar-label">'
                'Ranked talent slate</span><span class="mm-results-bar-meta">'
                f'{len(ranked)} matches - {len(excluded)} gated out</span></div></div>',
                unsafe_allow_html=True)
    controls = st.columns([0.25, 0.25, 0.50])
    show_n = controls[0].selectbox("Show", [5, 10, 20, 50], index=1,
                                   key="req_show_ranked")
    view = controls[1].segmented_control("View", ["Slate", "Table"], default="Slate",
                                         key="req_results_view")
    passing = [r for r in ranked if r.total >= min_score]
    controls[2].caption(f"{len(passing)} match(es) at or above {min_score:.2f}")

    if view == "Table":
        rows = []
        for i, r in enumerate(passing[: int(show_n)], 1):
            p = byid.get(r.candidate_id)
            if not p:
                continue
            rows.append({"Rank": i, "Candidate": p.display_name(st.session_state.blind),
                         "Score": r.total, "Why": _match_why(r),
                         "Caveat": "; ".join(r.exclusion_reasons)})
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch",
            column_config={"Score": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.3f"),
                           "Why": st.column_config.TextColumn(width="large")})
    else:
        for i, r in enumerate(passing[: int(show_n)], 1):
            p = byid.get(r.candidate_id)
            if p:
                _render_req_match_card(i, r, p, pq, weights, sem, pool)

    if excluded:
        with st.expander(f"{len(excluded)} gated out by must-have requirements"):
            for r in excluded:
                p = byid.get(r.candidate_id)
                nm = p.display_name(st.session_state.blind) if p else r.candidate_id
                st.markdown(f"**{html.escape(nm)}** - "
                            + "; ".join(html.escape(x) for x in r.exclusion_reasons))


def render_requisition(profiles, synth, pool, index, index_manifest, manifest, store,
                       client, bench, evals):
    _seed_requisition()

    req = st.session_state.requisition
    hero_slot = st.empty()
    left, center = st.columns([0.40, 0.60], gap="medium")
    with left:
        with st.container(border=True, key="req_brief_panel"):
            _render_req_brief_builder(client, store)
            with st.expander("Scoring lens and requirement controls", expanded=False):
                weights = _render_req_weight_controls()
                pq = _render_req_requirement_editor(req)

    ranked, excluded, sem, latency_ms = _score_requisition(pool, index, pq, weights)
    with hero_slot.container():
        _render_req_hero(req, ranked, excluded, latency_ms)

    with left:
        with st.container(border=True, key="req_strategy_panel"):
            _top_n, min_score = _render_req_strategy_panel(
                pq, ranked, excluded, store, sem, weights, pool)
    with center:
        with st.container(border=True, key="req_results_panel"):
            _render_req_results(ranked, excluded, pq, weights, sem, pool, min_score)


def _parse_req(client, jd: str) -> dict:
    from millennium.retrieval import parse_query_rules
    try:
        system, msgs, hint = requisition_prompt(jd)
        d = client.complete_json(system, msgs, hint, stage="requisition").data
    except (LLMUnavailable, Exception) as e:  # noqa: BLE001
        st.warning(f"LLM requisition parsing unavailable ({type(e).__name__}); "
                   f"using the deterministic rule parser.")
        return {"parsed": parse_query_rules(jd).output, "raw": jd, "method": "rule",
                "requirements": []}
    reqs = [{"text": r.get("text", ""), "kind": r.get("kind", "other"),
             "value": r.get("value", ""), "must_have": bool(r.get("must_have")),
             "quote": (r.get("quote") or "")[:120]}
            for r in (d.get("requirements") or [])]
    pq = parse_query_rules(jd).output
    pq.semantic_text = d.get("summary") or jd[:400]
    pq.interpretation = f"LLM parsed {len(reqs)} requirement(s); " + (d.get("summary") or "")
    pq.method = "llm"
    if d.get("min_years") is not None:
        pq.must_have["min_years"] = float(d["min_years"])
    if d.get("max_years") is not None:
        pq.must_have["max_years"] = float(d["max_years"])
    return {"parsed": _apply_requirement_edits(pq, pd.DataFrame(reqs)), "raw": jd,
            "method": "llm", "requirements": reqs}


_KIND_KEY = {"strategy": "strategies", "sector": "sectors", "skill": "skills",
             "certification": "certifications", "education": "degree_levels",
             "geography": "geo_regions", "language": "languages"}


def _apply_requirement_edits(pq: ParsedQuery, df) -> ParsedQuery:
    """Move each requirement between the must-have and preference blocks per the
    recruiter's ticks. This is the control that stops an over-eager parser from
    silently emptying the pool."""
    if df is None or len(df) == 0:
        return pq
    for block in ("must_have", "preferences"):
        for k in _KIND_KEY.values():
            getattr(pq, block).setdefault(k, [])
    for _, row in df.iterrows():
        key = _KIND_KEY.get(str(row.get("kind")))
        val = str(row.get("value") or "").strip()
        if not key or not val:
            continue
        src, dst = ("preferences", "must_have") if row.get("must_have") else ("must_have", "preferences")
        pq_src = getattr(pq, src).get(key) or []
        getattr(pq, src)[key] = [x for x in pq_src if x != val]
        dst_list = getattr(pq, dst).get(key) or []
        if val not in dst_list:
            getattr(pq, dst)[key] = dst_list + [val]
    return pq


def _pq_from_dict(d: dict) -> ParsedQuery:
    """Rehydrate a stored template. Tolerant of an older stored shape: a template
    saved under a previous schema must load with sane defaults rather than explode."""
    return ParsedQuery(
        semantic_text=d.get("semantic_text", ""),
        must_have=d.get("must_have") or {}, preferences=d.get("preferences") or {},
        exclusions=d.get("exclusions") or {},
        interpretation=d.get("interpretation", "loaded from a saved role template"),
        method="template")


def _semantic_scores(index, pq: ParsedQuery) -> dict:
    text = pq.semantic_text or ""
    if not text.strip():
        return {}
    hits = retrieve(index, text, "hybrid", top_k=500).output or []
    return {h.candidate_id: h.score for h in hits}


def _apply_jd_template() -> None:
    pick = st.session_state.jd_templates if "jd_templates" in st.session_state else None
    if pick and pick in JD_TEMPLATES:
        st.session_state.search_match_jd = JD_TEMPLATES[pick]
        st.session_state.jd_templates = None


def _match_why(r) -> str:
    bits = []
    for c in sorted(r.components, key=lambda x: -x.contribution)[:3]:
        if c.matched:
            bits.append(f"{c.name}: {', '.join(c.matched[:2])}")
        elif c.note:
            bits.append(c.note[:80])
    if r.exclusion_reasons:
        bits.append("; ".join(r.exclusion_reasons)[:80])
    return " · ".join(bits) if bits else "—"


def _resume_match_studio(results, pool, index, client, query: str) -> None:
    """Score the candidates currently in view against a pasted JD, then shortlist."""
    if not _flag_on("match_studio_open"):
        return
    if "search_match_jd" not in st.session_state:
        st.session_state.search_match_jd = SAMPLE_JD
    if "shortlist" not in st.session_state:
        st.session_state.shortlist = {}

    with st.container(border=True, key="resume_match"):
        head, close = st.columns([0.88, 0.12], vertical_alignment="top")
        with head:
            st.markdown(
                '<div class="mm-studio-title match">'
                '<span class="mm-action-icon">person_search</span>'
                '<div><b>Match to JD workspace</b>'
                f'<small>Scores {len(results)} visible candidate(s); gated-out people stay explainable.</small></div>'
                '</div>',
                unsafe_allow_html=True)
        if close.button("Close", icon=":material/close:", key="close_match_studio",
                        width="stretch", help="Close Match to JD"):
            _close_studio("match")
            st.rerun()
        st.pills("JD templates", list(JD_TEMPLATES), key="jd_templates",
                 on_change=_apply_jd_template,
                 help="Load a ready-made mandate, then edit it.")
        if query.strip():
            if st.button("Use current search as the JD", key="jd_from_query",
                         icon=":material/content_copy:"):
                st.session_state.search_match_jd = query.strip()
                st.rerun()
        st.text_area("Job description", height=380, key="search_match_jd",
                     placeholder="Paste a job description or mandate…")
        n_cap = max(1, min(25, len(results) or 1))
        r1, r2, r3 = st.columns(3)
        top_n = r1.number_input("Take top", min_value=1, max_value=n_cap,
                                value=min(5, n_cap), key="match_top_n",
                                help="How many of the ranked matches to shortlist.")
        min_score = r2.slider("Min score", 0.0, 1.0, 0.0, 0.05, key="match_min_score",
                              help="Drop anyone below this composite score.")
        scope = r3.segmented_control(
            "Score against", ["In view", "Whole pool"], default="In view",
            key="match_scope",
            help="In view respects the current search and sidebar filters.")
        t1, t2 = st.columns(2)
        skip_existing = t1.toggle("Skip already shortlisted", value=True,
                                  key="match_skip")
        jump = t2.toggle("Open Shortlist after", value=True, key="match_jump")

        b1, b2, b3 = st.columns([1.2, 1.4, 1.6])
        preview = b1.button("Preview ranking", key="match_preview_btn",
                            icon=":material/preview:", width="stretch")
        commit = b2.button("Match & shortlist", type="primary",
                           key="match_commit_btn",
                           icon=":material/star:", width="stretch")
        if b3.button("Full Requisition workspace", key="match_open_req",
                     icon=":material/open_in_new:", width="stretch"):
            jd = st.session_state.search_match_jd
            if jd.strip():
                st.session_state.requisition = _parse_req(client, jd)
            st.session_state.page = "Requisition"
            st.rerun()

        ranked = None
        excluded = []
        pq = None
        if preview or commit:
            jd = st.session_state.search_match_jd if "search_match_jd" in st.session_state else ""
            if not str(jd).strip():
                st.warning("Paste a job description first.")
            elif not results and scope == "In view":
                st.warning("No candidates in view — clear filters or search again.")
            else:
                with st.spinner("Parsing the mandate and scoring resumes…"):
                    parsed_req = _parse_req(client, jd)
                    pq = parsed_req["parsed"]
                    sem = _semantic_scores(index, pq)
                    w = ScoreWeights(**st.session_state.weights)
                    src = pool if scope == "Whole pool" else [r[0] for r in results]
                    out = rank(src, pq, w, sem).output
                    ranked, excluded = out["ranked"], out["excluded"]
                st.session_state.match_ranked = ranked
                st.session_state.match_excluded = excluded
                st.session_state.match_pq = pq
                st.session_state.requisition = parsed_req
        elif "match_ranked" in st.session_state:
            ranked = st.session_state.match_ranked
            excluded = st.session_state.match_excluded if "match_excluded" in st.session_state else []
            pq = st.session_state.match_pq if "match_pq" in st.session_state else None

        if ranked:
            passing = [r for r in ranked if r.total >= float(min_score)]
            take = passing[: int(top_n)]
            byid = _byid(pool)
            k = st.columns(3)
            C.kpi(k[0], len(ranked), "ranked")
            C.kpi(k[1], len(excluded), "gated out",
                  colour="#B45309" if excluded else theme.ACCENT)
            C.kpi(k[2], f"{ranked[0].total:.3f}" if ranked else "—", "top score")
            rows = []
            for i, r in enumerate(take, 1):
                p = byid.get(r.candidate_id)
                rows.append({
                    "#": i,
                    "Candidate": p.display_name(st.session_state.blind) if p else r.candidate_id,
                    "Score": r.total,
                    "Why": _match_why(r),
                    "Already": "yes" if r.candidate_id in st.session_state.shortlist else "",
                })
            if rows:
                st.dataframe(
                    pd.DataFrame(rows), hide_index=True, width="stretch",
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            min_value=0.0, max_value=1.0, format="%.3f"),
                        "Why": st.column_config.TextColumn(width="large"),
                    })
            else:
                st.caption(f"{len(ranked)} candidate(s) ranked, but none reached the "
                           f"{min_score:.2f} minimum score — lower \"Min score\" to see them.")
            if excluded:
                with st.expander(f"{len(excluded)} gated out by must-haves"):
                    for r in excluded[:12]:
                        p = byid.get(r.candidate_id)
                        nm = p.display_name(st.session_state.blind) if p else r.candidate_id
                        st.markdown(f"**{html.escape(nm)}** — "
                                    + html.escape("; ".join(r.exclusion_reasons)))
            if commit:
                sl = st.session_state.shortlist
                added = 0
                skipped = 0
                interp = (pq.interpretation if pq is not None else "")[:140]
                for i, r in enumerate(take, 1):
                    if skip_existing and r.candidate_id in sl:
                        skipped += 1
                        continue
                    sl[r.candidate_id] = {
                        "note": "", "tags": "jd-match", "source": "ai",
                        "basis": (f"Search match #{i} of {len(ranked)} "
                                  f"(score {r.total:.3f}) — {interp}"),
                    }
                    added += 1
                st.session_state.match_flash = (
                    f"Added {added} candidate(s) to Shortlist"
                    + (f" · skipped {skipped} already listed" if skipped else "")
                    + ". A human still approves every one."
                )
                if jump:
                    st.session_state.page = "Shortlist"
                st.rerun()
        elif pq is not None:
            # `ranked == []`: the mandate parsed and scoring ran, but every candidate
            # in scope was gated out by a must-have. This used to render nothing at
            # all -- indistinguishable from the button not having done anything.
            byid = _byid(pool)
            C.empty_state(
                "No candidates match this mandate",
                (f"All {len(excluded)} candidate(s) in scope were gated out by a "
                 f"must-have requirement." if excluded else
                 "Nobody was in scope to score — widen \"Score against\" to the "
                 "whole pool, or clear filters.") +
                " Relax a must-have on the Requirement control panel, or lower it "
                "to a preference, and try again.",
                icon="⌀")
            if excluded:
                with st.expander(f"{len(excluded)} gated out by must-haves"):
                    for r in excluded[:12]:
                        p = byid.get(r.candidate_id)
                        nm = p.display_name(st.session_state.blind) if p else r.candidate_id
                        st.markdown(f"**{html.escape(nm)}** — "
                                    + html.escape("; ".join(r.exclusion_reasons)))
        else:
            st.caption("Preview ranking to see scores before anything is shortlisted.")


def _pool_import_studio(client) -> None:
    """Add PDF / Word / JSON / CSV profiles to the working pool immediately."""
    if not _flag_on("import_studio_open"):
        return
    from pathlib import Path

    from millennium.config import SETTINGS
    from millennium.orchestrator import Pipeline
    from .pages_intake import _merge_into_pool, _parse_import_file, _stage

    with st.container(border=True, key="pool_import"):
        head, close = st.columns([0.88, 0.12], vertical_alignment="top")
        with head:
            st.markdown(
                '<div class="mm-studio-title import">'
                '<span class="mm-action-icon">upload_file</span>'
                '<div><b>Import profiles workspace</b>'
                '<small>PDF and Word parse through the resume pipeline; JSON and CSV join as human-unverified records.</small></div>'
                '</div>',
                unsafe_allow_html=True)
        if close.button("Close", icon=":material/close:", key="close_import_studio",
                        width="stretch", help="Close Import"):
            _close_studio("import")
            st.rerun()
        if SETTINGS.flags.demo_mode:
            st.caption("DEMO_MODE is on: a resume that is not in the LLM cache will "
                       "degrade to abstained fields rather than call the API.")
        files = st.file_uploader(
            "Profiles", type=["pdf", "docx", "json", "csv"],
            accept_multiple_files=True, key="search_import_files",
            help="PDF / Word = parse as resumes. JSON / CSV = structured records.")
        manuals = st.session_state.manual_profiles
        if manuals:
            st.markdown(f"**{len(manuals)} imported this session**")
            for p in manuals:
                cols = st.columns([0.78, 0.22])
                cols[0].markdown(
                    f"- **{html.escape(p.display_name(st.session_state.blind))}** "
                    f"· `{html.escape(p.provenance.source_file or p.doc_id)}`")
                if cols[1].button("Delete", key=f"imp_del_{p.candidate_id}",
                                  icon=":material/delete:"):
                    _remove_from_pool([p.candidate_id])
                    st.rerun()
        if not files:
            if st.button("Open full Intake", key="jump_intake_from_search",
                         icon=":material/login:"):
                st.session_state.page = "Intake"
                st.rerun()
            return

        structured, resumes = [], []
        for f in files:
            name = f.name.lower()
            if name.endswith((".json", ".csv")):
                structured.append(f)
            else:
                resumes.append(f)
        pending, problems, preview_rows = [], [], []
        for f in structured:
            profs, probs = _parse_import_file(f)
            pending += profs
            problems += probs
            for p in profs:
                preview_rows.append({
                    "File": f.name, "Kind": "structured",
                    "Candidate": p.display_name(),
                    "Headline": p.headline.display(""),
                })
        for f in resumes:
            preview_rows.append({
                "File": f.name, "Kind": "resume",
                "Candidate": "(will be parsed)", "Headline": "",
            })
        if preview_rows:
            st.dataframe(pd.DataFrame(preview_rows), hide_index=True, width="stretch")
        for prob in problems:
            st.markdown(f'<div class="mm-warn">{html.escape(prob)}</div>',
                        unsafe_allow_html=True)
        n_files = len(files)
        if st.button(f"Add {n_files} file(s) to the candidate pool", type="primary",
                     key="search_import_commit", icon=":material/person_add:"):
            for f in resumes:
                staged, probs = _stage(Path(f.name), f.getvalue())
                problems += probs
                if not staged:
                    continue
                try:
                    res = Pipeline(client=client, max_workers=1).process(staged)
                    if res.profile is not None:
                        pending.append(res.profile)
                    else:
                        problems.append(f"{f.name}: parsed but no profile ({res.status})")
                except Exception as e:  # noqa: BLE001
                    problems.append(f"{f.name}: {type(e).__name__}: {e}")
            n = _merge_into_pool(pending)
            st.session_state.import_flash = (
                f"Added {n} candidate(s) to the working pool"
                + (f" — {len(pending) - n} were already present" if n != len(pending) else "")
                + "."
            )
            st.rerun()


def _results_actions(selected_ids: list[str]) -> None:
    """Bulk shortlist / open / delete for ticked table rows."""
    n = len(selected_ids)
    with st.container(horizontal=True, key="results_actions"):
        if n == 0:
            st.caption("Tick rows to shortlist, open, or delete.")
            return
        st.caption(f"**{n}** selected")
        if st.button("Open", key="act_open", icon=":material/open_in_new:",
                     disabled=n == 0):
            st.session_state.selected = selected_ids[0]
            st.session_state.page = "Candidate"
            st.rerun()
        if st.button("Shortlist", key="act_shortlist", icon=":material/star:"):
            sl = st.session_state.shortlist
            added = 0
            for cid in selected_ids:
                if cid not in sl:
                    sl[cid] = {"note": "", "tags": "", "source": "human", "basis":
                               "Ticked on Search"}
                    added += 1
            st.toast(f"Added {added} to Shortlist")
            st.rerun()
        if st.button("Delete", key="act_delete", icon=":material/delete:"):
            n_del = _remove_from_pool(selected_ids)
            st.toast(f"Removed {n_del} from the working pool")
            st.rerun()


_SOURCE_BADGE = {
    "ai": ("LLM", "LLM-ranked", "#2DD4BF"),
    "human": ("☆", "Human-selected", "#8B9CB3"),
    "assistant": ("💬", "Added via chat", "#A78BFA"),
}


def _draft_outreach_email(client, p) -> tuple[str, str, bool]:
    """-> (subject, body, was_ai_generated). Falls back to a deterministic template on
    ANY failure -- including the default state of this repo, DEMO_MODE with no cached
    response for this brand-new prompt -- so the button always produces something
    usable rather than an error. `was_ai_generated=False` is shown to the user, never
    hidden, matching the rest of the app's habit of never blurring 'the model said
    this' with 'we computed this'."""
    name = p.display_name()
    headline = p.headline.display("their background")
    role = p.current_role()
    current = (f"{role.title_raw.display('')} at {role.employer_raw.display('')}"
              if role else "")
    skills = ", ".join(s.canonical for s in p.skills[:6])
    background = f"Headline: {headline}\nCurrent role: {current}\nKey skills: {skills}"
    try:
        system = (
            "You are a buy-side recruiter's writing assistant. Draft a short, "
            "professional first-contact outreach email to a candidate, using ONLY the "
            "verified background given below -- never invent an employer, school, "
            "number, or achievement that isn't stated. 120-180 words, end with a clear "
            "call to action to schedule a short call. Return strict JSON: "
            '{"subject": "...", "body": "..."}.')
        blocks = [{"role": "user", "content": f"Candidate background:\n{background}"}]
        r = client.complete_json(system, blocks, '{"subject":"","body":""}',
                                 stage="outreach_draft")
        d = r.data or {}
        if d.get("subject") and d.get("body"):
            return str(d["subject"]), str(d["body"]), True
    except Exception:  # noqa: BLE001 -- any LLM failure (incl. LLMUnavailable) falls back
        pass
    first = name.split()[0] if name and not name.startswith("Candidate") else "there"
    subject = (f"Your background in {headline}" if headline != "their background"
              else "Quick intro")
    body = (f"Hi {first},\n\nI came across your background — {headline}"
           + (f", currently {current}" if current else "")
           + " — and wanted to reach out directly. We're working a search that looks "
             "like a strong match for your experience.\n\nWould you be open to a "
             "short call this week?\n\nBest,")
    return subject, body, False


def _ics_invite(candidate_name: str, dt: datetime, duration_min: int,
                interviewer: str, notes: str) -> str:
    """A real, standards-compliant .ics file -- downloadable and importable into any
    calendar app. Deliberately NOT sent anywhere: this app has no calendar API
    credentials, so the honest feature is a file the recruiter sends themselves, not
    a fake 'invite sent' toast."""
    uid = f"{uuid.uuid4().hex}@millennium-bd"
    dtstamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dtstart = dt.strftime("%Y%m%dT%H%M%S")
    dtend = (dt + timedelta(minutes=duration_min)).strftime("%Y%m%dT%H%M%S")
    desc = f"Interviewer: {interviewer}. {notes}".replace("\\", "\\\\") \
        .replace(",", "\\,").replace("\n", "\\n")
    return (
        "BEGIN:VCALENDAR\r\nVERSION:2.0\r\n"
        "PRODID:-//Millennium BD//Candidate Intelligence//EN\r\n"
        "BEGIN:VEVENT\r\n"
        f"UID:{uid}\r\nDTSTAMP:{dtstamp}\r\nDTSTART:{dtstart}\r\nDTEND:{dtend}\r\n"
        f"SUMMARY:Interview — {candidate_name}\r\n"
        f"DESCRIPTION:{desc}\r\n"
        "END:VEVENT\r\nEND:VCALENDAR\r\n")


def _render_outreach(p, store, client) -> None:
    st.markdown(
        '<div class="mm-warn">✉ No email or calendar service is connected to this '
        'app. Messages below are drafted and logged, never sent over any network. '
        'Only the draft action uses the LLM; logging and invite generation stay '
        'local.</div>', unsafe_allow_html=True)
    email = p.sensitive.email.display("unknown")
    st.caption(f"Recipient on file: {email}")

    dkey = f"draft_{p.candidate_id}"
    C.llm_callout(
        "Outreach draft",
        "Uses the LLM to draft a first-contact email from verified candidate "
        "background only. It does not send email.",
        stage="outreach_draft")
    if st.button("Draft with LLM", icon=":material/auto_awesome:",
                 key=f"aidraft_{p.candidate_id}"):
        subject, body, ai = _draft_outreach_email(client, p)
        st.session_state[dkey] = {"subject": subject, "body": body}
        if not ai:
            st.info("LLM unavailable (DEMO_MODE, or no cached response for this new "
                    "prompt) — used the deterministic template instead.")
    draft = st.session_state.get(
        dkey, {"subject": f"Opportunity — {p.headline.display('')}"[:120], "body": ""})
    subject = st.text_input("Subject", draft["subject"], key=f"subj_{p.candidate_id}")
    body = st.text_area("Message", draft["body"], key=f"body_{p.candidate_id}", height=140)
    if st.button("📧 Log as sent", key=f"send_{p.candidate_id}", type="primary",
                disabled=not body.strip()):
        store.log_communication(p.candidate_id, "email_sent", subject, body,
                                {"to": email})
        st.success("Logged. (Simulated — no live email service is configured.)")
        st.rerun()

    st.markdown("**📅 Schedule an interview**")
    c1, c2, c3 = st.columns(3)
    d = c1.date_input("Date", key=f"date_{p.candidate_id}", label_visibility="collapsed")
    t = c2.time_input("Time", key=f"time_{p.candidate_id}", label_visibility="collapsed")
    dur = c3.number_input("Minutes", 15, 180, 45, 15, key=f"dur_{p.candidate_id}",
                          label_visibility="collapsed")
    interviewer = st.text_input("Interviewer", key=f"intv_{p.candidate_id}",
                                placeholder="Interviewer name")
    notes = st.text_area("Notes", key=f"inotes_{p.candidate_id}", height=60,
                         placeholder="Interview notes / agenda")
    ics_key = f"ics_data_{p.candidate_id}"
    if st.button("Generate invite (.ics)", key=f"ics_{p.candidate_id}"):
        dt = datetime.combine(d, t)
        store.log_communication(
            p.candidate_id, "interview_scheduled", f"Interview — {dt.isoformat()}",
            notes, {"interviewer": interviewer, "duration_min": int(dur),
                   "start": dt.isoformat()})
        st.session_state[ics_key] = _ics_invite(
            p.display_name(st.session_state.blind), dt, int(dur), interviewer, notes)
        st.success("Interview logged.")
    if st.session_state.get(ics_key):
        st.download_button("⬇ Download interview_invite.ics",
                          st.session_state[ics_key],
                          file_name=f"interview_{p.candidate_id[:8]}.ics",
                          mime="text/calendar", key=f"dl_ics_{p.candidate_id}")

    hist = store.list_communications(p.candidate_id)
    if hist:
        st.markdown("**History**")
        icons = {"email_sent": "📧", "email_inbound": "📥", "interview_scheduled": "📅"}
        for h in hist[:10]:
            st.markdown(
                f'<div class="mm-card">{icons.get(h["kind"], "•")} '
                f'<b>{html.escape(h["subject"] or h["kind"])}</b> '
                f'<span class="mm-sub">{h["created_at"]}</span><br>'
                f'<span class="mm-sub">{html.escape((h["body"] or "")[:160])}</span></div>',
                unsafe_allow_html=True)


def _render_inbox(sl: dict, byid: dict, store) -> None:
    st.markdown("##### 📥 Inbox — logged outreach & replies")
    st.caption("Not a live mailbox — this app has no IMAP/SMTP connection. Every row "
              "here was either logged from a candidate's Outreach panel above, or "
              "entered by hand below when a reply comes in through your real inbox.")
    names = {cid: byid[cid].display_name(st.session_state.blind)
            for cid in sl if cid in byid}
    ic1, ic2, ic3 = st.columns([0.3, 0.5, 0.2])
    pick = ic1.selectbox("Candidate", list(names.keys()),
                         format_func=lambda c: names.get(c, c), key="inbox_pick",
                         label_visibility="collapsed") if names else None
    reply = ic2.text_input("Paste their reply", key="inbox_reply_text",
                          label_visibility="collapsed",
                          placeholder="Paste a reply you received…")
    if ic3.button("Log reply", width="stretch", disabled=not (pick and reply.strip())):
        store.log_communication(pick, "email_inbound", "Reply", reply.strip())
        st.success("Logged.")
        st.rerun()

    rows = []
    for cid in sl:
        for h in store.list_communications(cid):
            rows.append({"when": h["created_at"], "candidate": names.get(cid, cid),
                        "kind": h["kind"], "subject": h["subject"],
                        "preview": (h["body"] or "")[:80]})
    if not rows:
        st.caption("No communications logged yet.")
        return
    df = pd.DataFrame(rows).sort_values("when", ascending=False)
    st.dataframe(df, width="stretch", hide_index=True)


def _sl_initials(name: str) -> str:
    return "".join(part[0] for part in name.replace("(", " ").split()
                   if part[:1].isalpha())[:2].upper() or "?"


def _shortlist_source_badge(source: str) -> str:
    icon, label, colour = _SOURCE_BADGE.get(source, _SOURCE_BADGE["human"])
    return (f'<span class="mm-chip" style="background:{colour}18;'
            f'color:{colour};border-color:{colour}44">{icon} {label}</span>')


def _shortlist_stage(cid: str, entry: dict, store) -> tuple[str, str]:
    kinds = {h["kind"] for h in store.list_communications(cid)}
    tags = str(entry.get("tags", "")).lower()
    if "interview_scheduled" in kinds or "interview" in tags:
        return "Interview", "ok"
    if "email_inbound" in kinds:
        return "Reply", "accent"
    if "email_sent" in kinds or "outreach" in tags or "screen" in tags:
        return "Outreach", "warn"
    if entry.get("note") or entry.get("basis"):
        return "Review", "plain"
    return "New", "plain"


def _shortlist_priority(p, entry: dict, store) -> tuple[str, str]:
    stage, _tone = _shortlist_stage(p.candidate_id, entry, store)
    if p.quality.needs_human_review:
        return "Needs review", "warn"
    if stage in {"Interview", "Reply"}:
        return "Active", "ok"
    if p.quality.completeness >= 0.9 and p.quality.evidence_coverage >= 0.85:
        return "Ready", "accent"
    return "Check details", "plain"


def _shortlist_table(sl: dict, byid: dict) -> pd.DataFrame:
    rows = []
    for cid in sl:
        p = byid.get(cid)
        if not p:
            continue
        rows.append({
            "Candidate": p.display_name(st.session_state.blind),
            "Region": tx.display("region", p.geo_region.label, "—") if p.geo_region else "—",
            "Years": p.years_experience.display("unknown"),
            "Seniority": p.seniority.label if p.seniority else "—",
            "Approach": p.quant_fundamental.label if p.quant_fundamental else "—",
            "Feeder": tx.display("feeder", p.feeder_path.label) if p.feeder_path else "—",
            "Strategies": ", ".join(tx.display("strategy", c.label) for c in p.strategies[:3]),
            "Sectors": ", ".join(tx.display("sector", c.label) for c in p.sectors[:3]),
            "Core skills": ", ".join(s.canonical for s in p.skills if s.depth == "core")[:60],
            "Certs": ", ".join(tx.display("certification", c.canonical)
                               for c in p.certifications if c.canonical),
            "Completeness": f"{p.quality.completeness:.0%}",
            "Review": "yes" if p.quality.needs_human_review else "no",
            "Note": sl[cid].get("note", ""),
            "Tags": sl[cid].get("tags", ""),
        })
    return pd.DataFrame(rows)


def _render_shortlist_hero(sl: dict, byid: dict, store) -> None:
    ids = [cid for cid in sl if cid in byid]
    active = reviews = interviews = noted = touches = 0
    for cid in ids:
        p = byid[cid]
        entry = sl[cid]
        stage, _tone = _shortlist_stage(cid, entry, store)
        active += int(stage in {"Outreach", "Reply", "Interview"})
        interviews += int(stage == "Interview")
        reviews += int(p.quality.needs_human_review)
        noted += int(bool(entry.get("note")))
        touches += len(store.list_communications(cid))
    st.markdown(
        '<div class="mm-sl-hero">'
        '<div><div class="mm-page-title">Shortlist command center</div>'
        '<div class="mm-page-sub">Curate the slate, keep outreach moving, compare '
        'finalists, and export a clean decision pack.</div></div>'
        '<div class="mm-sl-flow"><span>Slate</span><b></b><span>Outreach</span>'
        '<b></b><span>Decision</span></div></div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="mm-sl-metrics">'
        f'<div><b>{len(ids)}</b><span>shortlisted</span><small>current slate</small></div>'
        f'<div><b>{active}</b><span>active</span><small>outreach or interview</small></div>'
        f'<div><b>{interviews}</b><span>interviews</span><small>scheduled in app</small></div>'
        f'<div><b>{reviews}</b><span>need review</span><small>before sharing</small></div>'
        f'<div><b>{noted}</b><span>with notes</span><small>recruiter context</small></div>'
        f'<div><b>{touches}</b><span>touchpoints</span><small>logged history</small></div>'
        '</div>',
        unsafe_allow_html=True)


def _render_shortlist_empty(pool) -> None:
    st.markdown(
        '<div class="mm-sl-hero"><div><div class="mm-page-title">Shortlist command center</div>'
        '<div class="mm-page-sub">Your slate will appear here once candidates are added '
        'from Search, Candidate, chat, or Requisition.</div></div></div>',
        unsafe_allow_html=True)
    C.empty_state(
        "No one on the shortlist",
        "Add people from Search, from a profile, from the assistant, or in bulk from Requisition.")
    c1, c2 = st.columns(2)
    if c1.button("Open Search", type="primary", icon=":material/search:",
                 key="sl_empty_search", width="stretch"):
        st.session_state.page = "Search"
        st.rerun()
    if c2.button("Open Requisition", icon=":material/assignment:",
                 key="sl_empty_req", width="stretch"):
        st.session_state.page = "Requisition"
        st.rerun()
    if pool:
        st.markdown('<div class="mm-panel-heading compact"><span>Strong starting points</span>'
                    '<b>complete records</b></div>', unsafe_allow_html=True)
        starters = sorted(pool, key=lambda p: (-p.quality.completeness,
                                               -p.quality.evidence_coverage))[:3]
        cols = st.columns(3)
        for col, p in zip(cols, starters):
            with col:
                st.markdown(C.candidate_card(p, st.session_state.blind), unsafe_allow_html=True)
                if st.button("Add", key=f"sl_empty_add_{p.candidate_id}",
                             icon=":material/star:", width="stretch"):
                    st.session_state.shortlist[p.candidate_id] = {
                        "note": "", "tags": "", "source": "human",
                        "basis": "Added from Shortlist starter suggestions"}
                    st.rerun()


def _render_shortlist_rail(sl: dict, byid: dict, store) -> None:
    ids = [cid for cid in sl if cid in byid]
    st.session_state.setdefault("shortlist_selected", ids[0])
    if st.session_state.shortlist_selected not in ids:
        st.session_state.shortlist_selected = ids[0]
    st.markdown('<div class="mm-panel-heading"><span>Slate</span>'
                f'<b>{len(ids)} people</b></div>', unsafe_allow_html=True)
    q = st.text_input("Find in shortlist", "", key="sl_search",
                      placeholder="Search name, tags, notes",
                      label_visibility="collapsed")
    stage_filter = st.segmented_control(
        "Stage", ["All", "New", "Review", "Outreach", "Reply", "Interview"],
        default="All", key="sl_stage_filter")
    visible = []
    needle = q.strip().lower()
    for cid in ids:
        p = byid[cid]
        entry = sl[cid]
        stage, tone = _shortlist_stage(cid, entry, store)
        text = " ".join([p.display_name(st.session_state.blind),
                         str(p.headline.display("")), entry.get("note", ""),
                         entry.get("tags", ""), entry.get("basis", ""), stage]).lower()
        if stage_filter != "All" and stage != stage_filter:
            continue
        if needle and needle not in text:
            continue
        visible.append((cid, p, entry, stage, tone))
    if not visible:
        st.markdown('<div class="mm-review-empty">No shortlisted candidate matches.</div>',
                    unsafe_allow_html=True)
        return
    if st.session_state.shortlist_selected not in {cid for cid, *_ in visible}:
        st.session_state.shortlist_selected = visible[0][0]
        st.rerun()
    for cid, p, entry, stage, tone in visible:
        is_sel = cid == st.session_state.shortlist_selected
        priority, _ptone = _shortlist_priority(p, entry, store)
        name = p.display_name(st.session_state.blind)
        headline = p.headline.display("No headline") or "No headline"
        st.markdown(
            f'<div class="mm-sl-pick {tone}{" is-selected" if is_sel else ""}">'
            f'<div class="mm-review-avatar">{html.escape(_sl_initials(name))}</div>'
            f'<div class="mm-sl-pick-body"><div class="mm-sl-pick-top">'
            f'<span>{html.escape(name)}</span><b>{html.escape(stage)}</b></div>'
            f'<div class="mm-sl-pick-sub">{html.escape(str(headline))}</div>'
            f'<div class="mm-sl-pick-meta">{html.escape(priority)} · '
            f'{html.escape(entry.get("tags", "") or "no tags")}</div></div></div>',
            unsafe_allow_html=True)
        if st.button("Selected" if is_sel else "Open", key=f"sl_select_{cid}",
                     type="primary" if is_sel else "secondary", disabled=is_sel,
                     width="stretch"):
            st.session_state.shortlist_selected = cid
            st.rerun()


def _render_shortlist_profile(cid: str, p, sl: dict, store, client) -> None:
    entry = sl[cid]
    entry.setdefault("source", "human")
    entry.setdefault("basis", "")
    stage, tone = _shortlist_stage(cid, entry, store)
    priority, _ptone = _shortlist_priority(p, entry, store)
    name = p.display_name(st.session_state.blind)
    current = p.current_role()
    current_txt = (f"{current.title_raw.display('')} at {current.employer_raw.display('')}"
                   if current else p.headline.display("No current role extracted"))
    st.markdown(
        f'<div class="mm-sl-profile-head {tone}">'
        f'<div class="mm-review-avatar xl">{html.escape(_sl_initials(name))}</div>'
        f'<div class="mm-sl-profile-title"><div class="mm-review-case-kicker">'
        f'{html.escape(stage)} · {html.escape(priority)}</div>'
        f'<h2>{html.escape(name)}</h2><p>{html.escape(str(current_txt))}</p>'
        f'<div>{_shortlist_source_badge(entry["source"])}</div></div>'
        f'<div class="mm-sl-profile-score"><b>{p.quality.completeness:.0%}</b>'
        '<span>profile complete</span></div></div>',
        unsafe_allow_html=True)
    if entry["basis"]:
        st.markdown(f'<div class="mm-sl-basis">{html.escape(entry["basis"])}</div>',
                    unsafe_allow_html=True)

    cols = st.columns(4)
    C.kpi(cols[0], p.years_experience.display("unknown"), "Experience")
    C.kpi(cols[1], tx.display("region", p.geo_region.label, "—") if p.geo_region else "—",
          "Region")
    C.kpi(cols[2], len([s for s in p.skills if s.depth == "core"]), "Core skills")
    C.kpi(cols[3], "yes" if p.quality.needs_human_review else "no", "Needs review",
          colour=theme.WARNING if p.quality.needs_human_review else theme.SUCCESS)

    with st.container(border=True, key="sl_notes_panel"):
        st.markdown('<div class="mm-panel-heading"><span>Recruiter decision notes</span>'
                    '<b>saved in slate</b></div>', unsafe_allow_html=True)
        n1, n2 = st.columns([0.62, 0.38])
        sl[cid]["note"] = n1.text_area(
            "Recruiter note", entry.get("note", ""), key=f"n_{cid}", height=108,
            placeholder="Why this person belongs on the slate, concerns, next step...")
        sl[cid]["tags"] = n2.text_input(
            "Tags", entry.get("tags", ""), key=f"t_{cid}",
            placeholder="screen-call, backup, strong-fit")
        a1, a2, a3 = st.columns([0.28, 0.28, 0.44])
        if a1.button("Open profile", key=f"sl_open_profile_{cid}",
                     icon=":material/open_in_new:", width="stretch"):
            st.session_state.selected = cid
            st.session_state.page = "Candidate"
            st.rerun()
        if a2.button("Remove", key=f"r_{cid}", icon=":material/delete:",
                     width="stretch"):
            sl.pop(cid)
            if "shortlist_selected" in st.session_state:
                del st.session_state["shortlist_selected"]
            st.rerun()
        if a3.button("Open requisition matcher", key=f"sl_open_req_{cid}",
                     icon=":material/assignment:", width="stretch"):
            st.session_state.page = "Requisition"
            st.rerun()

    with st.container(border=True, key="sl_profile_exports_panel"):
        st.markdown('<div class="mm-panel-heading"><span>Profile export</span>'
                    '<b>selected finalist</b></div>', unsafe_allow_html=True)
        _profile_downloads_compact(p, prefix=f"sl_{cid[:8]}")

    with st.container(border=True, key="sl_outreach_panel"):
        st.markdown('<div class="mm-panel-heading"><span>Outreach and scheduling</span>'
                    '<b>draft, log, invite</b></div>', unsafe_allow_html=True)
        _render_outreach(p, store, client)


def _render_shortlist_side(sl: dict, byid: dict, store, client) -> None:
    st.markdown('<div class="mm-panel-heading"><span>Slate controls</span>'
                '<b>compare & export</b></div>', unsafe_allow_html=True)
    df = _shortlist_table(sl, byid)
    if not df.empty:
        avg_complete = sum(byid[cid].quality.completeness
                           for cid in sl if cid in byid) / max(1, len(df))
    else:
        avg_complete = 0.0
    st.markdown(
        '<div class="mm-sl-side-grid">'
        f'<div><b>{avg_complete:.0%}</b><span>avg complete</span></div>'
        f'<div><b>{len(df)}</b><span>export rows</span></div></div>',
        unsafe_allow_html=True)
    if not df.empty:
        st.dataframe(df[["Candidate", "Region", "Years", "Seniority", "Strategies",
                         "Sectors", "Completeness", "Review"]],
                     width="stretch", hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("CSV", df.to_csv(index=False), "shortlist.csv", "text/csv",
                       width="stretch")
    payload = {"shortlist": [
        {"candidate_id": cid, "note": sl[cid].get("note", ""),
         "tags": sl[cid].get("tags", ""),
         "profile": json.loads(byid[cid].model_dump_json(exclude={"raw_text"}))}
        for cid in sl if cid in byid]}
    d2.download_button("JSON", json.dumps(payload, indent=1), "shortlist.json",
                       "application/json", width="stretch")
    memo_key = "shortlist_ai_memo"
    C.llm_callout(
        "Slate memo",
        "Uses the LLM to summarize shortlist coverage, risks, and interview focus "
        "from the current slate. It does not approve candidates.",
        stage="shortlist_memo")
    if st.button("Generate LLM slate memo", icon=":material/auto_awesome:",
                 key="sl_ai_memo_btn", width="stretch", disabled=df.empty):
        st.session_state[memo_key] = llm_panels.shortlist_memo(client, sl, byid, store)
    if st.session_state.get(memo_key):
        _render_ai_json_panel(st.session_state[memo_key], "LLM slate memo")
    with st.expander("Inbox", expanded=True):
        _render_inbox(sl, byid, store)


# ========================================================================= SHORTLIST
def render_shortlist(profiles, synth, pool, index, index_manifest, manifest, store,
                     client, bench, evals):
    byid = _byid(pool)
    sl = st.session_state.shortlist
    if "match_flash" in st.session_state:
        st.success(st.session_state.match_flash)
        del st.session_state["match_flash"]
    if not sl:
        _render_shortlist_empty(pool)
        return
    valid_ids = [cid for cid in sl if cid in byid]
    if not valid_ids:
        _render_shortlist_empty(pool)
        return
    st.session_state.setdefault("shortlist_selected", valid_ids[0])
    if st.session_state.shortlist_selected not in valid_ids:
        st.session_state.shortlist_selected = valid_ids[0]

    _render_shortlist_hero(sl, byid, store)
    left, center, right = st.columns([0.24, 0.50, 0.26], gap="medium")
    with left:
        with st.container(border=True, key="sl_rail_panel"):
            _render_shortlist_rail(sl, byid, store)
    selected = st.session_state.shortlist_selected
    with center:
        with st.container(border=True, key="sl_profile_panel"):
            _render_shortlist_profile(selected, byid[selected], sl, store, client)
    with right:
        with st.container(border=True, key="sl_side_panel"):
            _render_shortlist_side(sl, byid, store, client)
