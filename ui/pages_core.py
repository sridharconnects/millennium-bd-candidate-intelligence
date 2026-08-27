"""Search / Candidate / Requisition / Shortlist — the recruiter's daily workspace."""
from __future__ import annotations

import html
import json
import time
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from millennium import taxonomy as tx
from millennium.config import ScoreWeights
from millennium.index import build_chunks
from millennium.llm import LLMUnavailable
from millennium.prompts import requisition_prompt
from millennium.retrieval import (ParsedQuery, apply_filters, retrieve, similar_candidates,
                                  understand_query)
from millennium.scoring import gap_analysis, minimal_edit, rank, weight_sensitivity
from . import components as C
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
                lines.append(f"| {name} | {fmt(f'{a['ndcg@10']:.3f}')} "
                             f"| {fmt(f'{a['mrr']:.3f}')} "
                             f"| {fmt(f'{a['latency_ms']:.1f} ms')} |")
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


# ============================================================================ SEARCH
def render_search(profiles, synth, pool, index, index_manifest, manifest, store,
                  client, bench, evals):
    C.synthetic_banner(len(synth) if st.session_state.include_synthetic else 0)

    # ONE search-and-results unit: the input row, the example chips, and the result
    # list all live in the same bordered container (`hero` is re-entered further down
    # once the results are computed), so the search visibly IS the thing filtering
    # the list rather than a separate widget floating above it. Pool at a glance
    # moved out to its own full-width section below.
    hero = st.container(border=True, key="search_hero")
    with hero:
        qcol, mcol, icol, scol, bcol = st.columns([0.47, 0.15, 0.05, 0.14, 0.19])
        with qcol:
            query = st.text_input(
                "Search", key="query", label_visibility="collapsed",
                placeholder="Describe the candidate you need — plain English works "
                            "(\"healthcare L/S in APAC, no banking background\")")
        with mcol:
            mode = st.selectbox("Retrieval", ["hybrid", "dense", "lexical"],
                                key="retrieval_mode", label_visibility="collapsed",
                                help="How matches are found and ranked. Click ⓘ for "
                                     "exactly what the selected mode does with your "
                                     "query, with measured accuracy for all three.")
        with icol:
            _mode_popover(st.session_state.retrieval_mode, evals)
        with scol:
            show_n = st.selectbox(
                "Show", [25, 50, 100, 250, "All"], index=1, key="f_show_n",
                label_visibility="collapsed",
                help="How many matches to display. With a large pool (e.g. the "
                     "500-record synthetic corpus), the default only shows the top "
                     "50 — raise this to see more.")
        with bcol:
            st.button("Search", type="primary", width="stretch")
        st.pills("Try an example search", list(EXAMPLES), key="search_examples",
                 on_change=_apply_example,
                 help="Each chip runs a ready-made plain-English query, so you can "
                      "see what the search understands without typing anything.")
        n_sl = len(st.session_state.shortlist)
        with st.container(horizontal=True, key="search_tools"):
            st.button("Match to JD", icon=":material/person_search:",
                      key="open_match_studio",
                      type="primary" if _flag_on("match_studio_open") else "secondary",
                      on_click=_toggle_studio, args=("match",),
                      help="Score resumes against a job description and shortlist "
                           "the top matches.")
            st.button("Import", icon=":material/upload_file:",
                      key="open_import_studio",
                      type="primary" if _flag_on("import_studio_open") else "secondary",
                      on_click=_toggle_studio, args=("import",),
                      help="Add PDF, Word, JSON, or CSV profiles to the pool.")
            if st.button(
                    f"Shortlist ({n_sl})" if n_sl else "Shortlist",
                    icon=":material/star:", key="jump_shortlist_from_search",
                    disabled=not n_sl,
                    help="Open the Shortlist tab."):
                st.session_state.page = "Shortlist"
                st.rerun()

    facets = _facets(pool)

    # A saved search restores the filter rail, not just the query text. Widget state
    # must be written before the widgets are constructed, and only values still present
    # in the current facets are restored -- a saved employer that has since been deleted
    # must not resurrect itself as a phantom filter.
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

    with st.sidebar:
        st.divider()
        st.markdown("#### Filters")
        F = {}
        F["region"] = st.multiselect("Region", facets["region"], key="f_region")
        F["strategy"] = st.multiselect("Strategy", facets["strategy"], key="f_strategy")
        F["sector"] = st.multiselect("Sector", facets["sector"], key="f_sector")
        F["skill"] = st.multiselect("Skills", facets["skill"], key="f_skill")
        F["seniority"] = st.multiselect(
            "Seniority", facets["seniority"], key="f_seniority",
            format_func=lambda s: f"{s} · {tx.display("seniority", s)}")
        F["years"] = st.slider("Years of experience", 0.0, 30.0, (0.0, 30.0), 0.5, key="f_years")
        F["include_unknown_years"] = st.checkbox(
            "Include candidates whose experience is unknown", value=True,
            help="Unknown ≠ zero. Three CVs in this corpus state tenure as a duration "
                 "with no dates, so no total can be derived without inventing one.")
        with st.expander("More filters"):
            F["approach"] = st.multiselect("Approach", facets["approach"], key="f_approach")
            F["feeder"] = st.multiselect("Feeder path", facets["feeder"], key="f_feeder")
            F["employer"] = st.multiselect("Employer", facets["employer"], key="f_employer")
            F["tier"] = st.multiselect("Employer tier", facets["tier"], key="f_tier")
            F["cert"] = st.multiselect("Certification", facets["cert"], key="f_cert")
            F["degree"] = st.multiselect("Degree level", facets["degree"], key="f_degree")
            F["language"] = st.multiselect("Language", facets["language"], key="f_language")
            F["min_completeness"] = st.slider("Minimum profile completeness", 0.0, 1.0, 0.0, 0.05)
            F["review_only"] = st.checkbox("Only records flagged for review")
        F.setdefault("min_completeness", 0.0)
        F.setdefault("review_only", False)
        F.setdefault("feeder", [])

        # Saved searches. A recruiter re-runs the same handful of searches every week;
        # making them rebuild the filter rail each time is pure friction.
        st.divider()
        st.markdown("#### Saved searches")
        saved = store.list_searches()
        if saved:
            pick = st.selectbox("Load", ["—"] + [s["name"] for s in saved],
                                label_visibility="collapsed")
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
        nm = st.text_input("Name", placeholder="e.g. APAC healthcare L/S",
                           label_visibility="collapsed", key="save_search_name")
        if st.button("Save current search", width="stretch",
                     disabled=not nm.strip()):
            store.save_search(nm.strip(), st.session_state.query, F,
                              st.session_state.retrieval_mode)
            st.success(f"Saved '{nm.strip()}'")

    show_cap = len(pool) if show_n == "All" else int(show_n)
    t0 = time.perf_counter()
    filtered = _manual_filter(pool, F)
    parsed = None
    hits = []
    if query.strip():
        parsed, presult = understand_query(query, client)
        # `top_k` only controls how many of the FUSED candidates are kept -- the two
        # sub-retrievers each still cap at SETTINGS.retrieval.top_k_{dense,lexical}
        # (50) internally, so a text query naturally tops out around there regardless
        # of `show_cap`. That is a relevance cutoff, not a bug: "500 matches" is not a
        # meaningful answer to a specific plain-English query the way it is when
        # browsing with no query at all (the `else` branch below).
        r = retrieve(index, parsed.semantic_text or query, mode, top_k=show_cap)
        hits = r.output or []
        allowed = {p.candidate_id for p in filtered}
        # Soft preferences must never eliminate, so query-derived filters are applied
        # only where the parser marked them as genuinely hard.
        gated, excluded, caveats = apply_filters(filtered, parsed)
        gset = {p.candidate_id for p in gated}
        ordered = [h for h in hits if h.candidate_id in allowed and h.candidate_id in gset]
        byid = _byid(pool)
        results = [(byid[h.candidate_id], h.score, h.explain, h.matched_chunks)
                   for h in ordered if h.candidate_id in byid]
        # Filter-only matches still belong in the list, below the ranked ones.
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
    # NB: do not also write "page" here. It used to be hardcoded to "Search", which
    # meant every Search render pinned the URL to that value; on the *next* rerun
    # (e.g. clicking a different sidebar item) the URL-restore block in app.py read
    # that stale param back and silently reverted the navigation. app.py now owns the
    # "page" query param centrally, written once from whatever the current page
    # actually is, so it can never fight with in-session navigation.

    # Results render INSIDE the same hero container as the search bar (re-entered
    # here now that they exist), separated by a divider: one surface, where typing
    # a query visibly narrows the list below it.
    with hero:
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
        st.divider()
        shown = min(show_cap, len(results))
        hd_l, hd_m, hd_r = st.columns([0.50, 0.22, 0.28])
        with hd_l:
            # "Showing X of Y" (not a bare count): the header itself says these are
            # the survivors of the search + filters, which is the whole mental model.
            if query.strip() or len(results) < len(pool):
                st.markdown(f"##### Showing {shown} of {len(pool)} candidates "
                            f"· {latency:.1f} ms")
            else:
                st.markdown(f"##### All {len(pool)} candidates · {latency:.1f} ms")
        with hd_m:
            if query.strip():
                st.button("✕ Clear search", key="clear_query", width="stretch",
                          on_click=_clear_query,
                          help="Remove the query and show the whole pool again "
                               "(sidebar filters stay applied)")
        with hd_r:
            view = st.segmented_control(
                # `default` is only consulted the first time this key is seen; once
                # the widget's own state exists, Streamlit uses that and ignores
                # `default`. So this does not need (and must not use) a `.get()` read
                # of session_state, which is also unsupported under the AppTest
                # harness this file is tested with.
                "View", ["Table", "Cards"], default="Table",
                key="view", label_visibility="collapsed",
                help="Table for dense scanning and sorting; cards when you want the "
                     "labels and flags at a glance.")
        if len(results) > shown:
            st.caption(f"showing {shown} of {len(results)} — raise \"Show\" above the "
                      f"search bar to see more")
        if not results:
            st.markdown('<div class="mm-warn">No candidates match. Try removing a '
                        'filter, or search in plain English instead — must-have terms '
                        'gate, preferences only score.</div>', unsafe_allow_html=True)
        if view == "Table":
            picked = _results_table(results[:shown])
            _results_actions(picked)
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
                           key=f"s_{p.candidate_id}",
                           icon=":material/star:"):
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

    # The gating explanations belong to the RESULTS (they say why someone is or is
    # not in the list above), so they render at the bottom of the hero.
    with hero:
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

    # Pool at a glance: its own full-width section OUTSIDE the search unit — it
    # describes the candidates currently in view, it is not a search control.
    # `key="pool_glance"` lets theme.py make it user-resizable (CSS `resize:vertical`
    # + a native drag handle at its bottom-right corner).
    with st.container(border=True, key="pool_glance"):
        st.markdown("##### Pool at a glance")
        st.caption("How the candidates currently shown above are distributed — "
                   "updates live with every search and filter. Drag the "
                   "bottom-right corner to resize.")
        _mini_charts(results)


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
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False},
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
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False},
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
    with st.container(border=True, key="cand_switcher"):
        i = ids.index(st.session_state.selected)
        st.markdown(
            f'<div class="mm-switch-head">Browse the pool'
            f'<span class="mm-sub"> · {i + 1} of {len(ids)} — pick a name or step '
            f'through</span></div>', unsafe_allow_html=True)
        prev_c, mid_c, next_c, del_c = st.columns(
            [0.14, 0.54, 0.14, 0.18], vertical_alignment="bottom")
        with prev_c:
            st.button("Prev", icon=":material/chevron_left:", key="cand_prev",
                      width="stretch", disabled=i == 0,
                      on_click=_step_candidate, args=(-1,),
                      help="Open the previous candidate in the pool")
        with mid_c:
            st.selectbox(
                "Switch candidate", ids, index=i, key="cand_switch_box",
                on_change=_sync_candidate_from_switch,
                format_func=lambda cid: byid[cid].display_name(st.session_state.blind),
                help="Every parsed profile in the current pool. Click to open a "
                     "different person without going back to Search.")
        with next_c:
            st.button("Next", icon=":material/chevron_right:", key="cand_next",
                      width="stretch", disabled=i >= len(ids) - 1,
                      icon_position="right",
                      on_click=_step_candidate, args=(1,),
                      help="Open the next candidate in the pool")
        with del_c:
            if st.button("Delete", icon=":material/delete:", key="cand_remove",
                         width="stretch",
                         help="Remove this profile from the working pool this session. "
                              "Imported records are dropped; corpus records come back "
                              "with Reset demo. GDPR erasure is on Review."):
                _remove_from_pool([st.session_state.selected])
                st.session_state.page = "Search"
                st.rerun()
    p = byid[st.session_state.selected]

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

    head = st.columns([0.55, 0.45])
    with head[0]:
        st.markdown(f"### {html.escape(p.display_name(st.session_state.blind))}")
        st.markdown(f'<div class="mm-sub">{html.escape(p.headline.display(""))}</div>'
                    f'<div style="margin-top:8px">{C.labels_row(p, 8)}</div>',
                    unsafe_allow_html=True)
        listed = p.candidate_id in st.session_state.shortlist
        act_l, act_r = st.columns([0.58, 0.42])
        with act_l:
            if listed:
                if st.button("On shortlist — remove", icon=":material/star:",
                             key="cand_shortlist", width="stretch",
                             help="Remove this person from the Shortlist tab."):
                    st.session_state.shortlist.pop(p.candidate_id)
                    st.rerun()
            else:
                if st.button("Add to shortlist", icon=":material/star:",
                             type="primary", key="cand_shortlist", width="stretch",
                             help="Add this person to the Shortlist tab. A human still "
                                  "approves; this only proposes."):
                    st.session_state.shortlist[p.candidate_id] = {
                        "note": "", "tags": "", "source": "human",
                        "basis": "Manually shortlisted from the candidate profile"}
                    st.rerun()
        with act_r:
            if listed and st.button("Open Shortlist", icon=":material/open_in_new:",
                                    key="cand_go_sl", width="stretch"):
                st.session_state.page = "Shortlist"
                st.rerun()
    with head[1]:
        k = st.columns(4)
        C.kpi(k[0], f"{p.years_experience.value:.1f}" if p.years_experience.is_known else "—",
              "years exp", "derived" if p.years_experience.is_known else "unknown")
        C.kpi(k[1], f"{p.quality.completeness:.0%}", "complete")
        C.kpi(k[2], f"{p.quality.evidence_coverage:.0%}", "evidenced")
        C.kpi(k[3], p.quality.abstention_count, "abstained",
              colour="#B45309" if p.quality.abstention_count else theme.ACCENT)

    _profile_exports(p)

    tabs = st.tabs(["Profile", "Evidence", "Timeline", "Similar", "Lineage", "Source"])

    with tabs[0]:
        C.provenance_banner(p)
        if p.quality.validation_flags:
            with st.expander(f"⚑ {len(p.quality.validation_flags)} validation flag(s) "
                             f"on this record", expanded=True):
                st.markdown(theme.flag_list(p.quality.validation_flags),
                           unsafe_allow_html=True)
        _profile_insights(p, pool)
        st.divider()

        # Left: everything the agent pipeline produced. Right: the real, unmodified
        # source file, so the two can be checked against each other directly rather
        # than trusting a re-rendered/extracted stand-in for the original.
        work_col, doc_col = st.columns([0.56, 0.44])

    with work_col:
        st.markdown('<span class="mm-sub" style="text-transform:uppercase;'
                    'letter-spacing:.04em;font-weight:600">Agent-extracted profile</span>',
                    unsafe_allow_html=True)
        a, b = st.columns([0.52, 0.48])
        with a:
            st.markdown("**Identity & contact**")
            for t, lbl in ((p.sensitive.full_name, "Name"), (p.sensitive.email, "Email"),
                           (p.sensitive.phone, "Phone"), (p.location_current, "Location"),
                           (p.work_authorization, "Work authorisation")):
                if st.session_state.blind and lbl in ("Name", "Email", "Phone"):
                    st.markdown(f'<div class="mm-row"><span class="mm-sub">{lbl}</span>'
                                f'<span style="color:{theme.MUTED}">masked (blind review)</span>'
                                f'</div>', unsafe_allow_html=True)
                else:
                    st.markdown(C.tracked_value(t, lbl), unsafe_allow_html=True)
            st.markdown(C.tracked_value(p.years_experience, "Total experience"),
                        unsafe_allow_html=True)
            st.markdown(C.tracked_value(p.years_relevant_experience, "Investment-relevant"),
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
                    theme.chip(f"{l.language}" + (f" · {l.proficiency}" if l.proficiency else ""))
                    for l in p.languages), unsafe_allow_html=True)
        with b:
            st.markdown("**Skills** — depth is inferred from how the skill is used, "
                        "not from where it is listed")
            for depth, tone in (("core", "verified"), ("applied", "derived"),
                                ("mentioned", "missing")):
                group = [s for s in p.skills if s.depth == depth]
                if group:
                    st.markdown(f'<span class="mm-sub">{depth}</span><br>'
                                + "".join(theme.chip(s.canonical, tone) for s in group),
                                unsafe_allow_html=True)
        st.divider()
        st.markdown("**Employment**")
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
                items = "".join(
                    f"<li>{html.escape(str(h.value))}</li>" for h in e.highlights)
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
        st.markdown("".join(flow_parts), unsafe_allow_html=True)
        st.markdown("**Education**")
        edu_rows = [{"Institution": e.institution.display("—"),
                     "Degree": e.degree_raw.display("—"), "Level": e.degree_level or "—",
                     "Field": e.field_of_study.display("—"),
                     "Year": e.graduation_year.display("—"), "Result": e.gpa_raw.display("—")}
                    for e in p.education]
        if edu_rows:
            st.dataframe(pd.DataFrame(edu_rows), width="stretch", hide_index=True)

    with doc_col:
        st.markdown('<span class="mm-sub" style="text-transform:uppercase;'
                    'letter-spacing:.04em;font-weight:600">Original document</span>',
                    unsafe_allow_html=True)
        st.caption("Exactly the file that was uploaded — not extracted text, not a "
                  "re-render. Compare it directly against the agent's work on the left.")
        C.original_document_view(p)

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
        st.caption(f"Extracted text after layout repair · {len(p.raw_text)} characters")
        st.text_area("Source", p.raw_text, height=440, label_visibility="collapsed")


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
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
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
- 3-7 years of experience in healthcare equity research or healthcare investment banking
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


def render_requisition(profiles, synth, pool, index, index_manifest, manifest, store,
                       client, bench, evals):
    st.markdown("##### Requisition matching")
    st.caption("Paste a job description. Requirements are parsed, you decide which are "
               "genuinely mandatory, and every score decomposes into its parts.")

    a, b = st.columns([0.48, 0.52])
    with a:
        jd = st.text_area("Job description", SAMPLE_JD, height=280)
        c1, c2 = st.columns(2)
        parse_clicked = c1.button("Parse requisition", type="primary",
                                  width="stretch")
        if c2.button("Use rules only", width="stretch",
                     help="Skip the LLM parse and use the deterministic parser."):
            from millennium.retrieval import parse_query_rules
            st.session_state.requisition = {
                "parsed": parse_query_rules(jd).output, "raw": jd, "method": "rule",
                "requirements": []}
        if parse_clicked:
            st.session_state.requisition = _parse_req(client, jd)

    with b:
        st.markdown("**Weights** — editable, and shown next to every score")
        w = st.session_state.weights
        cols = st.columns(2)
        keys = list(ScoreWeights().model_dump())
        for i, k in enumerate(keys):
            w[k] = cols[i % 2].slider(k.replace("_", " "), 0.0, 0.6, float(w[k]), 0.01,
                                      key=f"w_{k}")
        st.session_state.weights = w
        total = sum(w.values())
        st.caption(f"Raw total {total:.2f} — normalised to 1.00 at scoring time, so you "
                   f"can move one slider without rebalancing the rest.")

    # Role templates freeze a desk's weights and must-have set so the second
    # healthcare L/S req is not re-tuned from scratch — and therefore not scored
    # differently from the first.
    with st.expander("Role templates"):
        templates = store.list_templates()
        tc = st.columns([0.35, 0.2, 0.2, 0.25])
        if templates:
            pick = tc[0].selectbox("Template", ["—"] + [t["name"] for t in templates],
                                   label_visibility="collapsed")
            if pick != "—":
                rec = next(t for t in templates if t["name"] == pick)
                if tc[1].button("Load", width="stretch"):
                    if rec.get("weights"):
                        st.session_state.weights = rec["weights"]
                    st.session_state.requisition = {
                        "parsed": _pq_from_dict(rec.get("parsed_query") or {}),
                        "raw": rec.get("jd", ""), "method": "template",
                        "requirements": rec.get("requirements") or []}
                    st.rerun()
                if tc[2].button("Delete", width="stretch"):
                    store.delete_template(pick)
                    st.rerun()
        else:
            tc[0].caption("No templates saved yet.")
        tname = tc[3].text_input("Save as", placeholder="template name",
                                 label_visibility="collapsed")
        if tname.strip() and st.session_state.requisition:
            if st.button(f"Save '{tname.strip()}' as a role template"):
                r = st.session_state.requisition
                store.save_template(
                    tname.strip(), r.get("raw", ""), st.session_state.weights,
                    {"semantic_text": r["parsed"].semantic_text,
                     "must_have": r["parsed"].must_have,
                     "preferences": r["parsed"].preferences,
                     "exclusions": r["parsed"].exclusions,
                     "interpretation": r["parsed"].interpretation},
                    r.get("requirements", []))
                st.success(f"Saved template '{tname.strip()}' with the current weights "
                           f"and must-have set.")

    req = st.session_state.requisition
    if not req:
        st.info("Parse a requisition to see ranked candidates.")
        return

    pq: ParsedQuery = req["parsed"]
    if req.get("requirements"):
        st.markdown("**Parsed requirements** — tick what is genuinely mandatory")
        edited = st.data_editor(
            pd.DataFrame(req["requirements"]), width="stretch", hide_index=True,
            column_config={"must_have": st.column_config.CheckboxColumn("Must have"),
                           "text": st.column_config.TextColumn("Requirement", width="large"),
                           "quote": st.column_config.TextColumn("Source quote", width="medium")},
            key="req_editor")
        pq = _apply_requirement_edits(pq, edited)

    weights = ScoreWeights(**st.session_state.weights)
    sem = _semantic_scores(index, pq)
    t0 = time.perf_counter()
    out = rank(pool, pq, weights, sem).output
    st.session_state.last_latency_ms = (time.perf_counter() - t0) * 1000
    ranked, excluded = out["ranked"], out["excluded"]
    byid = _byid(pool)

    m = st.columns(4)
    C.kpi(m[0], len(ranked), "ranked")
    C.kpi(m[1], len(excluded), "gated out", "shown with reasons",
          "#B45309" if excluded else theme.ACCENT)
    C.kpi(m[2], f"{ranked[0].total:.3f}" if ranked else "—", "top score")
    C.kpi(m[3], f"{st.session_state.last_latency_ms:.0f} ms", "match latency")

    if ranked:
        ac1, ac2 = st.columns([0.7, 0.3])
        top_n = ac1.number_input(
            "Add top N to shortlist", min_value=1, max_value=min(50, len(ranked)),
            value=min(5, len(ranked)), key="req_top_n", label_visibility="collapsed")
        if ac2.button(f"🤖 Add top {int(top_n)} to shortlist", width="stretch"):
            sl = st.session_state.shortlist
            added = 0
            for i, r in enumerate(ranked[: int(top_n)], 1):
                if r.candidate_id not in sl:
                    sl[r.candidate_id] = {
                        "note": "", "tags": "", "source": "ai",
                        "basis": f"Ranked #{i} of {len(ranked)} for this requisition "
                                f"(score {r.total:.3f}) — {pq.interpretation[:140]}"}
                    added += 1
            st.success(f"Added {added} candidate(s) — {int(top_n) - added} were "
                      f"already on the shortlist. A human still approves every one "
                      f"on the Shortlist page; this only proposes.")

    st.divider()
    for i, r in enumerate(ranked[:20], 1):
        p = byid.get(r.candidate_id)
        if p is None:
            continue
        head = st.columns([0.05, 0.60, 0.35])
        head[0].markdown(f"### {i}")
        head[1].markdown(C.candidate_card(p, st.session_state.blind, r.total),
                         unsafe_allow_html=True)
        with head[2]:
            mx = max((c.contribution for c in r.components), default=0.35)
            st.markdown("".join(C.score_bar(c.name, c.weight, c.score, c.contribution, mx)
                                for c in r.components), unsafe_allow_html=True)
        if r.exclusion_reasons:
            st.markdown(
                '<div class="mm-warn"><b>Unverified must-have.</b> '
                + html.escape("; ".join(x.replace("unverified: ", "")
                                        for x in r.exclusion_reasons))
                + ' — the candidate was kept rather than gated out, because a CV that '
                  'does not mention something is unverified, not unqualified. Thirty '
                  'seconds of checking resolves it.</div>', unsafe_allow_html=True)
        with st.expander("Gaps, evidence, and what would change this"):
            g = gap_analysis(r).output
            gc = st.columns(3)
            gc[0].markdown("**Has**\n\n" + ("\n".join(f"- {x}" for x in g["has"]) or "—"))
            gc[1].markdown("**Lacks**\n\n" + ("\n".join(f"- {x}" for x in g["lacks"]) or "—"))
            gc[2].markdown("**Unknown**\n\n" + ("\n".join(f"- {x}" for x in g["unknown"]) or "—")
                           + "\n\n*Unknown is a research task, not a rejection.*")
            for c in r.components:
                if c.note:
                    st.caption(f"{c.name}: {c.note}")
            if st.button("Why not higher? (minimal edit)", key=f"cf_{r.candidate_id}"):
                res = minimal_edit(pool, pq, weights, r.candidate_id, sem).output
                if res["minimal"]:
                    e = res["minimal"]
                    st.success(f"Rank {e['from_rank']} → **{e['new_rank']}** if you "
                               f"{e['description']}.")
                elif res["edits"]:
                    e = res["edits"][0]
                    st.info(f"Closest single change: {e['description']} → rank {e['new_rank']}.")
                else:
                    st.info("No single requirement change moves this candidate up. "
                            "The ranking is not being driven by one filter.")
                st.caption(res["note"])

    if excluded:
        with st.expander(f"⊘ {len(excluded)} candidate(s) gated out by must-have requirements"):
            for r in excluded:
                p = byid.get(r.candidate_id)
                st.markdown(f"**{html.escape(p.display_name(st.session_state.blind)) if p else r.candidate_id}** — "
                            + "; ".join(html.escape(x) for x in r.exclusion_reasons))

    st.divider()
    st.markdown("##### Weight sensitivity")
    st.caption("Scenario analysis: each weight is perturbed ±0.10 and the ranking "
               "re-run. A candidate whose rank survives every perturbation is a real "
               "match; one that only appears at the top under one exact weight vector "
               "is an artefact of that vector.")
    if st.button("Run sensitivity sweep"):
        s = weight_sensitivity(pool, pq, weights, sem).output
        df = pd.DataFrame(s["stability"])
        if not df.empty:
            df["candidate"] = df["candidate_id"].map(
                lambda c: byid[c].display_name(st.session_state.blind) if c in byid else c)
            st.dataframe(df[["candidate", "base_rank", "max_rank_shift",
                             "mean_abs_shift", "verdict"]],
                         width="stretch", hide_index=True)


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
        st.markdown("##### :material/person_search: Resume matching")
        st.caption(
            f"Ranks the **{len(results)}** candidate(s) currently in view (search + "
            f"filters). Must-haves gate before scoring; gated-out people stay visible. "
            f"**Match & shortlist** writes the top N to the Shortlist tab."
        )
        st.pills("JD templates", list(JD_TEMPLATES), key="jd_templates",
                 on_change=_apply_jd_template,
                 help="Load a ready-made mandate, then edit it.")
        if query.strip():
            if st.button("Use current search as the JD", key="jd_from_query",
                         icon=":material/content_copy:"):
                st.session_state.search_match_jd = query.strip()
                st.rerun()
        st.text_area("Job description", height=140, key="search_match_jd",
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
            st.dataframe(
                pd.DataFrame(rows), hide_index=True, width="stretch",
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        min_value=0.0, max_value=1.0, format="%.3f"),
                    "Why": st.column_config.TextColumn(width="large"),
                })
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
        elif pq is None:
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
        st.markdown("##### :material/upload_file: Import profiles")
        st.caption(
            "PDF and Word run through the extraction pipeline. JSON and CSV are mapped "
            "as **human / unverified** (they never claim to be span-verified). Added "
            "records join the working pool on this rerun — Search, Requisition, and "
            "Analytics pick them up immediately."
        )
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
    "ai": ("🤖", "AI-ranked", "#0F766E"),
    "human": ("☆", "Human-selected", "#475569"),
    "assistant": ("💬", "Added via chat", "#1D4ED8"),
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
        'app. Messages below are drafted and logged, never sent over any network; '
        'the interview invite is a real, downloadable .ics file you send '
        'yourself.</div>', unsafe_allow_html=True)
    email = p.sensitive.email.display("unknown")
    st.caption(f"Recipient on file: {email}")

    dkey = f"draft_{p.candidate_id}"
    if st.button("🤖 Draft with AI", key=f"aidraft_{p.candidate_id}"):
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


# ========================================================================= SHORTLIST
def render_shortlist(profiles, synth, pool, index, index_manifest, manifest, store,
                     client, bench, evals):
    byid = _byid(pool)
    sl = st.session_state.shortlist
    st.markdown(f"##### Shortlist · {len(sl)} candidate(s)")
    if "match_flash" in st.session_state:
        st.success(st.session_state.match_flash)
        del st.session_state["match_flash"]
    if not sl:
        st.info("No candidates shortlisted yet. Add them from Search (tick rows, or "
                "Match to JD), from the chat assistant, or in bulk from Requisition.")
        return
    st.caption("A human curates and approves this list. The tool ranks and drafts; "
              "it does not decide or send anything on its own.")

    for cid in list(sl):
        p = byid.get(cid)
        if p is None:
            continue
        entry = sl[cid]
        entry.setdefault("source", "human")
        entry.setdefault("basis", "")
        a, b = st.columns([0.45, 0.55])
        with a:
            st.markdown(C.candidate_card(p, st.session_state.blind), unsafe_allow_html=True)
            icon, label, colour = _SOURCE_BADGE.get(entry["source"], _SOURCE_BADGE["human"])
            st.markdown(f'<span class="mm-chip" style="background:{colour}18;'
                       f'color:{colour};border-color:{colour}44">{icon} {label}</span>',
                       unsafe_allow_html=True)
            if entry["basis"]:
                st.caption(entry["basis"])
        with b:
            sl[cid]["note"] = st.text_area("Recruiter note", sl[cid].get("note", ""),
                                           key=f"n_{cid}", height=76)
            sl[cid]["tags"] = st.text_input("Tags", sl[cid].get("tags", ""), key=f"t_{cid}",
                                            placeholder="e.g. screen-call, backup, strong-fit")
            if st.button("Remove", key=f"r_{cid}"):
                sl.pop(cid)
                st.rerun()
        with st.expander(f"✉ Outreach & scheduling — "
                         f"{p.display_name(st.session_state.blind)}"):
            _render_outreach(p, store, client)

    C.section_break("Inbox", 3)
    _render_inbox(sl, byid, store)

    C.section_break("Compare", 1)
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
            "Note": sl[cid].get("note", ""), "Tags": sl[cid].get("tags", ""),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)

    d1, d2 = st.columns(2)
    d1.download_button("Download comparison (CSV)", df.to_csv(index=False),
                       "shortlist.csv", "text/csv", width="stretch")
    payload = {"shortlist": [
        {"candidate_id": cid, "note": sl[cid].get("note", ""), "tags": sl[cid].get("tags", ""),
         "profile": json.loads(byid[cid].model_dump_json(exclude={"raw_text"}))}
        for cid in sl if cid in byid]}
    d2.download_button("Download full records (JSON)",
                       json.dumps(payload, indent=1), "shortlist.json",
                       "application/json", width="stretch")
