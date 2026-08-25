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

EXAMPLES = [
    "healthcare equity long/short in APAC, no banking background",
    "quant developer, C++ derivatives pricing, Europe",
    "must have CFA and 5+ years credit or fixed income research",
    "sell-side TMT analyst ready to move buy-side, US",
    "systematic factor research with Python and backtesting",
]


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

    with st.container(border=True):
        qcol, mcol, scol, bcol = st.columns([0.52, 0.16, 0.16, 0.16])
        with qcol:
            query = st.text_input(
                "Search", key="query", label_visibility="collapsed",
                placeholder="Describe the candidate you need — plain English works "
                            "(\"healthcare L/S in APAC, no banking background\")")
        with mcol:
            mode = st.selectbox("Retrieval", ["hybrid", "dense", "lexical"],
                                key="retrieval_mode", label_visibility="collapsed",
                                help="hybrid = RRF fusion of semantic + keyword. Switch "
                                     "to compare; the ablation table on the System page "
                                     "reports which actually wins on the labelled query "
                                     "set.")
        with scol:
            show_n = st.selectbox(
                "Show", [25, 50, 100, 250, "All"], index=1, key="f_show_n",
                label_visibility="collapsed",
                help="How many matches to display. With a large pool (e.g. the "
                     "500-record synthetic corpus), the default only shows the top "
                     "50 — raise this to see more.")
        with bcol:
            st.button("Search", type="primary", width="stretch")
        view = st.segmented_control(
            # `default` is only consulted the first time this key is seen; once the
            # widget's own state exists, Streamlit uses that and ignores `default`. So
            # this does not need (and must not use) a `.get()` read of session_state,
            # which is also unsupported under the AppTest harness this file is tested
            # with.
            "View", ["Table", "Cards"], default="Table",
            key="view", label_visibility="collapsed",
            help="Table for dense scanning and sorting; cards when you want the "
                 "labels and flags at a glance.")

        ex = st.columns(len(EXAMPLES))
        for i, e in enumerate(EXAMPLES):
            if ex[i].button(e if len(e) < 34 else e[:31] + "…", key=f"ex{i}",
                            width="stretch", help=e):
                st.session_state.query = e
                # Must stay INSIDE the `if`. One indent level out and this fires
                # unconditionally on the loop's first iteration, so every script run
                # ends in a rerun that ends in a script run -- an infinite loop that
                # pegs a core at 100% and leaves the page stuck on skeletons forever,
                # with no error anywhere to point at the cause.
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

    if parsed:
        st.markdown(
            f'<div class="mm-banner"><b>Interpreted as</b> — {html.escape(parsed.interpretation)}'
            f' <span class="mm-mono">[{parsed.method} parser]</span></div>',
            unsafe_allow_html=True)

    left, right = st.columns([0.60, 0.40], gap="medium")
    with left, st.container(border=True):
        shown = min(show_cap, len(results))
        st.markdown(f"##### {len(results)} candidate(s) · {latency:.1f} ms")
        if len(results) > shown:
            st.caption(f"showing {shown} of {len(results)} — raise \"Show\" above the "
                      f"search bar to see more")
        if not results:
            st.markdown('<div class="mm-warn">No candidates match. Try removing a '
                        'filter, or search in plain English instead — must-have terms '
                        'gate, preferences only score.</div>', unsafe_allow_html=True)
        if view == "Table":
            _results_table(results[:shown])
        for p, score, explain, chunks in (results[:shown] if view == "Cards" else []):
            st.markdown(C.candidate_card(p, st.session_state.blind,
                                         score if score else None, explain),
                        unsafe_allow_html=True)
            b = st.columns([0.2, 0.2, 0.6])
            if b[0].button("Open", key=f"o_{p.candidate_id}"):
                st.session_state.selected = p.candidate_id
                st.session_state.page = "Candidate"
                st.rerun()
            star = "★ Listed" if p.candidate_id in st.session_state.shortlist else "☆ Shortlist"
            if b[1].button(star, key=f"s_{p.candidate_id}"):
                if p.candidate_id in st.session_state.shortlist:
                    st.session_state.shortlist.pop(p.candidate_id)
                else:
                    st.session_state.shortlist[p.candidate_id] = {
                        "note": "", "tags": "", "source": "human", "basis": ""}
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

    with right, st.container(border=True):
        st.markdown("##### Pool at a glance")
        _mini_charts(results)
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


def _results_table(results) -> None:
    """Dense, sortable results grid with row-selection to open a candidate.

    Cards are good for scanning labels; a table is what a recruiter actually works in
    when comparing twenty people on the same six dimensions. Sorting is native (click
    a header), and selecting a row opens the full profile — so the table is a
    navigation surface, not a dead-end read-only dump.
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
        return
    sel = st.dataframe(
        df.drop(columns=["_id"]), width="stretch", hide_index=True, height=460,
        on_select="rerun", selection_mode="single-row", key="results_table",
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
    if picked:
        st.session_state.selected = df.iloc[picked[0]]["_id"]
        st.session_state.page = "Candidate"
        st.rerun()
    st.caption("Click a column header to sort · click a row to open the full profile · "
               "a blank Yrs cell means unknown, never zero")


def _mini_charts(results) -> None:
    import plotly.express as px
    if not results:
        return
    rows = []
    for p, *_ in results:
        rows.append({
            "region": tx.display("region", p.geo_region.label, "Unknown") if p.geo_region else "Unknown",
            "sector": tx.display("sector", p.sectors[0].label, "Unknown") if p.sectors else "Unknown",
            "years": p.years_experience.value if p.years_experience.is_known else None,
            "seniority": p.seniority.label if p.seniority else "Unknown",
        })
    df = pd.DataFrame(rows)
    for col, title in (("region", "Region"), ("sector", "Primary sector"),
                       ("seniority", "Seniority")):
        vc = df[col].value_counts().reset_index()
        vc.columns = [col, "n"]
        fig = px.bar(vc, x="n", y=col, orientation="h", height=34 * len(vc) + 78,
                     color_discrete_sequence=[theme.ACCENT], text="n")
        fig.update_layout(margin=dict(l=0, r=0, t=26, b=0), title=title,
                          title_font_size=12, showlegend=False,
                          yaxis_title=None, xaxis_title=None, xaxis_visible=False,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(size=11))
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False}, key=f"mini_{col}")


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
    sel = st.selectbox("Candidate", ids, index=ids.index(st.session_state.selected),
                       format_func=lambda i: byid[i].display_name(st.session_state.blind))
    st.session_state.selected = sel
    p = byid[sel]

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
    with head[1]:
        k = st.columns(4)
        C.kpi(k[0], f"{p.years_experience.value:.1f}" if p.years_experience.is_known else "—",
              "years exp", "derived" if p.years_experience.is_known else "unknown")
        C.kpi(k[1], f"{p.quality.completeness:.0%}", "complete")
        C.kpi(k[2], f"{p.quality.evidence_coverage:.0%}", "evidenced")
        C.kpi(k[3], p.quality.abstention_count, "abstained",
              colour="#B45309" if p.quality.abstention_count else theme.ACCENT)

    tabs = st.tabs(["Profile", "Evidence", "Timeline", "Similar", "Lineage", "Source"])

    with tabs[0]:
        C.provenance_banner(p)
        if p.quality.validation_flags:
            with st.expander(f"⚑ {len(p.quality.validation_flags)} validation flag(s) "
                             f"on this record", expanded=False):
                st.markdown(theme.flag_list(p.quality.validation_flags),
                           unsafe_allow_html=True)
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
        for e in p.employment:
            dates = f"{e.dates.start.normalized_value or '?'} → {e.dates.end.normalized_value or '?'}"
            if not e.dates.start.is_known and e.dates.duration_months.is_known:
                dates = (f"{e.dates.duration_months.value} months stated · "
                         f"absolute dates unknown")
            tier = tx.display("tier", e.employer_tier, "")
            st.markdown(
                f'<div class="mm-card"><div class="mm-row" style="justify-content:space-between">'
                f'<span class="mm-name">{html.escape(e.title_raw.display("—"))}</span>'
                f'<span class="mm-sub mm-mono">{html.escape(dates)}</span></div>'
                f'<div class="mm-sub">{html.escape(e.employer_canonical or e.employer_raw.display("—"))}'
                f'{" · " + html.escape(tier) if tier and tier != "Unknown" else ""}'
                f'{" · " + html.escape(e.location.display("")) if e.location.is_known else ""}'
                f'{" · L" + str(e.seniority_level) if e.seniority_level else ""}'
                f'{" · internship" if e.is_internship else ""}</div></div>',
                unsafe_allow_html=True)
            if e.highlights:
                with st.expander(f"{len(e.highlights)} grounded highlight(s)"):
                    for h in e.highlights:
                        st.markdown(f"- {html.escape(str(h.value))}")
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
    if not sl:
        st.info("No candidates shortlisted yet. Add them from Search, from the chat "
                "assistant, or in bulk from Requisition's ranked results.")
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
