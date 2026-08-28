"""Application shell — SaaS rail + sticky product top bar.

Navigation stays a single `st.radio` keyed `nav_radio` (AppTest and every
programmatic `page = ...` write depend on that split). Visual grouping is CSS.
"""
from __future__ import annotations

import html
import json

import pandas as pd
import streamlit as st

from millennium.config import SETTINGS
from millennium.export import flat_row

# Desk first. Pipeline next. Intelligence last.
NAV_PAGES = [
    "Search", "Candidate", "Shortlist", "Requisition",
    "Intake", "Review",
    "Analytics", "Overview", "Workflow", "System",
]

PAGE_META = {
    "Search": ("Search", "Command the pool in plain English."),
    "Candidate": ("Candidate", "Extracted profile beside the original document."),
    "Shortlist": ("Shortlist", "Human-curated. The tool ranks; it does not decide."),
    "Requisition": ("Requisition", "Paste a JD. You mark what is actually mandatory."),
    "Intake": ("Intake", "Parse a resume through the live pipeline."),
    "Review": ("Review", "Records the pipeline refused to publish silently."),
    "Analytics": ("Analytics", "Coverage, gaps, and data quality across the pool."),
    "Overview": ("Overview", "How ten resumes become an evidence-grounded graph."),
    "Workflow": ("Workflow", "Every pipeline, labelled by what actually produces it."),
    "System": ("System", "Ablation, fairness, cost, and the agent registry."),
}


def _sync_page_from_nav() -> None:
    st.session_state.page = st.session_state.nav_radio


def render_sidebar(*, shortlist_n: int, review_n: int, synth_n: int,
                   on_reset) -> None:
    """Workspace rail: brand, grouped nav, purpose, preferences, desk status."""
    demo = SETTINGS.flags.demo_mode
    st.markdown(
        '<div class="mm-brand-lockup">'
        '<div class="mm-mark">M</div>'
        '<div><div class="mm-brand-name">Millennium</div>'
        '<div class="mm-brand-sub">Candidate intelligence</div></div></div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="mm-rail-snapshot">'
        f'<div><b>{shortlist_n}</b><span>shortlist</span></div>'
        f'<div><b>{review_n}</b><span>review</span></div>'
        f'<div><b>{synth_n}</b><span>synth</span></div>'
        '</div>',
        unsafe_allow_html=True)

    if "nav_radio" not in st.session_state or st.session_state["nav_radio"] != st.session_state.page:
        st.session_state.nav_radio = st.session_state.page

    def _label(p: str) -> str:
        base = p
        if p == "Shortlist" and shortlist_n:
            return f"{base}  ({shortlist_n})"
        if p == "Review" and review_n:
            return f"{base}  ({review_n})"
        return base

    with st.container(key="rail_nav"):
        st.radio("Navigation", NAV_PAGES, key="nav_radio", on_change=_sync_page_from_nav,
                 label_visibility="collapsed", format_func=_label)

    _title, purpose = PAGE_META.get(st.session_state.page, ("", ""))
    st.markdown(
        f'<div class="mm-nav-purpose">{html.escape(purpose)}</div>',
        unsafe_allow_html=True)

    with st.expander("Preferences", expanded=False):
        st.toggle("Blind review mode", key="blind",
                  help="Masks name and contact. The scorer never sees them in any mode.")
        st.toggle("Include synthetic corpus", key="include_synthetic",
                  help="Adds the LLM-generated benchmark corpus. Labelled everywhere "
                       "and excluded from accuracy metrics.")
        if st.session_state.include_synthetic and synth_n:
            st.caption(f"{synth_n} synthetic records in the working pool")
        if st.button("Reset demo", icon=":material/restart_alt:", width="stretch",
                     key="reset_demo_btn"):
            on_reset()

    mode = "Offline replay" if demo else "Live API"
    tone = "mm-status-demo" if demo else "mm-status-live"
    st.markdown(
        f'<div class="mm-rail-foot">'
        f'<span class="mm-status {tone}">{html.escape(mode)}</span>'
        f'<span class="mm-rail-meta">schema {html.escape(SETTINGS.schema_version)}'
        f' · tax {html.escape(SETTINGS.taxonomy_version)}</span></div>',
        unsafe_allow_html=True)


def render_topbar(*, n_pool: int, n_manual: int, review_n: int, abst: int,
                  cost: float, hist: list, on_back, review_profiles, abst_profiles,
                  profiles, manual, pool, blind: bool) -> None:
    """Sticky product bar: history, page identity, compact status pills."""
    page = st.session_state.page
    title, subtitle = PAGE_META.get(page, (page, ""))
    crumb = hist[-1] if hist else ""

    with st.container(key="app_chrome"):
        nav_c, title_c, stat_c = st.columns([0.13, 0.42, 0.45], vertical_alignment="center")
        with nav_c:
            with st.container(horizontal=True, gap="small"):
                st.button("Back", icon=":material/arrow_back:", key="back_btn",
                          on_click=on_back, disabled=not hist,
                          help=f"Back to {crumb}" if crumb else "Nothing to go back to yet")
                if st.button("Home", icon=":material/home:", key="home_btn",
                             help="Return to Search"):
                    st.session_state.page = "Search"
                    st.rerun()
        with title_c:
            crumb_html = (f'<span class="mm-crumb">{html.escape(crumb)} / </span>'
                          if crumb else "")
            st.markdown(
                f'<div class="mm-page-head">'
                f'<div class="mm-page-kicker">Millennium BD</div>'
                f'<div class="mm-page-title">{crumb_html}{html.escape(title)}</div>'
                f'<div class="mm-page-sub">{html.escape(subtitle)}</div></div>',
                unsafe_allow_html=True)
        with stat_c:
            _status_pills(n_pool, n_manual, review_n, abst, cost, review_profiles,
                          abst_profiles, profiles, manual, pool, blind)


def _status_pills(n_pool, n_manual, review_n, abst, cost, review_profiles,
                  abst_profiles, profiles, manual, pool, blind) -> None:
    """Tiny popovers — pool facts without a second dashboard row."""

    def _go(label: str, target: str, key: str) -> None:
        if st.button(label, key=key, width="stretch"):
            st.session_state.page = target
            st.rerun()

    with st.container(horizontal=True, gap="small", key="kpi_band",
                      horizontal_alignment="right"):
        with st.popover(f"**{n_pool}** pool"):
            st.markdown(f"**{len(profiles)}** parsed"
                        + (f" + **{n_manual}** uploaded this session." if n_manual else "."))
            for p in list(profiles)[:12] + list(manual):
                region = p.geo_region.label if p.geo_region else "—"
                st.markdown(f"- **{p.display_name(blind)}** — "
                            f"{p.seniority.label if p.seniority else '—'} · {region}")
            _go("Open Analytics", "Analytics", "kpi_go_analytics")
        rev_lab = f":orange[**{review_n}**] review" if review_n else f"**{review_n}** review"
        with st.popover(rev_lab):
            if review_profiles:
                st.markdown("Waiting for a human eye — not rejected:")
                for p in review_profiles:
                    why = p.quality.review_reasons[0] if p.quality.review_reasons else "flagged"
                    st.markdown(f"- **{p.display_name(blind)}** — {why}")
            else:
                st.markdown("Nothing is waiting for review.")
            _go("Open Review", "Review", "kpi_go_review")
        with st.popover(f"**{abst}** abstained"):
            st.markdown("A value the model proposed whose quote could not be found — "
                        "discarded, not guessed.")
            for p in abst_profiles:
                st.markdown(f"- **{p.display_name(blind)}** — "
                            f"{p.quality.abstention_count} field(s)")
            _go("Open Review", "Review", "kpi_go_review2")
        with st.popover(f"**${cost:.3f}**"):
            n = max(len(profiles), 1)
            st.markdown(f"Parse cost **${cost:.3f}** (≈ ${cost / n:.3f} / resume). "
                        "Cached on disk — this demo replays for $0.00.")
            _go("Open System", "System", "kpi_go_system")
        with st.popover("Export all", icon=":material/download:"):
            include_pii = not blind
            export_pool = list(pool or profiles)
            exclude_fields = {"raw_text"} if include_pii else {"raw_text", "sensitive"}
            payload = {
                "count": len(export_pool),
                "blind_mode": blind,
                "candidates": [
                    json.loads(p.model_dump_json(exclude=exclude_fields))
                    for p in export_pool
                ],
            }
            csv_df = pd.DataFrame([flat_row(p, include_pii=include_pii)
                                   for p in export_pool])
            st.caption("Exports the current working pool from any page.")
            st.download_button(
                "All candidates JSON",
                json.dumps(payload, indent=1, ensure_ascii=False),
                "all_candidates.json",
                "application/json",
                icon=":material/data_object:",
                key="global_export_json",
                width="stretch")
            st.download_button(
                "All candidates CSV",
                csv_df.to_csv(index=False),
                "all_candidates.csv",
                "text/csv",
                icon=":material/table:",
                key="global_export_csv",
                width="stretch")
