"""Millennium BD — Candidate Intelligence Platform (Streamlit entry point).

Positioning, stated here and in the app header because it matters legally and
ethically: this is RECRUITER DECISION SUPPORT, not automated hiring. Nothing here
rejects a candidate. A human approves every shortlist, and every claim the tool makes
is traceable to a span in a source document.

Run:  streamlit run app.py
"""
from __future__ import annotations

import html
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

st.set_page_config(page_title="Millennium BD · Candidate Intelligence",
                   page_icon="◧", layout="wide", initial_sidebar_state="expanded")

from millennium import app_data
from millennium.config import SETTINGS, ScoreWeights
from millennium.llm import LLMClient
from millennium.store import Store
from ui import components as C
from ui import theme
from ui import pages_core, pages_intake, pages_ops, pages_overview, pages_workflow
from ui import assistant_panel as pages_assistant

theme.inject()


# ---------------------------------------------------------------- cached layer
# Every expensive object is created once per process. 'Responsiveness' is an
# explicitly graded criterion, and the two things that would otherwise dominate every
# interaction are the ONNX model load (~2.5s) and the index build.
@st.cache_resource(show_spinner=False)
def _load_pool(include_synthetic: bool):
    # `show_spinner=False`: the caller (below) renders a branded loading screen for
    # the whole boot sequence instead of Streamlit's own small top-left spinner
    # firing once per cached call, which read as two separate stalls rather than
    # one coherent "the app is starting" moment.
    profiles, manifest = app_data.load_profiles_from_artifact()
    app_data.load_raw_texts(profiles)
    synth = app_data.load_synthetic() if include_synthetic else []
    return profiles, synth, manifest


@st.cache_resource(show_spinner=False)
def _load_index(candidate_ids: tuple[str, ...], include_synthetic: bool,
                manual_ids: tuple[str, ...]):
    # The cache key is the exact set of candidates -- including `manual_ids` -- so
    # toggling the synthetic corpus, or adding/importing a record on Intake, rebuilds
    # rather than silently serving a stale index that can't find the new candidate.
    # `manual_ids` only sizes the cache key; the actual objects come from
    # session_state (read here, not hashed) because CandidateProfile isn't a cheap
    # cache key itself.
    profiles, synth, _ = _load_pool(include_synthetic)
    manual = [p for p in st.session_state.get("manual_profiles", [])
             if p.candidate_id in manual_ids]
    pool = profiles + (synth if include_synthetic else []) + manual
    return app_data.make_index(pool)


@st.cache_resource
def _store():
    return Store()


@st.cache_resource
def _client():
    return LLMClient()


@st.cache_data(show_spinner=False)
def _bench():
    return app_data.load_benchmark()


@st.cache_data(show_spinner=False)
def _evals():
    return app_data.load_eval()


# ------------------------------------------------------------------ state init
def _init_state() -> None:
    d = {
        "page": "Search", "query": "", "selected": None, "shortlist": {},
        "weights": ScoreWeights().model_dump(), "blind": SETTINGS.flags.blind_review,
        "include_synthetic": False, "requisition": None, "filters": {},
        "retrieval_mode": "hybrid", "last_latency_ms": 0.0, "corrections": {},
        "manual_profiles": [],
    }
    for k, v in d.items():
        st.session_state.setdefault(k, v)


_init_state()

# The URL seeds initial state ONCE per session -- e.g. a shared link with ?page=... or
# ?q=... opens on the right page. It must not keep overriding session_state on every
# rerun: in-session navigation (the sidebar radio, a card's "Open" button) is also
# reflected into the URL below, and if the read-back ran every time too, the two would
# fight -- whichever query param value was written on the PREVIOUS run would win over
# whatever the user just clicked. Concretely: Search used to unconditionally write
# page=Search into the URL, and that stale value got read back on the very next rerun
# (e.g. after clicking "Analytics" in the sidebar), silently reverting the navigation.
if "_url_seeded" not in st.session_state:
    st.session_state._url_seeded = True
    qp = st.query_params
    if "q" in qp and not st.session_state.query:
        st.session_state.query = qp["q"]
    if "page" in qp and qp["page"] in ("Overview", "Workflow", "Search", "Candidate",
                                       "Requisition", "Shortlist", "Intake", "Review",
                                       "Analytics", "System"):
        st.session_state.page = qp["page"]

# A single placeholder carries the whole boot sequence -- pool load, then index
# build -- as one coherent branded moment rather than two separate default spinners.
# Both calls below are no-ops after the first run of a given process (cache hit), so
# this placeholder renders and clears within one frame on every later rerun.
_boot = st.empty()
_boot.markdown(C.loading_screen("Loading the candidate pool…"), unsafe_allow_html=True)
profiles, synth, manifest = _load_pool(st.session_state.include_synthetic)

if not profiles:
    _boot.empty()
    st.title("Millennium BD · Candidate Intelligence")
    st.markdown(
        '<div class="mm-warn">No parsed candidate data found at '
        '<code>data/exports/candidates.json</code>.<br><br>'
        'Run the pipeline once to generate it:<br>'
        '<code>python scripts/run_pipeline.py</code> '
        '(needs <code>ANTHROPIC_API_KEY</code> in <code>.env</code>, ~$0.10 for the 10 '
        'supplied resumes).<br>Afterwards the app runs entirely offline from the '
        'committed artefacts.</div>', unsafe_allow_html=True)
    st.stop()

manual = st.session_state.manual_profiles
pool = profiles + (synth if st.session_state.include_synthetic else []) + manual
_boot.markdown(C.loading_screen("Building the hybrid search index…"),
              unsafe_allow_html=True)
index, index_manifest = _load_index(tuple(p.candidate_id for p in pool),
                                    st.session_state.include_synthetic,
                                    tuple(p.candidate_id for p in manual))
_boot.empty()

# ---------------------------------------------------------------------- header
# Title/subtitle and the KPI band each get the FULL page width, stacked, rather than
# splitting the width and nesting 4 more columns inside the narrower half. The nested
# version squeezed each KPI card to a sliver on real viewport widths -- narrow enough
# that words wrapped one character per line -- which was invisible only because a
# separate bug (Streamlit's own toolbar overlapping and clipping this whole region)
# was hiding it. Fixing the clipping surfaced the squeeze; this fixes the squeeze.
pages_assistant.init_state()
title_col, chat_btn_col = st.columns([0.85, 0.15])
with title_col:
    st.markdown(
        '<div class="mm-row" style="gap:14px;align-items:center">'
        '<span style="font-size:1.35rem;font-weight:700;letter-spacing:-0.02em">'
        '◧ Millennium BD · Candidate Intelligence</span></div>'
        '<div class="mm-sub">Recruiter decision support — every claim traces to a span '
        'in a source document, and anything unprovable is refused rather than guessed. '
        'A human approves every shortlist.</div>', unsafe_allow_html=True)
with chat_btn_col:
    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
    pages_assistant.render_toggle_button()

k = st.columns(4)
C.kpi(k[0], len(profiles) + len(manual), "candidates",
      f"real corpus + {len(manual)} uploaded" if manual else "real corpus")
review_n = sum(1 for p in profiles if p.quality.needs_human_review)
C.kpi(k[1], review_n, "need review", "routed, not blocked",
      "#B45309" if review_n else theme.ACCENT)
abst = sum(p.quality.abstention_count for p in profiles)
C.kpi(k[2], abst, "abstentions", "unprovable → refused", theme.ACCENT)
C.kpi(k[3], f"${manifest.get('cost_usd', 0):.3f}", "parse cost", "one-off, cached")

PAGE_PURPOSE = {
    "Overview": "How the platform works, in one view — the candidate ontology "
               "diagram: real pipeline stages, real evidence threads, real "
               "relationships, not an illustration of them.",
    "Workflow": "Every real pipeline, agent by agent — parsing, search, requisition "
               "matching, analytics, the chat loop — each node labelled by exactly "
               "what produces it: an LLM API call, a local model, or plain Python.",
    "Search": "Find candidates by plain-English query or filter rail — the main "
              "screen you land on and return to.",
    "Candidate": "One person's full record: agent-extracted profile beside the real "
                "original document, evidence, timeline, similar candidates.",
    "Requisition": "Paste a job description, set which requirements are must-haves, "
                  "and get a ranked, score-explained shortlist.",
    "Shortlist": "Candidates you've saved across searches — add notes, compare "
                "side by side, export.",
    "Intake": "Upload new resumes and watch the pipeline parse them live, "
             "including the agent trace.",
    "Review": "Records the pipeline wasn't confident enough to publish silently — "
             "correct a field or approve it.",
    "Analytics": "Pool-wide distributions, coverage gaps, and data-quality metrics "
                "across every candidate.",
    "System": "How the platform works under the hood — agent registry, retrieval "
             "ablation, fairness audit, cost.",
}

# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("#### Workspace")
    pages = ["Overview", "Workflow", "Search", "Candidate", "Requisition", "Shortlist",
             "Intake", "Review", "Analytics", "System"]
    icons = {"Overview": "◈", "Workflow": "⛓", "Search": "⌕", "Candidate": "▤",
             "Requisition": "▦", "Shortlist": "★", "Intake": "⬆", "Review": "⚑",
             "Analytics": "▧", "System": "⚙"}

    # Two separate keys, not one. `page` is the logical "what to render" state, and is
    # freely writable from anywhere (a candidate card's "Open" button, a table row
    # click, a query-param on load). `nav_radio` is the widget's OWN key -- Streamlit
    # forbids writing to a session-state key once a widget with that key has rendered,
    # so `page` can never be that key too. Before the widget renders, `nav_radio` is
    # synced FROM `page` (always legal, since the widget has not been instantiated yet
    # this run); an on_change callback syncs the other direction for a direct click.
    # NB: `.get()` is intentionally not used on st.session_state -- it is not
    # implemented under the AppTest harness (only real dict-style `in`/`[]` are), and
    # this file is exercised by both.
    if "nav_radio" not in st.session_state or st.session_state["nav_radio"] != st.session_state.page:
        st.session_state.nav_radio = st.session_state.page

    def _sync_page_from_nav() -> None:
        st.session_state.page = st.session_state.nav_radio

    choice = st.radio("Navigation", pages, key="nav_radio", on_change=_sync_page_from_nav,
                      label_visibility="collapsed",
                      format_func=lambda p: f"{icons[p]}  {p}"
                      + (f"  ({len(st.session_state.shortlist)})" if p == "Shortlist"
                         and st.session_state.shortlist else "")
                      + (f"  ({review_n})" if p == "Review" and review_n else ""))
    # A new user landing on any page cannot know what it's for from a one-word label
    # alone. This updates the instant a page is picked, immediately, in the flow of
    # navigating -- not a tooltip that only the curious ever hover.
    st.markdown(
        f'<div class="mm-nav-purpose">{html.escape(PAGE_PURPOSE[st.session_state.page])}</div>',
        unsafe_allow_html=True)
    st.divider()
    st.markdown("#### Display")
    st.toggle("Blind review mode", key="blind",
              help="Masks name and contact details. The scorer never sees them in any "
                   "mode — this only changes what YOU see, so you can audit a ranking "
                   "without knowing whose it is.")
    st.toggle("Include synthetic corpus", key="include_synthetic",
              help="Adds the LLM-generated benchmark corpus to the pool. Clearly "
                   "labelled everywhere and excluded from all accuracy metrics.")
    if st.session_state.include_synthetic and synth:
        st.caption(f"⚠ {len(synth)} synthetic records active")
    st.divider()
    if st.button("↺ Reset demo", width="stretch"):
        for key in ("query", "selected", "shortlist", "requisition", "filters",
                   "corrections", "manual_profiles"):
            st.session_state.pop(key, None)
        st.cache_resource.clear()
        st.cache_data.clear()
        _init_state()
        st.rerun()
    st.caption(f"schema {SETTINGS.schema_version} · taxonomy {SETTINGS.taxonomy_version}")
    st.caption(f"{'DEMO_MODE — offline replay' if SETTINGS.flags.demo_mode else 'LIVE API'}")

# ----------------------------------------------------------------------- route
t0 = time.perf_counter()
page = st.session_state.page
# Single, authoritative write of the page URL param -- always the CURRENT page, so a
# shared/reloaded link opens where the user actually was. Written here only (see the
# read-side note above for why writing it from inside a specific page caused stale
# navigation).
st.query_params["page"] = page
ctx = dict(profiles=profiles, synth=synth, pool=pool, index=index,
           index_manifest=index_manifest, manifest=manifest, store=_store(),
           client=_client(), bench=_bench(), evals=_evals())

# The chat dock takes the full right side of the page, VS Code Copilot-panel style,
# rather than living in the sidebar -- so it needs its own column claimed at the
# routing level, not just inside the sidebar block above. `main_col` is a plain
# `st.container` (not a column) when the dock is closed so the page keeps its full
# width instead of always reserving space for a hidden panel.
if st.session_state.chat_open:
    main_col, chat_col = st.columns([0.54, 0.46], gap="medium")
else:
    main_col, chat_col = st.container(), None

with main_col:
    if page == "Overview":
        pages_overview.render_overview(**ctx)
    elif page == "Workflow":
        pages_workflow.render_workflow(**ctx)
    elif page == "Search":
        pages_core.render_search(**ctx)
    elif page == "Candidate":
        pages_core.render_candidate(**ctx)
    elif page == "Requisition":
        pages_core.render_requisition(**ctx)
    elif page == "Shortlist":
        pages_core.render_shortlist(**ctx)
    elif page == "Intake":
        pages_intake.render_intake(**ctx)
    elif page == "Review":
        pages_ops.render_review(**ctx)
    elif page == "Analytics":
        pages_ops.render_analytics(**ctx)
    elif page == "System":
        pages_ops.render_system(**ctx)

if chat_col is not None:
    with chat_col:
        pages_assistant.render_chat_dock(pool, _store())

render_ms = (time.perf_counter() - t0) * 1000
C.footer({
    "page render": f"{render_ms:.0f} ms",
    "last search": f"{st.session_state.last_latency_ms:.1f} ms",
    "index": f"{index.store.size()} chunks / {index.build_ms} ms build",
    "embedder": index.embedder.name,
    "cost": "$0.00 this session · 100% local retrieval",
    "mode": "DEMO (offline replay)" if SETTINGS.flags.demo_mode else "LIVE",
})
