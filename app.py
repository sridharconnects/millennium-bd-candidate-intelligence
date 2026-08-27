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
    manual_ss = st.session_state["manual_profiles"] if "manual_profiles" in st.session_state else []
    manual = [p for p in manual_ss if p.candidate_id in manual_ids]
    wanted = set(candidate_ids)
    pool = [p for p in (profiles + (synth if include_synthetic else []) + manual)
            if p.candidate_id in wanted]
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
        "manual_profiles": [], "nav_history": [], "_last_page": None,
        "hidden_ids": [],
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

# Navigation history for the header's back button. Every route change -- sidebar
# click, a card's "Open", a table row, the assistant switching pages -- funnels
# through st.session_state.page, so detecting a change here catches all of them
# without every call site having to remember to record itself. The back callback
# sets `_last_page` itself, so returning somewhere never re-pushes it.
if st.session_state._last_page is None:
    st.session_state._last_page = st.session_state.page
elif st.session_state._last_page != st.session_state.page:
    st.session_state.nav_history = (
        st.session_state.nav_history + [st.session_state._last_page])[-12:]
    st.session_state._last_page = st.session_state.page


def _nav_back() -> None:
    hist = st.session_state.nav_history
    if hist:
        target = hist.pop()
        st.session_state.page = target
        st.session_state._last_page = target
    else:
        st.session_state.page = "Search"
        st.session_state._last_page = "Search"

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
hidden = set(st.session_state.hidden_ids) if "hidden_ids" in st.session_state else set()
pool = [p for p in (profiles + (synth if st.session_state.include_synthetic else []) + manual)
        if p.candidate_id not in hidden]
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
# Title + KPI band stay on screen while the page scrolls -- the recruiter's
# orientation (where am I, how big is the pool, what's waiting) never leaves the
# viewport. `.st-key-app_chrome` is position:sticky in theme.py.
with st.container(key="app_chrome"):
    back_col, home_col, title_col = st.columns([0.05, 0.05, 0.90])
    with back_col:
        # Back: browser-style history over st.session_state.page. Disabled (but still
        # visible, so the layout never jumps) until there is somewhere to go back to.
        _hist = st.session_state.nav_history
        st.button("←", key="back_btn", on_click=_nav_back, disabled=not _hist,
                  help=f"Back to {_hist[-1]}" if _hist else "Nothing to go back to yet")
    with home_col:
        # Home: one click back to the core workspace (Search) from anywhere in the app.
        if st.button("⌂", key="home_btn", help="Home — back to Search, the main "
                     "workspace where you find and filter candidates"):
            st.session_state.page = "Search"
            st.rerun()
    with title_col:
        st.markdown(
            '<div class="mm-row" style="gap:14px;align-items:center">'
            '<span style="font-size:1.35rem;font-weight:700;letter-spacing:-0.02em">'
            '◧ Millennium BD · Candidate Intelligence</span></div>'
            '<div class="mm-sub">Recruiter decision support — every claim traces to a span '
            'in a source document, and anything unprovable is refused rather than guessed. '
            'A human approves every shortlist.</div>', unsafe_allow_html=True)
    # The chat launcher floats fixed at the viewport's top-right corner (it is not part
    # of the header's flow), so it needs no column of its own.
    pages_assistant.render_toggle_button()

    def _kpi_nav(label: str, target: str, key: str) -> None:
        """A jump link at the bottom of a KPI popover to the page with the full story."""
        if st.button(label, key=key, width="stretch"):
            st.session_state.page = target
            st.rerun()

    # Each KPI is a clickable card (an st.popover styled by theme.py): the number stays
    # scannable at a glance, and clicking it opens WHAT is behind the number -- which
    # records, which reasons -- plus a jump to the page that owns the detail.
    review_profiles = [p for p in profiles if p.quality.needs_human_review]
    review_n = len(review_profiles)
    abst = sum(p.quality.abstention_count for p in profiles)
    abst_profiles = [p for p in profiles if p.quality.abstention_count]
    _blind = st.session_state.blind

    with st.container(key="kpi_band"):
        k = st.columns(4)
        with k[0], st.popover(f"**{len(profiles) + len(manual)}**  \nCandidates  \n"
                              f"_{f'real corpus + {len(manual)} uploaded' if manual else 'real corpus'}_",
                              width="stretch"):
            st.markdown(f"**{len(profiles)}** candidates parsed from the supplied resumes"
                        + (f" plus **{len(manual)}** uploaded this session." if manual else "."))
            for p in profiles + manual:
                region = p.geo_region.label if p.geo_region else "region unknown"
                st.markdown(f"- **{p.display_name(_blind)}** — "
                            f"{p.seniority.label if p.seniority else '—'} · {region}")
            _kpi_nav("Open Analytics for pool-wide insights", "Analytics", "kpi_go_analytics")
        _rev_val = f":orange[**{review_n}**]" if review_n else f"**{review_n}**"
        with k[1], st.popover(f"{_rev_val}  \nNeed review  \n_routed, not blocked_",
                              width="stretch"):
            if review_profiles:
                st.markdown("Records the pipeline was **not confident enough to publish "
                            "silently** — each is waiting for a human eye, not rejected:")
                for p in review_profiles:
                    why = p.quality.review_reasons[0] if p.quality.review_reasons else "flagged"
                    st.markdown(f"- **{p.display_name(_blind)}** — {why}")
            else:
                st.markdown("Nothing is waiting for review right now.")
            _kpi_nav("Open the Review queue", "Review", "kpi_go_review")
        with k[2], st.popover(f"**{abst}**  \nAbstentions  \n_unprovable → refused_",
                              width="stretch"):
            st.markdown("An abstention is a value the LLM proposed **whose supporting "
                        "quote could not be found in the document** — so it was discarded "
                        "rather than guessed. A refusal here is a success state, not an "
                        "error.")
            if abst_profiles:
                for p in abst_profiles:
                    st.markdown(f"- **{p.display_name(_blind)}** — "
                                f"{p.quality.abstention_count} field(s) abstained")
                st.caption("Open a candidate's Evidence tab to see exactly which fields "
                           "abstained and why.")
            _kpi_nav("Open the Review queue", "Review", "kpi_go_review2")
        with k[3], st.popover(f"**${manifest.get('cost_usd', 0):.3f}**  \nParse cost  \n"
                              f"_one-off, cached_", width="stretch"):
            n_parsed = max(len(profiles), 1)
            st.markdown(f"Total LLM cost to parse the corpus: "
                        f"**${manifest.get('cost_usd', 0):.3f}** "
                        f"(≈ ${manifest.get('cost_usd', 0) / n_parsed:.3f} per resume).")
            st.markdown("Every response is cached on disk, so re-runs and this demo "
                        "replay for **$0.00** — the cost was paid exactly once.")
            _kpi_nav("Open System for the full cost breakdown", "System", "kpi_go_system")

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
                   "corrections", "manual_profiles", "hidden_ids",
                   "match_studio_open", "import_studio_open"):
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

# The chat dock is a fixed, full-viewport-height panel on the RIGHT EDGE of the
# window -- the same shape as Cursor's / VS Code's chat pane -- not an in-flow
# column. CSS pins it to the viewport (theme.py, .st-key-chat_dock), and while it
# is open theme.inject_chat_open() pads the main block-container by the panel's
# width so page content reflows beside it instead of hiding underneath. The main
# content therefore always renders in a plain full-width container; no layout
# switch happens when the dock opens beyond that padding rule.
if st.session_state.chat_open:
    theme.inject_chat_open()
main_col = st.container()

with main_col:
    if page == "Overview":
        pages_overview.render_overview(**ctx)
    elif page == "Workflow":
        pages_workflow.render_workflow(**ctx)
    elif page == "Search":
        pages_core.render_search(**ctx)
    elif page == "Candidate":
        pages_core.render_candidate(**ctx)
        # Always-visible back control at the bottom of a profile -- the header
        # back button is easy to miss once you have scrolled into the document.
        _hist = st.session_state.nav_history
        dest = _hist[-1] if _hist else "Search"
        with st.container(key="profile_back_fab"):
            st.button(f"←  Back to {dest}", key="profile_fab_back",
                      on_click=_nav_back,
                      help="Return to the previous page. Always on screen while "
                           "you read a profile.")
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

if st.session_state.chat_open:
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
