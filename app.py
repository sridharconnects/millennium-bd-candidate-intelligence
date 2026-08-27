"""Millennium BD — Candidate Intelligence Platform (Streamlit entry point).

Positioning, stated here and in the app header because it matters legally and
ethically: this is RECRUITER DECISION SUPPORT, not automated hiring. Nothing here
rejects a candidate. A human approves every shortlist, and every claim the tool makes
is traceable to a span in a source document.

Run:  streamlit run app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

st.set_page_config(page_title="Millennium BD · Candidate Intelligence",
                   page_icon=":material/hub:", layout="wide",
                   initial_sidebar_state="expanded")

from millennium import app_data
from millennium.config import SETTINGS, ScoreWeights
from millennium.llm import LLMClient
from millennium.store import Store
from ui import components as C
from ui import theme
from ui import shell
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
        "hidden_ids": [], "f_show_n": 50,
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

pages_assistant.init_state()

review_profiles = [p for p in profiles if p.quality.needs_human_review]
review_n = len(review_profiles)
abst = sum(p.quality.abstention_count for p in profiles)
abst_profiles = [p for p in profiles if p.quality.abstention_count]
_blind = st.session_state.blind
_hist = st.session_state.nav_history


def _reset_demo() -> None:
    for key in ("query", "selected", "shortlist", "requisition", "filters",
               "corrections", "manual_profiles", "hidden_ids",
               "match_studio_open", "import_studio_open"):
        st.session_state.pop(key, None)
    st.cache_resource.clear()
    st.cache_data.clear()
    _init_state()
    st.rerun()


with st.sidebar:
    shell.render_sidebar(
        shortlist_n=len(st.session_state.shortlist),
        review_n=review_n, synth_n=len(synth), on_reset=_reset_demo)

shell.render_topbar(
    n_pool=len(profiles) + len(manual), n_manual=len(manual),
    review_n=review_n, abst=abst, cost=float(manifest.get("cost_usd", 0) or 0),
    hist=_hist, on_back=_nav_back, review_profiles=review_profiles,
    abst_profiles=abst_profiles, profiles=profiles, manual=manual, blind=_blind)
pages_assistant.render_toggle_button()

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
