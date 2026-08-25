"""Headless render tests for every page of the Streamlit app.

These exist because a page that has only ever been rendered by hand, once, is a page
that breaks during the demo. Streamlit swallows nothing -- an exception in a page
surfaces as an error element rather than a crash -- so the assertion below is that
*zero* error elements appear on any page.

The synthetic corpus stands in for parsed output here. That is legitimate for a render
test: `CandidateProfile` objects from the generator have exactly the same shape as
parsed ones, and what is under test is the UI's handling of that shape, not extraction
accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["Overview", "Workflow", "Search", "Candidate", "Requisition", "Shortlist",
         "Intake", "Review", "Analytics", "System"]


@pytest.fixture(scope="module")
def sample_profiles():
    """A small, diverse slice of the synthetic corpus."""
    from millennium.config import SETTINGS
    from millennium.schema import CandidateProfile
    p = SETTINGS.paths.synthetic / "synthetic_candidates.json"
    if not p.exists():
        pytest.skip("run scripts/make_synthetic.py first")
    payload = json.loads(p.read_text())
    profiles = [CandidateProfile.model_validate(c) for c in payload["candidates"][:24]]
    # Exercise the awkward states the UI must handle, not just the happy path.
    profiles[0].quality.needs_human_review = True
    profiles[0].quality.review_reasons = ["low completeness (48% of core fields)"]
    profiles[0].quality.validation_flags = ["contact: no usable contact details found"]
    profiles[1].years_experience.value = None
    profiles[1].years_experience.validation_status = "abstained"
    profiles[1].years_experience.notes = ["no dated employment entries"]
    profiles[2].provenance.injection_flags = ["instruction_override", "white_on_white_text"]
    profiles[3].provenance.near_duplicate_of = ["SYNTHETIC_00019.generated"]
    profiles[4].employment = []          # a profile with nothing to draw a timeline from
    profiles[5].employment_gaps = [{"after": "Acme", "before": "Beta", "months": 14,
                                    "from": "2019-03", "to": "2020-05"}]
    return profiles


def _app(monkeypatch, profiles, page: str):
    from streamlit.testing.v1 import AppTest

    from millennium import app_data
    monkeypatch.setattr(app_data, "load_profiles_from_artifact",
                        lambda path=None: (profiles, {"cost_usd": 0.0913,
                                                      "llm_calls": 30,
                                                      "cost_per_doc_usd": 0.0091}))
    monkeypatch.setattr(app_data, "load_raw_texts", lambda ps: None)
    monkeypatch.setattr(app_data, "load_synthetic", lambda: [])

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.session_state["page"] = page
    at.session_state["selected"] = profiles[0].candidate_id
    at.run()
    return at


def _errors(at) -> list[str]:
    out = [f"EXCEPTION: {e.value}" for e in at.exception]
    out += [f"st.error: {e.value}" for e in at.error]
    return out


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_error(monkeypatch, sample_profiles, page):
    at = _app(monkeypatch, sample_profiles, page)
    errs = _errors(at)
    assert not errs, f"{page} page produced {len(errs)} error(s):\n" + "\n".join(errs[:4])


def test_search_returns_results_and_renders_filters(monkeypatch, sample_profiles):
    at = _app(monkeypatch, sample_profiles, "Search")
    assert not _errors(at)
    assert len(at.sidebar.multiselect) >= 5, "filter rail is missing facets"
    body = " ".join(m.value for m in at.markdown)
    assert "candidate(s)" in body


def test_search_with_a_natural_language_query(monkeypatch, sample_profiles):
    from streamlit.testing.v1 import AppTest

    from millennium import app_data
    monkeypatch.setattr(app_data, "load_profiles_from_artifact",
                        lambda path=None: (sample_profiles, {}))
    monkeypatch.setattr(app_data, "load_raw_texts", lambda ps: None)
    monkeypatch.setattr(app_data, "load_synthetic", lambda: [])
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.session_state["page"] = "Search"
    at.session_state["query"] = "healthcare equity long/short in Asia Pacific"
    at.run()
    assert not _errors(at)
    body = " ".join(m.value for m in at.markdown)
    assert "Interpreted as" in body, "query interpretation banner did not render"


def test_requisition_page_parses_and_ranks(monkeypatch, sample_profiles):
    at = _app(monkeypatch, sample_profiles, "Requisition")
    assert not _errors(at)
    # Weight sliders must all be present and editable.
    labels = {s.label for s in at.slider}
    assert {"skills", "strategy", "sector", "semantic"} <= labels, labels
    at.button(key="FormSubmitter" if False else at.button[0].key).click().run() \
        if at.button else None
    assert not _errors(at)


def test_blind_mode_masks_names_everywhere(monkeypatch, sample_profiles):
    from streamlit.testing.v1 import AppTest

    from millennium import app_data
    monkeypatch.setattr(app_data, "load_profiles_from_artifact",
                        lambda path=None: (sample_profiles, {}))
    monkeypatch.setattr(app_data, "load_raw_texts", lambda ps: None)
    monkeypatch.setattr(app_data, "load_synthetic", lambda: [])
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.session_state["page"] = "Search"
    at.session_state["blind"] = True
    at.run()
    assert not _errors(at)
    body = " ".join(m.value for m in at.markdown)
    real_names = [str(p.sensitive.full_name.value) for p in sample_profiles[:8]
                  if p.sensitive.full_name.is_known]
    leaked = [n for n in real_names if n in body]
    assert not leaked, f"blind mode leaked identities: {leaked[:3]}"


def test_analytics_renders_charts(monkeypatch, sample_profiles):
    at = _app(monkeypatch, sample_profiles, "Analytics")
    assert not _errors(at)
    assert len(at.tabs) >= 4, "analytics tabs missing"


def test_system_page_renders_every_tab(monkeypatch, sample_profiles):
    at = _app(monkeypatch, sample_profiles, "System")
    assert not _errors(at)
    assert len(at.tabs) >= 8, "system tabs missing"
    body = " ".join(m.value for m in at.markdown)
    assert "Fairness by construction" in body or "fairness" in body.lower()


def test_saved_search_restores_query_and_filters(monkeypatch, sample_profiles, tmp_path):
    """A saved search must restore the filter rail, not merely the query text."""
    from streamlit.testing.v1 import AppTest

    from millennium import app_data
    from millennium.store import Store

    store = Store(tmp_path / "s.sqlite")
    store.save_search("APAC pool", "healthcare analyst",
                      {"region": ["Asia-Pacific"], "years": [2.0, 20.0]}, "hybrid")

    monkeypatch.setattr(app_data, "load_profiles_from_artifact",
                        lambda path=None: (sample_profiles, {}))
    monkeypatch.setattr(app_data, "load_raw_texts", lambda ps: None)
    monkeypatch.setattr(app_data, "load_synthetic", lambda: [])

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.session_state["page"] = "Search"
    at.session_state["pending_filters"] = {"region": ["Asia-Pacific"],
                                           "years": [2.0, 20.0],
                                           "employer": ["A Firm That No Longer Exists"]}
    at.run()
    assert not _errors(at)
    assert at.session_state["f_region"] == ["Asia-Pacific"]
    assert at.session_state["f_years"] == (2.0, 20.0)
    # A stale facet value must not resurrect itself as a phantom filter.
    assert at.session_state["f_employer"] == []


def test_role_template_round_trips(tmp_path):
    from millennium.store import Store
    from ui.pages_core import _pq_from_dict

    store = Store(tmp_path / "t.sqlite")
    store.save_template("HC L/S NY", "jd text", {"skills": 0.35, "sector": 0.25},
                        {"semantic_text": "healthcare long short",
                         "must_have": {"sectors": ["healthcare"]},
                         "preferences": {"skills": ["python"]}, "exclusions": {}},
                        [{"text": "3-7 years", "kind": "experience", "must_have": True}])
    t = store.list_templates()[0]
    assert t["weights"]["skills"] == 0.35
    pq = _pq_from_dict(t["parsed_query"])
    assert pq.must_have["sectors"] == ["healthcare"]
    assert pq.method == "template"
    # An older/partial stored shape must load with defaults rather than explode.
    assert _pq_from_dict({}).must_have == {}


def test_every_rendered_filter_is_actually_applied():
    """Guards a silent class of UI bug: a filter widget that renders, looks live, and
    filters nothing. Found exactly that on the 'Feeder path' control."""
    import re
    src = (ROOT / "ui" / "pages_core.py").read_text()
    body = src.split("def _manual_filter")[1].split("\ndef ")[0]
    rendered = set(re.findall(r'F\["(\w+)"\]\s*=\s*st\.', src))
    applied = set(re.findall(r'F\["(\w+)"\]', body))
    dead = rendered - applied
    assert not dead, f"filter widget(s) rendered but never applied: {sorted(dead)}"


def test_feeder_path_filter_excludes(sample_profiles):
    from millennium import taxonomy as tx
    from ui.pages_core import _manual_filter

    base = {k: [] for k in ("region", "strategy", "sector", "skill", "seniority",
                            "approach", "feeder", "employer", "tier", "cert",
                            "degree", "language")}
    base |= {"years": (0.0, 40.0), "include_unknown_years": True,
             "min_completeness": 0.0, "review_only": False}

    assert len(_manual_filter(sample_profiles, base)) == len(sample_profiles)

    target = next(p for p in sample_profiles if p.feeder_path)
    label = tx.FEEDER_PATHS[target.feeder_path.label]["display"]
    kept = _manual_filter(sample_profiles, base | {"feeder": [label]})
    assert kept, "feeder filter excluded everything"
    assert len(kept) < len(sample_profiles), "feeder filter is a no-op"
    for p in kept:
        assert p.feeder_path and tx.FEEDER_PATHS[p.feeder_path.label]["display"] == label


def test_unknown_taxonomy_labels_do_not_crash_the_ui(monkeypatch, sample_profiles):
    """Taxonomies are versioned, so a stored profile can carry a label a later version
    retired. That is a display problem, not a correctness problem, and it must never
    take down a page."""
    import copy

    from millennium.schema import Classification

    profs = copy.deepcopy(sample_profiles)
    profs[0].strategies = [Classification(label="retired_strategy_v0", confidence=0.8)]
    profs[0].sectors = [Classification(label="quantum_widgets", confidence=0.8)]
    profs[0].feeder_path = Classification(label="mystery_pipeline", confidence=0.5)
    profs[0].seniority = Classification(label="L99", confidence=0.5)
    profs[1].seniority = Classification(label="not-a-level", confidence=0.5)
    for e in profs[1].employment:
        e.employer_tier = "tier_from_the_future"

    for page in ("Search", "Candidate", "Analytics", "Shortlist"):
        at = _app(monkeypatch, profs, page)
        assert not _errors(at), f"{page} crashed on an unknown taxonomy label"


def test_display_helper_degrades_gracefully():
    from millennium import taxonomy as tx
    assert tx.display("strategy", "equity_long_short") == "Equity Long/Short"
    assert tx.display("strategy", "retired_label_v0") == "Retired Label V0"
    assert tx.display("region", "apac") == "Asia-Pacific"
    assert tx.display("seniority", "L4") == "Senior Analyst / Associate"
    assert tx.display("seniority", "L99") == "Level 99"
    assert tx.display("seniority", "garbage") == "garbage"
    assert tx.display("tier", "unheard_of") == "Unheard Of"
    assert tx.display("sector", None) == "—"
    assert tx.display("sector", None, "Unknown") == "Unknown"


def test_open_button_navigates_without_the_widget_state_exception(monkeypatch, sample_profiles):
    """Regression: the sidebar nav radio was keyed 'page', the same name as the plain
    session-state variable other widgets used to navigate. Streamlit forbids writing to
    a session-state key once a widget owns it, so clicking a candidate's "Open" button
    raised StreamlitAPIException instead of navigating. Reproduces the exact sequence:
    render Search, click Open on the first result, rerun, land on Candidate with no
    exception and the sidebar radio reflecting the new page."""
    at = _app(monkeypatch, sample_profiles, "Search")
    at.session_state["view"] = "Cards"
    at.run()
    assert not _errors(at)

    open_buttons = [b for b in at.button if b.key and b.key.startswith("o_")]
    assert open_buttons, "no Open button rendered on the Search page"
    open_buttons[0].click().run()

    errs = _errors(at)
    assert not errs, f"navigation raised: {errs}"
    assert at.session_state["page"] == "Candidate"
    radios = [r for r in at.radio if r.key == "nav_radio"]
    assert radios and radios[0].value == "Candidate", \
        "sidebar radio did not follow the programmatic navigation"


def test_results_table_has_a_stable_explicit_key(monkeypatch, sample_profiles):
    """The results table drives navigation on row-click, so it must not rely on
    Streamlit's positional auto-key -- that key shifts if a column is added or removed,
    silently breaking the click handler on the next unrelated change.

    Row-selection itself is not simulated here: this Streamlit version's AppTest
    wrapper for `st.dataframe` exposes only `.run()` and `.value` (verified against
    `streamlit.testing.v1.element_tree.Dataframe`), with no supported way to drive an
    `on_select` event. Faking the internal selection-state shape by hand is guessing at
    an undocumented contract, not testing the app -- an earlier version of this test did
    exactly that and failed on the harness's own internals, not on app code. The
    `test_no_page_key_reused_by_a_widget` guard below covers the actual bug class (a
    widget key colliding with the `page` state variable) generally, regardless of which
    element triggers the write.
    """
    at = _app(monkeypatch, sample_profiles, "Search")
    at.session_state["view"] = "Table"
    at.run()
    assert not _errors(at)
    assert at.dataframe, "results table did not render"
    assert at.dataframe[0].key == "results_table", \
        "results table lost its explicit key -- row-click navigation would silently break"


def test_no_page_key_reused_by_a_widget():
    """Structural guard for the actual bug class, independent of which UI element
    triggers navigation. `page` is the plain, freely-writable session-state variable
    every navigation path (Open button, table row, requisition redirects, ...) sets
    directly; Streamlit forbids writing to a session-state key once a widget owns it.
    If any widget is ever given key="page" again, every one of those writes breaks with
    StreamlitAPIException -- exactly the crash this whole file exists to catch."""
    import re
    for f in (ROOT / "app.py", *sorted((ROOT / "ui").glob("*.py"))):
        src = f.read_text()
        hits = re.findall(r'key\s*=\s*["\']page["\']', src)
        assert not hits, f"{f.relative_to(ROOT)} binds a widget to key='page', which " \
                         f"will crash every programmatic navigation in the app"


def test_clicking_the_radio_itself_still_navigates(monkeypatch, sample_profiles):
    """The other direction: a direct click on the sidebar radio must still update the
    logical `page` state via the on_change sync callback."""
    at = _app(monkeypatch, sample_profiles, "Search")
    radios = [r for r in at.radio if r.key == "nav_radio"]
    assert radios
    radios[0].set_value("Analytics").run()
    assert not _errors(at)
    assert at.session_state["page"] == "Analytics"


def test_candidate_profile_shows_provenance_and_split_view(monkeypatch, sample_profiles):
    """The Profile tab must show HOW the record was parsed (LLM / rule baseline /
    LLM-unavailable) and split into agent-work vs original-document panels, without
    extra clicks beyond opening the tab itself -- both explicit product requirements.

    `sample_profiles` are synthetic benchmark rows with no real file backing them on
    disk, so the right panel is expected to degrade to a clear "not available" message
    here rather than a real document render -- that graceful-degradation path is
    exactly what's under test; the real-render path is covered separately below on an
    actual parsed candidate that does have a file on disk.
    """
    at = _app(monkeypatch, sample_profiles, "Candidate")
    assert not _errors(at)
    body = " ".join(m.value for m in at.markdown)
    assert "Parsed via" in body, "no provenance banner on the Profile tab"
    assert "Agent-extracted profile" in body, "left panel is not labelled"
    assert "Original document" in body, "right panel is not labelled"
    assert "not available in this session" in body, \
        "a synthetic record with no real backing file should degrade gracefully, not silently show nothing"


def test_candidate_profile_renders_the_real_source_file(profiles):
    """On an actually-parsed candidate (one of the ten supplied resumes, still present
    on disk), the right panel must render a real document -- st.pdf for the PDFs,
    the mammoth-converted HTML for the .docx files -- not just the fallback message.

    `st.cache_resource` is process-global and AppTest does not reset it between
    instantiations within one pytest run: `_load_pool` is keyed only on
    `include_synthetic`, so once ANY earlier test has populated that cache entry (the
    `sample_profiles`-based tests above do, repeatedly, with a synthetic dataset that
    happens to be identical across all of them and so never exposes the staleness),
    every later AppTest run with the same key silently gets that stale result back
    regardless of what this test monkeypatches. This test is the first in the suite
    to load a genuinely different dataset through the real app, which is exactly what
    surfaces it -- clearing the cache first is the fix, not a workaround.
    """
    import streamlit as st

    st.cache_resource.clear()

    from streamlit.testing.v1 import AppTest

    from millennium import app_data  # noqa: F401 -- imported for monkeypatch target clarity

    pdf_candidates = [p for p in profiles
                      if p.provenance and p.provenance.source_file.endswith(".pdf")]
    if not pdf_candidates:
        import pytest
        pytest.skip("no PDF-backed candidate in the current export")
    target = pdf_candidates[0]

    import millennium.app_data as ad_module
    orig = ad_module.load_profiles_from_artifact
    try:
        ad_module.load_profiles_from_artifact = lambda path=None: (profiles, {})
        ad_module.load_raw_texts = lambda ps: None
        ad_module.load_synthetic = lambda: []

        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
        at.session_state["page"] = "Candidate"
        at.session_state["selected"] = target.candidate_id
        at.run()
        errs = [f"{e.value}" for e in at.exception] + [f"{e.value}" for e in at.error]
        assert not errs, errs[:3]
        body = " ".join(m.value for m in at.markdown)
        assert "not available in this session" not in body, \
            f"real file for {target.provenance.source_file} failed to resolve/render"
    finally:
        ad_module.load_profiles_from_artifact = orig


def test_review_queue_colour_codes_reasons_flags_and_abstentions(monkeypatch, sample_profiles):
    """Reasons, validation flags and abstained fields must be visually distinguishable
    by category (colour-coded cards), not a flat undifferentiated bullet list."""
    at = _app(monkeypatch, sample_profiles, "Review")
    assert not _errors(at)
    body = " ".join(m.value for m in at.markdown)
    assert "Abstained fields" in body
    assert "border-left:3px" in body, "flags are not rendered as colour-coded cards"


def test_flag_classifier_assigns_distinct_categories():
    """Sanity check on the classifier itself: the specific strings the pipeline is
    known to emit must not all collapse into one bucket, or the colour-coding is
    decorative rather than informative."""
    from ui import theme
    samples = {
        "PIPELINE DEGRADED: subagent(s) failed — parse.llm_identity": "pipeline",
        "INJECTION DEFENCE: neutralised 3 span(s); categories=['role_hijack']": "injection",
        "near-duplicate of Jane Doe (Jaccard 0.71)": "duplicate",
        "18-month gap between Acme and Beta (2019-03 to 2020-09)": "timeline",
        "no usable contact details found": "contact",
        "LLM unavailable -- all fields from this pass abstain": "llm_status",
        "repaired 6 ligature/subsetting artefacts": "repair",
        "proposed value discarded: its quote could not be located": "abstained",
        "low completeness (48% of core fields)": "quality",
    }
    seen = set()
    for text, expected in samples.items():
        got = theme.classify_flag(text)
        assert got == expected, f"{text!r} classified as {got!r}, expected {expected!r}"
        seen.add(got)
    assert len(seen) >= 6, "classifier is not actually differentiating flag types"


def test_intake_page_offers_langchain_orchestration_toggle(monkeypatch, sample_profiles):
    """The orchestration selector must render and switching to it must not error,
    even though actually running it needs a live key (covered separately in
    test_langchain_agents.py)."""
    at = _app(monkeypatch, sample_profiles, "Intake")
    assert not _errors(at)
    orch = [r for r in at.radio if r.label == "Orchestration"]
    assert orch, "no orchestration selector on the Intake page"
    orch[0].set_value("LangChain agent (experimental)").run()
    assert not _errors(at)
    body = " ".join(m.value for m in at.markdown)
    assert "LangChain orchestration mode" in body
