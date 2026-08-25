"""Tests for the app-wide chat assistant's tool layer.

Split the same way as the LangChain tests: candidate resolution and the read-only
tools need no API key and no Streamlit runtime, so they run in every CI environment.
The live chat loop (`run_turn`) makes real, unpredictable model calls and is exercised
manually / in the app, matching the pattern for the other key-gated features.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def pool():
    from millennium import app_data
    ps, _ = app_data.load_profiles_from_artifact()
    if not ps:
        pytest.skip("run scripts/run_pipeline.py first")
    app_data.load_raw_texts(ps)
    return ps


def test_case_insensitive_resolution_does_not_pick_the_wrong_person(pool):
    """Regression: without case normalisation, 'Ryan' scored higher (36%) against an
    unrelated 'Omar El-Hassan' (60%) than against the actual 'RYAN PATEL' in the
    corpus (all-caps name), because fuzzy scoring without a processor is
    case-sensitive. Silently resolving to the wrong candidate is the single worst
    failure mode a name-based tool like this can have."""
    from millennium.assistant import _resolve_candidate
    p, why = _resolve_candidate("Ryan", pool)
    assert p is not None and "RYAN PATEL" in p.display_name(False), why


def test_resolution_matches_every_real_candidate_by_partial_name(pool):
    """A bare honorific ('Dr.') is correctly excluded from this check -- it isn't a
    name, and the resolver declining to treat it as a confident match on anyone is
    the right behaviour, not a bug this test should be asserting against."""
    from millennium.assistant import _resolve_candidate
    honorifics = {"dr.", "dr", "mr.", "ms.", "mrs."}
    for target in pool:
        full = target.display_name(False)
        query = next((t for t in full.split() if t.lower().strip(".") not in honorifics
                     and t.lower() not in honorifics), full.split()[0])
        p, why = _resolve_candidate(query, pool)
        assert p is not None and p.candidate_id == target.candidate_id, \
            f"{query!r} should resolve to {full!r}, got {p.display_name(False) if p else None!r} ({why})"


def test_resolution_declines_rather_than_guessing_on_nonsense_input(pool):
    from millennium.assistant import _resolve_candidate
    p, why = _resolve_candidate("qqzzxx99 not a real name", pool)
    assert p is None
    assert "no confident match" in why


def test_pool_summary_reflects_real_counts(pool):
    from millennium.assistant import _tool_get_pool_summary
    out = _tool_get_pool_summary({}, pool, None)
    assert out["total_candidates"] == len(pool)
    assert sum(out["by_region"].values()) <= len(pool)
    assert 0.0 <= out["mean_completeness"] <= 1.0


def test_candidate_summary_never_leaks_protected_attributes(pool):
    """The chat assistant reads real profile data to answer questions -- it must not
    surface name/email/phone as part of that, since a chat transcript is exactly the
    kind of place a protected attribute could leak into a decision-support context
    the fairness design elsewhere in this project explicitly keeps it out of."""
    from millennium.assistant import _tool_get_candidate_summary
    target = pool[0]
    out = _tool_get_candidate_summary({"name_or_id": target.candidate_id}, pool, None)
    forbidden = {"email", "phone", "date_of_birth", "marital_status", "nationality"}
    assert forbidden.isdisjoint(out.keys())
    if target.sensitive.email.is_known:
        assert str(target.sensitive.email.value) not in str(out)


def test_unknown_tool_name_degrades_cleanly():
    from millennium.assistant import execute_tool
    out = execute_tool("delete_everything", {}, [], None)
    assert out["ok"] is False and "unknown tool" in out["error"]


def test_a_tool_that_raises_does_not_crash_the_dispatcher(pool, monkeypatch):
    from millennium import assistant as A
    def boom(inp, pool, store):
        raise RuntimeError("simulated failure")
    monkeypatch.setitem(A._DISPATCH, "get_pool_summary", boom)
    out = A.execute_tool("get_pool_summary", {}, pool, None)
    assert out["ok"] is False and "simulated failure" in out["error"]


def test_run_turn_fails_cleanly_without_a_key(pool, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from millennium.assistant import AssistantUnavailable, run_turn
    with pytest.raises(AssistantUnavailable):
        run_turn([{"role": "user", "content": "hello"}], pool, None)


def test_list_candidates_supports_comparison_questions_in_one_call(pool):
    """Regression: a live 'who is most senior in APAC' query took 7 tool calls across
    5 rounds and ran out of time, because get_pool_summary only gives aggregate
    counts and the model had to look up each candidate individually. list_candidates
    answers exactly this class of question in one call."""
    from millennium.assistant import _tool_list_candidates
    out = _tool_list_candidates({"region": "Asia-Pacific"}, pool, None)
    assert out["count"] >= 1
    apac_names = {c.display_name(False) for c in pool
                 if c.geo_region and "apac" in c.geo_region.label.lower()}
    listed_names = {c["name"] for c in out["candidates"]}
    assert listed_names == apac_names, (listed_names, apac_names)
    for c in out["candidates"]:
        assert "seniority" in c and "years_experience" in c


def test_list_candidates_filters_compose(pool):
    from millennium.assistant import _tool_list_candidates
    out_all = _tool_list_candidates({}, pool, None)
    out_filtered = _tool_list_candidates({"region": "Americas", "min_years": 8}, pool, None)
    assert out_filtered["count"] <= out_all["count"]
    for c in out_filtered["candidates"]:
        assert c["region"] == "Americas"
        assert c["years_experience"] == "unknown" or c["years_experience"] >= 8


def test_shortlist_tools_do_not_depend_on_external_init(pool, monkeypatch):
    """Regression: add_to_shortlist assumed app.py's startup code had already run
    `st.session_state.setdefault('shortlist', {})`. A live test called it from a bare
    context where that never ran, and it failed with an AttributeError that the
    assistant honestly reported as a 'technical error' rather than silently losing
    the action -- correct degradation, but the tool itself needed to be
    self-sufficient rather than depend on unrelated code having run first."""
    import streamlit as st
    if "shortlist" in st.session_state:
        del st.session_state["shortlist"]

    from millennium.assistant import _tool_add_to_shortlist, _tool_list_shortlist
    target = pool[0]
    out = _tool_add_to_shortlist({"name_or_id": target.candidate_id, "note": "test"}, pool, None)
    assert out["ok"], out
    assert st.session_state.shortlist[target.candidate_id]["note"] == "test"
    assert _tool_list_shortlist({}, pool, None)["count"] == 1
