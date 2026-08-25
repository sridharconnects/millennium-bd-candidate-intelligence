"""Every optional subsystem must be disableable without breaking the core path.

A Tier-2 feature that can take down the demo is worse than not shipping it.
"""
import importlib
import os

import pytest


@pytest.fixture
def all_flags_off(monkeypatch):
    for k in ("ENABLE_SEMANTIC", "ENABLE_LLM_QUERY", "ENABLE_COUNTERFACTUALS",
              "ENABLE_INJECTION_SCAN", "ENABLE_SYNTHETIC"):
        monkeypatch.setenv(k, "0")
    monkeypatch.setenv("DEMO_MODE", "1")
    import millennium.config as cfg
    importlib.reload(cfg)
    yield cfg
    for k in ("ENABLE_SEMANTIC", "ENABLE_LLM_QUERY", "ENABLE_COUNTERFACTUALS",
              "ENABLE_INJECTION_SCAN", "ENABLE_SYNTHETIC"):
        monkeypatch.delenv(k, raising=False)
    importlib.reload(cfg)


def test_flags_actually_read_the_environment(all_flags_off):
    f = all_flags_off.Settings().flags
    assert not f.enable_semantic and not f.enable_llm_query_parse
    assert not f.enable_injection_scan and not f.enable_counterfactuals


def test_retrieval_still_works_without_the_semantic_embedder():
    """With ENABLE_SEMANTIC off the app must fall back to the hashing embedder and
    keep serving results, not raise."""
    from millennium.index import HashingEmbedder, build_index
    from millennium.retrieval import retrieve
    from millennium.schema import CandidateProfile, Tracked

    p = CandidateProfile(candidate_id="c1", doc_id="d1")
    p.headline = Tracked(value="Healthcare equity long/short analyst",
                         validation_status="verified")
    idx = build_index([p], HashingEmbedder())
    hits = retrieve(idx, "healthcare analyst", "hybrid").output
    assert hits and hits[0].candidate_id == "c1"


def test_injection_scan_can_be_skipped_cleanly():
    from millennium.agents import ingestion  # noqa: F401 -- registers the subagent
    from millennium.agents.base import run_subagent
    from millennium.ingest import Document
    doc = Document(doc_id="d1", source_file="x.docx", file_type="docx", text="hello world")
    r = run_subagent("ingest.injection_scan", doc, None, False)
    assert r.status == "skipped"
    assert r.output["text"] == "hello world"


def test_query_parsing_falls_back_to_rules_when_the_llm_is_unavailable():
    from millennium.llm import LLMClient
    from millennium.retrieval import understand_query
    pq, res = understand_query("healthcare long/short in APAC", LLMClient(demo_mode=True))
    assert pq is not None and pq.method == "rule"
    assert "healthcare" in pq.preferences.get("sectors", []) + pq.must_have.get("sectors", [])


def test_a_failing_subagent_degrades_rather_than_raises():
    from millennium.agents.base import AgentResult, run_subagent, subagent

    @subagent("test.explodes", "testing", "1.0")
    def _boom():
        """Deliberately raises."""
        raise RuntimeError("simulated failure")

    r = run_subagent("test.explodes")
    assert isinstance(r, AgentResult)
    assert r.status == "failed" and r.output is None
    assert any("simulated failure" in e for e in r.errors)
    assert any("degraded" in w for w in r.warnings)


def test_the_whole_app_runs_with_every_optional_flag_off(monkeypatch, tmp_path):
    """The acceptance criterion that matters most on demo day: turn off every optional
    subsystem and the core product must still work end to end.

    `millennium.config.SETTINGS` is a process-wide singleton; every other module binds
    it once via `from .config import SETTINGS` at ITS OWN import time and never looks
    at it again. `importlib.reload(cfg)` only updates `cfg`'s own name binding, so any
    module importing SETTINGS for the FIRST time after this test's reload -- before it
    is ever reloaded back -- silently inherits the all-flags-off object for the rest of
    the pytest session. This bit `langchain_agents.py`: nothing else in the suite
    imports it before this test runs, so it was the first to bind the poisoned
    singleton, and its injection-scan tool consequently ran with the scanner
    permanently (and invisibly) disabled. The `try/finally` below is not decorative --
    remove it and re-run the full suite together with `test_langchain_agents.py` to
    watch the same class of failure reappear.
    """
    import importlib
    import json

    for k in ("ENABLE_SEMANTIC", "ENABLE_LLM_QUERY", "ENABLE_COUNTERFACTUALS",
              "ENABLE_INJECTION_SCAN", "ENABLE_SYNTHETIC"):
        monkeypatch.setenv(k, "0")
    monkeypatch.setenv("DEMO_MODE", "1")

    import millennium.config as cfg
    importlib.reload(cfg)
    try:
        assert not cfg.SETTINGS.flags.enable_semantic

        from millennium import app_data
        from millennium.schema import CandidateProfile

        corpus = cfg.SETTINGS.paths.synthetic / "synthetic_candidates.json"
        if not corpus.exists():
            import pytest
            pytest.skip("run scripts/make_synthetic.py first")
        profiles = [CandidateProfile.model_validate(c)
                    for c in json.loads(corpus.read_text())["candidates"][:12]]

        monkeypatch.setattr(app_data, "load_profiles_from_artifact",
                            lambda path=None: (profiles, {}))
        monkeypatch.setattr(app_data, "load_raw_texts", lambda ps: None)
        monkeypatch.setattr(app_data, "load_synthetic", lambda: [])

        from streamlit.testing.v1 import AppTest
        root = __import__("pathlib").Path(__file__).resolve().parent.parent
        for page in ("Overview", "Workflow", "Search", "Candidate", "Requisition",
                    "Analytics", "System"):
            at = AppTest.from_file(str(root / "app.py"), default_timeout=90)
            at.session_state["page"] = page
            at.session_state["selected"] = profiles[0].candidate_id
            at.session_state["query"] = "healthcare analyst" if page == "Search" else ""
            at.run()
            errs = [f"{e.value}" for e in at.exception] + [f"{e.value}" for e in at.error]
            assert not errs, f"{page} broke with all optional flags off: {errs[:2]}"
    finally:
        # `monkeypatch` only unsets these env vars automatically once THIS function
        # returns -- we are still inside it here, so reloading now with the vars still
        # set would just reload a second, equally-poisoned flags-off Settings object.
        # Unset them explicitly, ourselves, before reloading, mirroring exactly what
        # the `all_flags_off` fixture's own teardown above does.
        for k in ("ENABLE_SEMANTIC", "ENABLE_LLM_QUERY", "ENABLE_COUNTERFACTUALS",
                 "ENABLE_INJECTION_SCAN", "ENABLE_SYNTHETIC", "DEMO_MODE"):
            monkeypatch.delenv(k, raising=False)
        importlib.reload(cfg)
