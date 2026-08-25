"""Tests for the LangChain multi-agent orchestrator.

Split deliberately into two tiers: the handle-based tool wiring (ingest, injection
scan, section segmentation) needs no API key and no LangChain agent loop at all, so it
runs in every CI environment. The full `AgentExecutor` run needs a live key -- it makes
its own routing decisions, which cannot be replayed from the deterministic disk cache
the rest of this project relies on -- and is skipped cleanly when one isn't present,
matching the pattern already used for the other key-gated tests in this suite.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clear_sessions():
    from millennium import langchain_agents as LA
    LA._SESSIONS.clear()
    yield
    LA._SESSIONS.clear()


# ------------------------------------------------------------- registration regression
def test_subagents_used_by_name_are_actually_registered():
    """The exact bug a live run found: this module calls subagents by name string
    ('parse.llm_identity', 'classify.skills', ...) without importing the modules whose
    `@subagent` decorators register them. The failure mode is silent -- `run_subagent`
    catches the lookup miss and returns a well-formed `status='failed'` AgentResult
    rather than raising, so it looked like a genuine extraction failure rather than an
    import-ordering bug until error detail was added to surface it.

    Run as a subprocess with a completely fresh interpreter, deliberately -- an
    in-process `del sys.modules[...]` + reimport was tried first and it worked, but it
    mutates the shared `REGISTRY` and `SETTINGS` singletons for the rest of the pytest
    session, which broke an unrelated test that happened to run afterward. A subprocess
    is the only way to test "does importing just this module register what it needs"
    without that risk to every other test in the suite.
    """
    import subprocess
    import sys

    required = ["ingest.injection_scan", "parse.segment_sections", "parse.rule_contacts",
               "parse.llm_identity", "parse.llm_employment", "parse.llm_profile",
               "parse.merge_identity", "parse.merge_employment", "parse.llm_adjudicate",
               "classify.skills", "classify.strategy", "classify.sector",
               "classify.geography", "classify.seniority", "classify.quant_profile",
               "classify.feeder_path", "validate.dates", "validate.spans",
               "validate.consistency", "validate.completeness", "validate.route_review"]
    script = (
        "import sys; sys.path.insert(0, 'src')\n"
        "import millennium.langchain_agents\n"
        "from millennium.agents.base import REGISTRY\n"
        "missing = [n for n in " + repr(required) + " if n not in REGISTRY]\n"
        "print('MISSING:' + ','.join(missing))\n")
    r = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr[-2000:]
    missing_line = next(ln for ln in r.stdout.splitlines() if ln.startswith("MISSING:"))
    missing = [x for x in missing_line[len("MISSING:"):].split(",") if x]
    assert not missing, f"importing langchain_agents did not register: {missing}"


# -------------------------------------------------------------------- tool-level tests
def test_ingest_tool_never_returns_document_text():
    """The security property the whole design rests on: every tool's return value
    must be a short structured summary, never raw document content, so the
    orchestrating agent's own context never contains untrusted resume text."""
    from millennium import langchain_agents as LA
    out = LA._tool_ingest_document(str(ROOT / "Omar El-Hassan 202405.pdf"))
    assert "text" not in out and "raw_text" not in out and "content" not in out
    assert out["doc_id"] and out["characters"] > 0
    assert isinstance(out["repairs_applied"], list)


def test_scan_for_injection_tool_returns_categories_not_text():
    from millennium import langchain_agents as LA
    r1 = LA._tool_ingest_document(str(ROOT / "tests" / "fixtures" / "injected_resume.pdf"))
    out = LA._tool_scan_for_injection(r1["doc_id"])
    assert "text" not in out
    assert out["attack_detected"] is True
    assert "instruction_override" in out["flags"] or "role_hijack" in out["flags"]


def test_segment_tool_reports_counts_not_values():
    """Emails/certifications come back as counts and closed-taxonomy labels only --
    never the actual extracted string, which could in principle carry attacker-chosen
    content the model regex happened to match."""
    from millennium import langchain_agents as LA
    r1 = LA._tool_ingest_document(str(ROOT / "Michael Rodriguez, CFA.docx"))
    out = LA._tool_segment_and_scan_contacts(r1["doc_id"])
    assert isinstance(out["emails_found"], int)
    assert "certifications_found" in out
    assert all(isinstance(c, str) and " " not in c for c in out["certifications_found"]), \
        "certification values should be closed-taxonomy keys (e.g. 'cfa'), not raw text"


def test_no_tool_input_schema_accepts_free_text():
    """Structural enforcement of the doc_id-only design: inspect every tool's pydantic
    input schema and confirm none has a field that could carry document content."""
    pytest.importorskip("langchain_anthropic")
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("building the agent (even just to inspect its tools) needs a key")
    from millennium import langchain_agents as LA
    from millennium.llm import LLMClient
    agent = LA._build_agent(LLMClient(demo_mode=True), "test-run")
    for tool in agent.tools:
        fields = getattr(tool.args_schema, "model_fields", {})
        for name in fields:
            assert name in ("doc_id", "file_path"), (
                f"tool {tool.name!r} exposes an unexpected argument {name!r} -- only "
                f"doc_id/file_path handles are permitted, per the design in the "
                f"langchain_agents module docstring")


# ---------------------------------------------------------------- full live agent run
LIVE = bool(os.getenv("ANTHROPIC_API_KEY")) and os.getenv("DEMO_MODE", "1") != "1"


@pytest.mark.skipif(not LIVE, reason="needs ANTHROPIC_API_KEY and DEMO_MODE=0 -- the "
                                     "agent's own routing decisions are live API calls "
                                     "that cannot replay from the deterministic cache")
def test_full_agent_run_produces_a_valid_profile():
    from millennium.langchain_agents import LangChainPipeline
    from millennium.llm import LLMClient

    pipe = LangChainPipeline(client=LLMClient(demo_mode=False))
    profile, report = pipe.process(ROOT / "Omar El-Hassan 202405.pdf")
    assert report["status"] == "ok", report
    assert profile is not None
    assert profile.provenance.extractor == "millennium.langchain_agents/AgentExecutor"
    assert report["n_tool_calls"] >= 7
    assert "finalize_profile" in report["tool_sequence"]
    assert report["tool_sequence"].index("ingest_document") == 0
