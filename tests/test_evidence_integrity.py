"""End-to-end integrity checks over real parsed output.

The first test here guards against the single most damaging possible demo failure:
showing one candidate's resume text as evidence under a different candidate's name.
"""
import pytest


def test_no_evidence_span_leaks_across_candidates(profiles):
    doc_ids = {p.doc_id for p in profiles}
    assert len(doc_ids) == len(profiles), "two candidates share a doc_id"
    for p in profiles:
        for ev in p.all_evidence():
            assert ev.doc_id == p.doc_id, (
                f"EVIDENCE LEAK: {p.candidate_id} displays a span from {ev.doc_id}")


def test_every_span_lands_inside_its_document(profiles):
    for p in profiles:
        if not p.raw_text:
            continue
        for ev in p.all_evidence():
            assert 0 <= ev.char_start < ev.char_end <= len(p.raw_text), (
                f"{p.candidate_id}: span [{ev.char_start}:{ev.char_end}] outside a "
                f"{len(p.raw_text)}-char document")


def test_exact_spans_still_match_the_source_text(profiles):
    bad = []
    for p in profiles:
        if not p.raw_text:
            continue
        for ev in p.all_evidence():
            if ev.match_kind != "exact":
                continue
            actual = p.raw_text[ev.char_start:ev.char_end]
            if actual != ev.snippet:
                bad.append((p.candidate_id, ev.snippet[:40], actual[:40]))
    assert not bad, f"{len(bad)} exact span(s) drifted from the source: {bad[:3]}"


def test_every_displayed_value_is_grounded_or_flagged(profiles):
    """No field may present a value as fact without either a verified span, a
    derivation from verified fields, or an explicit human correction."""
    ungrounded = []
    for p in profiles:
        for name, t in (("full_name", p.sensitive.full_name),
                        ("email", p.sensitive.email),
                        ("headline", p.headline)):
            if t.is_known and not t.evidence and t.validation_status not in (
                    "derived", "human_corrected"):
                ungrounded.append((p.candidate_id, name))
        for e in p.employment:
            for name, t in (("employer", e.employer_raw), ("title", e.title_raw)):
                if t.is_known and not t.evidence and t.validation_status not in (
                        "derived", "human_corrected"):
                    ungrounded.append((p.candidate_id, name))
    assert not ungrounded, f"{len(ungrounded)} ungrounded value(s): {ungrounded[:5]}"


def test_derived_experience_is_never_llm_sourced(profiles):
    """Totals are computed in Python from verified dates. If one ever came back
    marked 'llm', the model was asked to do arithmetic it must not be asked to do."""
    for p in profiles:
        if p.years_experience.is_known:
            assert p.years_experience.extraction_method == "derived", (
                f"{p.candidate_id}: years_experience came from "
                f"{p.years_experience.extraction_method}, not a Python derivation")


def test_attribution_traps_from_the_gold_set_are_avoided(profiles):
    """Companies named inside a bullet, and volunteer/non-profit organisations, must
    not appear as recruiting-relevant employers.

    Checked against `scorable().employers_canonical`, not raw `p.employment` -- a
    volunteer entry (Ryan Patel's non-profit co-founder role, a student club
    presidency) is CORRECT to appear in the full employment history, just flagged
    `is_volunteer=True` and excluded from the list the matching/scoring layer sees.
    Only `employers_canonical` needs to stay clean of both failure modes.
    """
    import json
    from millennium.config import SETTINGS
    gold = json.loads((SETTINGS.paths.gold / "gold_labels.json").read_text())["candidates"]
    by_file = {p.provenance.source_file: p for p in profiles if p.provenance}
    failures = []
    for fname, g in gold.items():
        p = by_file.get(fname)
        if not p:
            continue
        employers = {e.lower() for e in p.scorable().employers_canonical}
        for trap in g.get("must_not_contain", []):
            if trap.lower() in employers:
                failures.append((fname, trap))
    assert not failures, f"attribution traps triggered: {failures}"


def test_volunteer_roles_are_flagged_not_dropped(profiles):
    """The positive half of the same fix: volunteer/non-profit entries are real CV
    content and must survive extraction, just correctly labelled."""
    by_file = {p.provenance.source_file: p for p in profiles if p.provenance}
    ryan = by_file.get("RYAN PATEL - Resume.pdf")
    if ryan is None:
        pytest.skip("Ryan Patel's resume was not in this parsed set")
    volunteer_employers = {(e.employer_canonical or e.employer_raw.value or "").lower()
                           for e in ryan.employment if e.is_volunteer}
    assert any("global education alliance" in v for v in volunteer_employers), \
        "the non-profit co-founder role was not extracted at all -- it should be " \
        "kept and flagged, not dropped"


def test_all_ten_supplied_resumes_produced_a_profile(profiles):
    real = [p for p in profiles if not (p.provenance and p.provenance.is_synthetic)]
    assert len(real) == 10, f"expected 10 real profiles, got {len(real)}"
    for p in real:
        assert p.provenance and p.provenance.schema_version
        assert p.quality.completeness > 0
