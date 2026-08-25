"""Agent 3 -- Validation. Decide what we are willing to claim, and what we are not.

Everything here answers one question: would a recruiter be embarrassed if this number
turned out to be wrong? Where the answer is yes and we cannot prove it, we abstain and
say so, rather than degrade quietly.
"""
from __future__ import annotations

from datetime import date

from ..schema import CandidateProfile, QualityReport, Tracked
from ..validate import (check_email, detect_contradictions, find_gaps, find_overlaps,
                        months_between, total_experience_months)
from .base import AgentResult, subagent

AGENT = "validation"

# Fields whose presence defines a usable candidate record. Completeness is measured
# against this list, not against every field in the schema, so a candidate is not
# penalised for omitting a home address.
CORE_FIELDS = ["full_name", "headline", "location_current", "employment", "education",
               "skills", "years_experience", "geography", "seniority"]


@subagent("validate.spans", AGENT, "1.1")
def spans(profile: CandidateProfile) -> AgentResult:
    """Audit evidence integrity across the whole profile.

    Two invariants. First, every piece of evidence must point at THIS candidate's
    document -- cross-candidate evidence leakage would be the single most damaging
    demo failure, so it is checked here and again in tests. Second, every span must
    land inside the document's bounds and its snippet must still match the source.
    """
    errs, warns = [], []
    n_ok = 0
    text_len = len(profile.raw_text)
    for ev in profile.all_evidence():
        if ev.doc_id != profile.doc_id:
            errs.append(f"EVIDENCE LEAK: span belongs to doc {ev.doc_id}, "
                        f"profile is doc {profile.doc_id}")
            continue
        if not (0 <= ev.char_start < ev.char_end <= max(text_len, 1)):
            errs.append(f"span out of bounds: [{ev.char_start}:{ev.char_end}] "
                        f"in a {text_len}-char document")
            continue
        actual = profile.raw_text[ev.char_start:ev.char_end]
        # Full equality, not a prefix check. `snippet` for an "exact" match is defined
        # to BE `text[char_start:char_end]` (see classification._ev and validate.
        # verify_span) -- a prefix check let a since-fixed bug through, where snippet
        # held a padded context window instead of the exact span.
        if ev.match_kind == "exact" and actual != ev.snippet:
            warns.append(f"snippet drifted from source at {ev.char_start}")
        n_ok += 1
    total = n_ok + len(errs)
    return AgentResult(name="", status="failed" if errs else "ok",
                       output={"verified_spans": n_ok, "invalid_spans": len(errs)},
                       confidence=n_ok / total if total else 1.0,
                       errors=errs[:10], warnings=warns[:10])


@subagent("validate.dates", AGENT, "1.2")
def dates(profile: CandidateProfile) -> AgentResult:
    """Derive experience totals in Python and surface timeline anomalies.

    The model is never asked how many years of experience a candidate has. It is a
    computation over verified dates, and where the dates are not verified the answer
    is 'unknown' -- which is a far more useful thing to show a recruiter than a
    confident wrong number.
    """
    spans_ = [(e.employer_canonical or e.employer_raw.display(),
               e.dates.start.normalized_value or e.dates.start.value,
               e.dates.end.normalized_value or e.dates.end.value,
               # Volunteer/non-profit roles (a co-founder title on a charity, a student
               # club presidency) are excluded from experience totals the same way
               # internships are -- they're real CV content, just not professional
               # tenure. See EmploymentEntry.is_volunteer for why this exists.
               e.is_internship or e.is_volunteer) for e in profile.employment]

    total, basis = total_experience_months(spans_)
    warns: list[str] = []

    # Fall back to stated durations only when NO entry is dated, and label it as such.
    if total is None:
        stated = [e.dates.duration_months.value for e in profile.employment
                  if e.dates.duration_months.is_known
                  and not (e.is_internship or e.is_volunteer)]
        if stated:
            total = sum(stated)
            basis = (f"sum of {len(stated)} stated tenure(s); no absolute dates in this "
                     f"document, so concurrency cannot be ruled out")
            warns.append("experience derived from stated durations, not dates -- "
                         "overlapping roles would inflate this figure")

    if total is None:
        years = Tracked.abstain("no dated or duration-bearing employment entries", "derived")
    else:
        years = Tracked.derived(round(total / 12, 1), 0.9 if "union" in basis else 0.6, basis)

    relevant, rbasis = total_experience_months(
        [s for s, e in zip(spans_, profile.employment)
         if (e.employer_tier in ("pod_shop", "quant_fund", "hedge_fund_other", "long_only")
             or any(k in (e.title_raw.value or "").lower()
                    for k in ("research", "analyst", "investment", "portfolio", "quant")))])
    rel = (Tracked.derived(round(relevant / 12, 1), 0.85, rbasis) if relevant is not None
           else Tracked.abstain("no datable investment-relevant roles", "derived"))

    cur = next((e for e in profile.employment if e.dates.is_current), None)
    tenure = Tracked.missing()
    if cur and cur.dates.start.normalized_value:
        m = months_between(cur.dates.start.normalized_value, "present")
        if m is not None:
            tenure = Tracked.derived(m, 0.9, f"current role since {cur.dates.start.normalized_value}")

    gaps = find_gaps(spans_)
    overlaps = find_overlaps(spans_)
    for g in gaps:
        warns.append(f"{g['months']}-month gap between {g['after']} and {g['before']} "
                     f"({g['from']} to {g['to']})")
    for o in overlaps:
        warns.append(f"{o['a']} and {o['b']} overlap by {o['overlap_months']} months -- {o['note']}")

    return AgentResult(name="", output={"years_experience": years,
                                        "years_relevant": rel,
                                        "current_tenure": tenure,
                                        "gaps": gaps, "overlaps": overlaps},
                       confidence=years.confidence, warnings=warns)


@subagent("validate.consistency", AGENT, "1.1")
def consistency(profile: CandidateProfile) -> AgentResult:
    """Cross-field contradiction detection (graduation vs first role, duplicates, ...)."""
    flags = detect_contradictions(profile)
    if profile.sensitive.email.is_known:
        ok, why = check_email(str(profile.sensitive.email.value))
        if not ok:
            flags.append(f"contact: {why}")
    if not profile.sensitive.email.is_known and not profile.sensitive.phone.is_known:
        flags.append("no usable contact details found -- candidate cannot be reached "
                     "from this document alone")
    return AgentResult(name="", status="partial" if flags else "ok",
                       output=flags, confidence=1.0 - min(0.5, 0.1 * len(flags)),
                       warnings=flags)


@subagent("validate.completeness", AGENT, "1.2")
def completeness(profile: CandidateProfile) -> AgentResult:
    """Score record completeness and evidence coverage; decide on human review."""
    present = 0
    missing: list[str] = []
    for f in CORE_FIELDS:
        v = getattr(profile, f, None)
        if v is None:
            v = getattr(profile.sensitive, f, None)
        ok = bool(v.is_known) if isinstance(v, Tracked) else bool(v)
        present += int(ok)
        if not ok:
            missing.append(f)
    comp = round(present / len(CORE_FIELDS), 3)

    tracked = profile.all_tracked()
    known = [t for t in tracked if t.is_known]
    with_ev = [t for t in known if t.evidence]
    coverage = round(len(with_ev) / max(1, len(known)), 3)
    abstained = sum(1 for t in tracked if t.validation_status == "abstained")
    conflicted = sum(1 for t in tracked if t.validation_status == "conflicted")

    return AgentResult(name="", output=QualityReport(
        # extraction_quality is filled in by the orchestrator, which knows the document;
        # provenance is not attached to the profile until the finalize stage, so it must
        # not be read here.
        extraction_quality=0.0,
        completeness=comp, evidence_coverage=coverage,
        abstention_count=abstained, conflict_count=conflicted,
        validation_flags=[f"missing core field: {m}" for m in missing]),
        confidence=comp,
        warnings=[f"{len(missing)} core field(s) unknown: {', '.join(missing)}"] if missing else [])


@subagent("validate.route_review", AGENT, "1.2")
def route_review(profile: CandidateProfile, extraction_quality: float,
                 injection_flags: list[str]) -> AgentResult:
    """Decide whether a human must look at this record before it is trusted.

    Routing is intentionally generous. A record that reaches a recruiter with a silent
    error costs far more than one that asks for thirty seconds of attention.
    """
    reasons: list[str] = []
    if extraction_quality < 0.6:
        reasons.append(f"poor text extraction quality ({extraction_quality:.2f})")
    if profile.quality.completeness < 0.6:
        reasons.append(f"low completeness ({profile.quality.completeness:.0%} of core fields)")
    if profile.quality.evidence_coverage < 0.85:
        reasons.append(f"evidence coverage below threshold ({profile.quality.evidence_coverage:.0%})")
    if profile.quality.abstention_count >= 4:
        reasons.append(f"{profile.quality.abstention_count} fields abstained")
    if profile.quality.conflict_count:
        reasons.append(f"{profile.quality.conflict_count} unresolved rule/LLM conflicts")
    if any(f in ("instruction_override", "role_hijack", "scoring_manipulation",
                 "white_on_white_text", "microscopic_text", "fake_turn_marker")
           for f in injection_flags):
        reasons.append("document contained a prompt-injection payload")
    if not profile.employment:
        reasons.append("no employment history could be grounded")
    if not profile.years_experience.is_known:
        reasons.append("experience total could not be derived from verified dates")
    contradiction_flags = [f for f in profile.quality.validation_flags
                           if not f.startswith("missing core field")]
    if contradiction_flags:
        reasons.append(f"{len(contradiction_flags)} timeline/consistency flag(s)")
    return AgentResult(name="", output={"needs_review": bool(reasons), "reasons": reasons},
                       confidence=1.0,
                       warnings=[f"routed to human review: {r}" for r in reasons])
