"""Agent 7 -- Insight. Pool-level analytics, including the question nobody asks for.

Distributions are table stakes. The genuinely useful output for a BD team is coverage
gap detection: not "here is the shape of the pool" but "you have twelve people for
your equity L/S reqs and one for credit, and that is where sourcing should go next".
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

from .. import taxonomy as tx
from .base import AgentResult, subagent

AGENT = "insight"


@subagent("insight.distributions", AGENT, "1.1")
def distributions(profiles: list) -> AgentResult:
    """Facet counts across every searchable dimension, for charts and filter rails."""
    d: dict[str, Counter] = defaultdict(Counter)
    for p in profiles:
        if p.geo_region:
            d["region"][tx.display("region", p.geo_region.label)] += 1
        if p.geography:
            d["country"][p.geography.label] += 1
        if p.seniority:
            lvl = int(p.seniority.label[1:]) if p.seniority.label.startswith("L") else None
            if lvl:
                d["seniority"][f"L{lvl} · {tx.display("seniority", lvl)}"] += 1
        if p.quant_fundamental:
            d["approach"][p.quant_fundamental.label.title()] += 1
        if p.feeder_path:
            d["feeder"][tx.display("feeder", p.feeder_path.label)] += 1
        for c in p.strategies:
            d["strategy"][tx.display("strategy", c.label)] += 1
        for c in p.sectors:
            d["sector"][tx.display("sector", c.label)] += 1
        for s in p.skills:
            d["skill"][s.canonical] += 1
            if s.depth in ("core", "applied"):
                d["skill_deep"][s.canonical] += 1
        for e in p.education:
            if e.degree_level:
                d["degree"][e.degree_level] += 1
        for e in p.employment:
            if e.employer_tier and e.employer_tier != "unknown":
                d["employer_tier"][tx.display("tier", e.employer_tier)] += 1
            if e.employer_canonical:
                d["employer"][e.employer_canonical] += 1
        for c in p.certifications:
            if c.canonical:
                d["certification"][tx.display("certification", c.canonical)] += 1
        for l in p.languages:
            d["language"][l.language] += 1
        if p.years_experience.is_known:
            y = p.years_experience.value
            bucket = "0-2y" if y < 2 else "2-5y" if y < 5 else "5-8y" if y < 8 else "8-12y" if y < 12 else "12y+"
            d["experience_band"][bucket] += 1
        else:
            d["experience_band"]["unknown"] += 1
    return AgentResult(name="", output={k: dict(v.most_common()) for k, v in d.items()})


@subagent("insight.skill_cooccurrence", AGENT, "1.0")
def skill_cooccurrence(profiles: list, min_count: int = 2) -> AgentResult:
    """Which capabilities travel together in this pool -- shapes realistic requisitions."""
    pairs: Counter = Counter()
    for p in profiles:
        deep = sorted({s.canonical for s in p.skills if s.depth in ("core", "applied")})
        for a, b in combinations(deep, 2):
            pairs[(a, b)] += 1
    top = [{"a": a, "b": b, "count": n} for (a, b), n in pairs.most_common(40) if n >= min_count]
    return AgentResult(name="", output=top)


@subagent("insight.coverage_gaps", AGENT, "1.2")
def coverage_gaps(profiles: list, thin_threshold: int = 2) -> AgentResult:
    """Find the strategy x sector x region cells where the pool is thin or empty.

    This inverts the usual analytics framing. A recruiter does not need to be told
    that most of their pool is in equity research; they need to be told which
    requisitions they currently cannot fill.
    """
    grid: Counter = Counter()
    for p in profiles:
        region = p.geo_region.label if p.geo_region else "unknown"
        for s in (p.strategies or []):
            for sec in (p.sectors or [{}]):
                key = (s.label, getattr(sec, "label", "any"), region)
                grid[key] += 1

    strat_counts = Counter()
    sector_counts = Counter()
    region_counts = Counter()
    for p in profiles:
        for s in p.strategies:
            strat_counts[s.label] += 1
        for s in p.sectors:
            sector_counts[s.label] += 1
        if p.geo_region:
            region_counts[p.geo_region.label] += 1

    gaps = []
    for label in tx.STRATEGIES:
        n = strat_counts.get(label, 0)
        if n <= thin_threshold:
            gaps.append({"dimension": "strategy", "label": tx.display("strategy", label),
                         "key": label, "count": n,
                         "severity": "none" if n == 0 else "thin"})
    for label in tx.SECTORS:
        n = sector_counts.get(label, 0)
        if n <= thin_threshold:
            gaps.append({"dimension": "sector", "label": tx.display("sector", label),
                         "key": label, "count": n,
                         "severity": "none" if n == 0 else "thin"})
    for label in ("americas", "emea", "apac"):
        n = region_counts.get(label, 0)
        if n <= thin_threshold:
            gaps.append({"dimension": "region", "label": tx.display("region", label),
                         "key": label, "count": n,
                         "severity": "none" if n == 0 else "thin"})
    gaps.sort(key=lambda g: (g["count"], g["dimension"]))
    covered = [{"cell": f"{tx.display("strategy", a)} · {tx.display("sector", b) if b in tx.SECTORS else b} · {tx.REGION_DISPLAY.get(c, c)}",
                "count": n} for (a, b, c), n in grid.most_common(12)]
    return AgentResult(name="", output={"gaps": gaps, "strongest_cells": covered})


@subagent("insight.data_quality", AGENT, "1.1")
def data_quality(profiles: list) -> AgentResult:
    """Pool-level honesty metrics: abstention, coverage, review load."""
    n = len(profiles) or 1
    return AgentResult(name="", output={
        "candidates": len(profiles),
        "mean_completeness": round(sum(p.quality.completeness for p in profiles) / n, 3),
        "mean_evidence_coverage": round(sum(p.quality.evidence_coverage for p in profiles) / n, 3),
        "total_abstentions": sum(p.quality.abstention_count for p in profiles),
        "needs_review": sum(1 for p in profiles if p.quality.needs_human_review),
        "with_contact": sum(1 for p in profiles
                            if p.sensitive.email.is_known or p.sensitive.phone.is_known),
        "with_experience_total": sum(1 for p in profiles if p.years_experience.is_known),
        "flagged_injection": sum(1 for p in profiles
                                 if p.provenance and p.provenance.injection_flags),
        "near_duplicates": sum(1 for p in profiles
                               if p.provenance and p.provenance.near_duplicate_of),
    })
