"""Agent 6 -- Matching. Requisition -> ranked, explained, auditable shortlist.

Three properties a recruiter needs and most ranking tools do not provide:

* **Must-haves gate before scoring, and gated-out candidates stay visible.** Silently
  dropping people is how a pool "runs dry" without anyone noticing that one
  over-strict filter did it. Excluded candidates are returned with the reason.
* **Every score decomposes.** The UI renders weight x component for each dimension,
  so a recruiter can see that a candidate ranked third because of geography rather
  than capability, and re-weight if they disagree.
* **The scorer cannot see who the candidate is.** Its signature accepts only
  `ScorableProfile`, which structurally lacks name, contact, address, marital status
  and nationality. Fairness here is a property of the type, checked by a test, not a
  claim in a document.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from . import taxonomy as tx
from .agents.base import AgentResult, subagent
from .config import ScoreWeights
from .schema import ScorableProfile
from .retrieval import ParsedQuery

AGENT = "matching"


@dataclass
class Component:
    name: str
    weight: float
    score: float
    contribution: float
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class MatchResult:
    candidate_id: str
    total: float
    components: list[Component] = field(default_factory=list)
    semantic_score: float = 0.0
    excluded: bool = False
    exclusion_reasons: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)

    def as_row(self) -> dict:
        d = {"candidate_id": self.candidate_id, "score": round(self.total, 4)}
        for c in self.components:
            d[f"c_{c.name}"] = round(c.contribution, 4)
        return d


def _coverage(have: list[str], want: list[str]) -> tuple[float, list[str], list[str]]:
    """Share of requested items present. Empty request scores neutral, not zero --
    a dimension the recruiter did not ask about must not penalise anybody."""
    if not want:
        return -1.0, [], []
    hv = {str(h).lower() for h in have if h}
    matched = [w for w in want if str(w).lower() in hv]
    missing = [w for w in want if str(w).lower() not in hv]
    return len(matched) / len(want), matched, missing


@subagent("match.score_candidate", AGENT, "1.3")
def score_candidate(sc: ScorableProfile, pq: ParsedQuery, weights: ScoreWeights,
                    semantic: float = 0.0) -> AgentResult:
    """Score ONE candidate against a parsed requisition.

    The type of `sc` is the fairness guarantee: `ScorableProfile` has no field that
    could carry a protected attribute, so no amount of downstream logic can key on one.
    """
    w = weights.normalised()
    prefs = pq.preferences or {}
    musts = pq.must_have or {}

    def want(key: str) -> list[str]:
        return sorted(set((prefs.get(key) or []) + (musts.get(key) or [])))

    comps: list[Component] = []

    def add(name: str, have: list[str], wanted: list[str], note: str = "") -> None:
        cov, matched, missing = _coverage(have, wanted)
        neutral = cov < 0
        s = 0.5 if neutral else cov
        unknown = [] if [h for h in have if h] else (missing if not neutral else [])
        comps.append(Component(
            name=name, weight=w[name], score=s, contribution=w[name] * s,
            matched=matched, missing=[m for m in missing if m not in unknown],
            unknown=unknown,
            note=note or ("not requested — scored neutral" if neutral else "")))

    add("skills", [s.canonical for s in sc.skills], want("skills"))
    add("strategy", sc.strategies, want("strategies"))
    add("sector", sc.sectors, want("sectors"))
    add("geography", [x for x in (sc.geo_region, sc.geography) if x],
        want("geo_regions") + want("countries"))

    # Experience: a band, not a threshold. Being over the band is a much softer miss
    # than being under it, because seniority above target is usually negotiable.
    lo, hi = musts.get("min_years"), musts.get("max_years")
    y = sc.years_experience
    if lo is None and hi is None:
        comps.append(Component("experience", w["experience"], 0.5, w["experience"] * 0.5,
                               note="no experience band requested — scored neutral"))
    elif y is None:
        comps.append(Component("experience", w["experience"], 0.4, w["experience"] * 0.4,
                               unknown=["years_experience"],
                               note="experience could not be derived from verified dates — "
                                    "scored as unknown, not as zero"))
    else:
        lo_f, hi_f = float(lo or 0), float(hi or 99)
        if lo_f <= y <= hi_f:
            s, note = 1.0, f"{y:.1f}y is inside the requested {lo_f:g}–{hi_f:g}y band"
        elif y < lo_f:
            s = max(0.0, 1 - (lo_f - y) / max(lo_f, 1) )
            note = f"{y:.1f}y is {lo_f - y:.1f}y short of the {lo_f:g}y minimum"
        else:
            s = max(0.35, 1 - (y - hi_f) / 20)
            note = f"{y:.1f}y exceeds the {hi_f:g}y ceiling (soft penalty)"
        comps.append(Component("experience", w["experience"], s, w["experience"] * s, note=note))

    comps.append(Component("semantic", w["semantic"], semantic, w["semantic"] * semantic,
                           note="hybrid retrieval score for the free-text intent"))
    dq = sc.data_quality
    comps.append(Component("data_quality", w["data_quality"], dq, w["data_quality"] * dq,
                           note=f"profile completeness {dq:.0%} — a thin CV is ranked "
                                f"lower because we know less, not because it is worse"))

    total = sum(c.contribution for c in comps)
    return AgentResult(name="", output=MatchResult(candidate_id=sc.candidate_id,
                                                   total=round(total, 4), components=comps,
                                                   semantic_score=semantic),
                       confidence=dq)


@subagent("match.rank", AGENT, "1.2")
def rank(profiles: list, pq: ParsedQuery, weights: ScoreWeights,
         semantic: dict[str, float] | None = None) -> AgentResult:
    """Gate on must-haves, score the survivors, and keep the excluded list visible."""
    from .retrieval import apply_filters

    kept, excluded, caveats = apply_filters(profiles, pq)
    semantic = semantic or {}
    lo = min(semantic.values(), default=0.0)
    hi = max(semantic.values(), default=1.0)
    span = (hi - lo) or 1.0

    results: list[MatchResult] = []
    for p in kept:
        norm_sem = (semantic.get(p.candidate_id, lo) - lo) / span if semantic else 0.5
        r = score_candidate(p.scorable(), pq, weights, round(norm_sem, 4)).output
        results.append(r)
    results.sort(key=lambda r: -r.total)

    for r in results:
        if r.candidate_id in caveats:
            # Kept, but a must-have could not be confirmed from the document. Surfaced
            # rather than silently ignored, and it lowers nothing automatically -- a
            # thirty-second check by a recruiter resolves it.
            r.exclusion_reasons = [f"unverified: {c}" for c in caveats[r.candidate_id]]
    ex_results = [MatchResult(candidate_id=e["candidate_id"], total=0.0, excluded=True,
                              exclusion_reasons=e["reasons"]) for e in excluded]
    return AgentResult(name="", output={"ranked": results, "excluded": ex_results,
                                        "caveats": caveats},
                       confidence=1.0 if results else 0.0,
                       warnings=[f"{len(ex_results)} candidate(s) gated out by must-have "
                                 f"requirements — review them if the pool looks thin"]
                       if ex_results else [])


@subagent("match.gap_analysis", AGENT, "1.1")
def gap_analysis(result: MatchResult) -> AgentResult:
    """What this candidate has, lacks, and what we simply do not know about them.

    The third bucket is the one that matters: 'we could not determine whether they
    have a CFA' is a research task, whereas 'they do not have a CFA' is a rejection.
    Collapsing the two is how good candidates get dropped.
    """
    has, lacks, unknown = [], [], []
    for c in result.components:
        has += [f"{c.name}: {m}" for m in c.matched]
        lacks += [f"{c.name}: {m}" for m in c.missing]
        unknown += [f"{c.name}: {u}" for u in c.unknown]
    return AgentResult(name="", output={"has": has, "lacks": lacks, "unknown": unknown})


# --------------------------------------------------------------- counterfactuals
@subagent("match.weight_sensitivity", AGENT, "1.1")
def weight_sensitivity(profiles: list, pq: ParsedQuery, weights: ScoreWeights,
                       semantic: dict | None = None, delta: float = 0.10,
                       top_k: int = 10) -> AgentResult:
    """Perturb each weight +/- delta and measure rank stability.

    Scenario analysis, not prediction. A candidate whose rank survives every plausible
    re-weighting is genuinely a strong match; one who only appears at the top under one
    exact weight vector is an artefact of that vector, and a recruiter should know which
    of the two they are looking at before they pick up the phone.
    """
    base = [r.candidate_id for r in rank(profiles, pq, weights, semantic).output["ranked"]][:top_k]
    moves: dict[str, list[int]] = {cid: [] for cid in base}
    scenarios = []
    for field_name in weights.model_dump():
        for sign in (+1, -1):
            w2 = weights.model_copy(deep=True)
            setattr(w2, field_name, max(0.0, getattr(w2, field_name) + sign * delta))
            order = [r.candidate_id for r in rank(profiles, pq, w2, semantic).output["ranked"]]
            scenarios.append({"weight": field_name, "delta": sign * delta,
                              "top": order[:3]})
            for cid in base:
                new = order.index(cid) if cid in order else len(order)
                moves[cid].append(new - base.index(cid))

    stability = []
    for cid, deltas in moves.items():
        worst = max((abs(d) for d in deltas), default=0)
        stability.append({"candidate_id": cid, "base_rank": base.index(cid) + 1,
                          "max_rank_shift": worst,
                          "mean_abs_shift": round(sum(abs(d) for d in deltas) / max(1, len(deltas)), 2),
                          "verdict": "robust" if worst <= 1 else
                                     "sensitive" if worst <= 3 else "unstable"})
    stability.sort(key=lambda s: s["base_rank"])
    return AgentResult(name="", output={"stability": stability, "scenarios": scenarios,
                                        "delta": delta})


@subagent("match.minimal_edit", AGENT, "1.2")
def minimal_edit(profiles: list, pq: ParsedQuery, weights: ScoreWeights,
                 target_candidate: str, semantic: dict | None = None,
                 target_rank: int = 3) -> AgentResult:
    """Smallest change to the requisition that lifts a candidate into the top-K.

    This is the trade-off a recruiter makes daily -- "if I drop the CFA preference, who
    opens up?" -- expressed as a search over single-requirement removals rather than a
    conversation. Reported as a what-if about the REQUISITION, never as a judgement
    about the person.
    """
    def order_of(q: ParsedQuery) -> list[str]:
        return [r.candidate_id for r in rank(profiles, q, weights, semantic).output["ranked"]]

    base_order = order_of(pq)
    base_rank = base_order.index(target_candidate) + 1 if target_candidate in base_order else None

    edits: list[dict] = []
    for block in ("must_have", "preferences", "exclusions"):
        src = getattr(pq, block) or {}
        for key, vals in src.items():
            if not isinstance(vals, list):
                continue
            for v in vals:
                q2 = copy.deepcopy(pq)
                getattr(q2, block)[key] = [x for x in vals if x != v]
                o = order_of(q2)
                nr = o.index(target_candidate) + 1 if target_candidate in o else None
                if nr and (base_rank is None or nr < base_rank):
                    edits.append({"action": "drop", "block": block, "key": key, "value": v,
                                  "new_rank": nr, "from_rank": base_rank,
                                  "description": f"drop the {block.replace('_',' ')} "
                                                 f"'{v}' ({key})"})
    # Also try relaxing a numeric experience floor, the most common over-constraint.
    if (pq.must_have or {}).get("min_years"):
        q2 = copy.deepcopy(pq)
        q2.must_have["min_years"] = None
        o = order_of(q2)
        nr = o.index(target_candidate) + 1 if target_candidate in o else None
        if nr and (base_rank is None or nr < base_rank):
            edits.append({"action": "relax", "block": "must_have", "key": "min_years",
                          "value": pq.must_have["min_years"], "new_rank": nr,
                          "from_rank": base_rank,
                          "description": f"remove the {pq.must_have['min_years']}y minimum-experience requirement"})

    edits.sort(key=lambda e: (e["new_rank"], e["block"] != "must_have"))
    achieved = [e for e in edits if e["new_rank"] <= target_rank]
    return AgentResult(name="", output={
        "candidate_id": target_candidate, "base_rank": base_rank,
        "target_rank": target_rank, "edits": edits[:8],
        "minimal": achieved[0] if achieved else None,
        "note": "scenario analysis over the requisition, not a prediction about the candidate"})
