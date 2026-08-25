"""Agent 4 -- Classification.

Rule-and-evidence based labelling that runs *alongside* the LLM's own classification.
Where both agree, confidence rises; where they disagree, confidence falls and the
field is routed to review. Every label records the trigger that fired, so a recruiter
can see exactly why a candidate was tagged 'statistical_arbitrage' and overrule it.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .. import taxonomy as tx
from ..schema import Classification, Evidence, SkillEntry
from ..validate import verify_span
from .base import AgentResult, subagent

AGENT = "classification"


def _ev(text: str, doc_id: str, start: int, end: int) -> Evidence:
    """Evidence for a rule/taxonomy hit at an exact character span.

    `snippet` MUST equal `text[char_start:char_end]` verbatim -- that is the contract
    `match_kind="exact"` makes, checked by tests/test_evidence_integrity.py and by
    `validate.spans`. It previously stored a padded, whitespace-cleaned CONTEXT window
    instead (start-60 to end+60) while still claiming "exact", so it silently failed
    its own invariant on every multi-word skill/strategy/sector hit. A contextual
    excerpt is genuinely useful for humans, but it belongs in a separate field, not
    smuggled into `snippet` under a false label -- the evidence viewer already builds
    its own context window directly from `raw_text` (see ui/components.evidence_block)
    and never reads `.snippet` at all, so nothing downstream needed the padded version.
    """
    return Evidence(doc_id=doc_id, char_start=start, char_end=end,
                    snippet=text[start:end], match_kind="exact")


@subagent("classify.skills", AGENT, "1.2")
def skills(text: str, doc_id: str, llm_skills: list[dict] | None = None) -> AgentResult:
    """Alias-map surface forms to canonical skills and grade depth by usage.

    Depth is evidence-driven rather than self-reported: a skill named inside a
    described task counts as 'applied'; one that recurs across several roles is
    'core'; one that only appears in a comma-separated tools list is 'mentioned'.
    That distinction is what lets a recruiter filter for people who have actually
    used kdb+ rather than people who typed it.
    """
    hits = tx.find_skills(text)
    by_canon: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for canon, surface, s, e in hits:
        by_canon[canon].append((surface, s, e))

    # A "tools list" line is dense with commas and short: mentions there are shallow.
    lines = text.split("\n")
    line_bounds, pos = [], 0
    for ln in lines:
        line_bounds.append((pos, pos + len(ln), ln))
        pos += len(ln) + 1

    def context_depth(start: int) -> str:
        for a, b, ln in line_bounds:
            if a <= start < b:
                if ln.count(",") >= 3 and len(ln) < 260:
                    return "mentioned"
                return "applied" if len(ln) > 60 else "mentioned"
        return "mentioned"

    out: list[SkillEntry] = []
    llm_names = {tx.norm(s.get("name", "")) for s in (llm_skills or [])}
    llm_depth = {tx.norm(s.get("name", "")): s.get("depth") for s in (llm_skills or [])}

    for canon, occ in by_canon.items():
        depths = [context_depth(s) for _, s, _ in occ]
        n_applied = depths.count("applied")
        if n_applied >= 2 or len(occ) >= 4:
            depth = "core"
        elif n_applied >= 1:
            depth = "applied"
        else:
            depth = "mentioned"
        # The model may have seen depth we cannot: take the stronger of the two.
        order = {"mentioned": 0, "applied": 1, "core": 2}
        for alias in tx.SKILLS[canon]["aliases"]:
            d = llm_depth.get(tx.norm(alias))
            if d and order.get(d, 0) > order[depth]:
                depth = d
        out.append(SkillEntry(
            canonical=canon,
            surface_forms=sorted({s for s, _, _ in occ}),
            category=tx.SKILLS[canon]["category"],
            depth=depth,
            evidence=[_ev(text, doc_id, s, e) for _, s, e in occ[:3]],
        ))

    agreement = (len({tx.norm(a) for c in by_canon for a in tx.SKILLS[c]["aliases"]} & llm_names)
                 / max(1, len(llm_names))) if llm_names else 1.0
    out.sort(key=lambda s: ({"core": 0, "applied": 1, "mentioned": 2}[s.depth], s.canonical))
    return AgentResult(name="", output=out, confidence=round(0.6 + 0.4 * agreement, 3),
                       evidence=[e for s in out for e in s.evidence[:1]])


def _label_set(text: str, doc_id: str, finder, table: dict,
               llm_items: list[dict] | None, min_hits: int = 1) -> list[Classification]:
    counts: Counter = Counter()
    spans: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for label, surface, s, e in finder(text):
        counts[label] += 1
        spans[label].append((surface, s, e))

    llm_map = {i.get("label"): i for i in (llm_items or []) if i.get("label") in table}
    out: list[Classification] = []
    for label in set(counts) | set(llm_map):
        n = counts.get(label, 0)
        li = llm_map.get(label)
        # Confidence blends lexical support with the model's own read. Agreement is
        # what earns a high score; a label supported by only one side stays modest.
        rule_conf = min(0.85, 0.35 + 0.14 * n) if n else 0.0
        llm_conf = float(li.get("confidence", 0.6)) if li else 0.0
        if n and li:
            conf, why = min(0.97, 0.55 + 0.25 * min(1, n / 3) + 0.25 * llm_conf), "rule+LLM agree"
        elif n:
            conf, why = rule_conf, f"{n} lexical trigger(s) only"
        else:
            conf, why = min(0.62, llm_conf), "LLM only, no lexical trigger"
        triggers = sorted({s for s, _, _ in spans.get(label, [])})[:6]
        ev = [_ev(text, doc_id, s, e) for _, s, e in spans.get(label, [])[:2]]
        if not ev and li and li.get("quote"):
            m = verify_span(li["quote"], text, doc_id)
            if m:
                ev = [m]
            else:
                continue  # LLM label with an unverifiable quote is dropped outright
        low = (n < min_hits and not li) or bool(li and li.get("low_support"))
        out.append(Classification(
            label=label, confidence=round(conf, 3),
            rationale=(li or {}).get("rationale") or why,
            triggers=triggers, evidence=ev, low_support=low))
    out.sort(key=lambda c: -c.confidence)
    return out


@subagent("classify.strategy", AGENT, "1.2")
def strategy(text: str, doc_id: str, llm_items=None) -> AgentResult:
    """Assign investment-strategy labels from the closed Millennium taxonomy."""
    out = _label_set(text, doc_id, tx.find_strategies, tx.STRATEGIES, llm_items, min_hits=2)
    return AgentResult(name="", output=out,
                       confidence=max([c.confidence for c in out], default=0.0),
                       warnings=[] if out else ["no investment strategy could be evidenced"])


@subagent("classify.sector", AGENT, "1.2")
def sector(text: str, doc_id: str, llm_items=None) -> AgentResult:
    """Assign GICS-lite sector coverage labels."""
    out = _label_set(text, doc_id, tx.find_sectors, tx.SECTORS, llm_items, min_hits=2)
    return AgentResult(name="", output=out,
                       confidence=max([c.confidence for c in out], default=0.0))


@subagent("classify.geography", AGENT, "1.1")
def geography(text: str, doc_id: str, employment: list, llm_geo: dict | None = None) -> AgentResult:
    """Resolve the candidate's *current* market, not merely every place mentioned.

    Weighting is deliberate: the location on the most recent role outranks the header,
    which outranks any other mention. A CV listing a Mumbai education and a London job
    should surface under EMEA, and 'where do they work now' is the question a BD
    recruiter is actually asking.
    """
    votes: Counter = Counter()
    spans: dict[str, tuple[int, int]] = {}
    for country, region, surface, s, e in tx.match_geography(text):
        w = 3.0 if s < 400 else 1.0
        votes[(country, region)] += w
        spans.setdefault(country, (s, e))

    for i, e in enumerate(employment[:2]):
        loc = e.location.value if e.location.is_known else None
        if not loc:
            continue
        for country, region, _, _, _ in tx.match_geography(loc):
            votes[(country, region)] += 12.0 if i == 0 else 6.0

    if llm_geo and llm_geo.get("value"):
        for country, region, _, _, _ in tx.match_geography(str(llm_geo["value"])):
            votes[(country, region)] += 8.0

    if not votes:
        return AgentResult(name="", status="partial", output=(None, None),
                           warnings=["no geography could be evidenced"])
    (country, region), score = votes.most_common(1)[0]
    total = sum(votes.values()) or 1
    conf = round(min(0.96, 0.4 + 0.6 * (score / total)), 3)
    s, e = spans.get(country, (0, 0))
    ev = [_ev(text, doc_id, s, e)] if e else []
    return AgentResult(
        name="", confidence=conf,
        output=(Classification(label=country, confidence=conf, evidence=ev,
                               rationale=f"weighted vote {score:.0f}/{total:.0f}; most recent role location dominates",
                               triggers=[country]),
                Classification(label=region, confidence=conf,
                               rationale=tx.REGION_DISPLAY.get(region, region), evidence=ev)))


@subagent("classify.seniority", AGENT, "1.2")
def seniority(employment: list, years: float | None) -> AgentResult:
    """Normalise the most recent title to level 1-7, adjusted for employer tier."""
    real = [e for e in employment if not e.is_internship]
    if not real:
        return AgentResult(name="", status="partial", output=None,
                           warnings=["no non-internship role to derive seniority from"])
    cur = real[0]
    title = cur.title_raw.value or ""
    tier = cur.employer_tier or "unknown"
    level, why = tx.title_to_level(title, tier)

    # Tenure sanity: a level-6 title after 18 months is title inflation more often
    # than it is a genuine PM seat, so we note the tension rather than silently trust.
    notes = [why]
    if years is not None:
        if level >= 6 and years < 5:
            notes.append(f"title implies L{level} but only {years:.1f}y total experience "
                         f"-- flagged for reviewer confirmation")
        if level <= 2 and years > 8:
            level = min(7, level + 1)
            notes.append(f"raised one level: {years:.1f}y experience is inconsistent with a junior title")
    conf = 0.85 if cur.title_raw.is_known else 0.4
    return AgentResult(name="", confidence=conf, evidence=cur.title_raw.evidence[:1],
                       output=Classification(label=f"L{level}", confidence=conf,
                                             rationale="; ".join(notes),
                                             triggers=[title, tx.TIER_DISPLAY.get(tier, tier)],
                                             evidence=cur.title_raw.evidence[:1]))


@subagent("classify.quant_profile", AGENT, "1.1")
def quant_profile(text: str, skill_entries: list, llm_item: dict | None = None) -> AgentResult:
    """Place the candidate on the quantitative / fundamental / credit spectrum."""
    quant_skills = {"python", "cpp", "csharp", "r_lang", "kdb", "matlab", "machine_learning",
                    "time_series", "statistics", "backtesting", "sql", "java"}
    fund_skills = {"financial_modelling", "equity_research", "due_diligence"}
    have = {s.canonical for s in skill_entries}
    q = len(have & quant_skills)
    f = len(have & fund_skills)
    low = tx.norm(text)
    credit = sum(low.count(k) for k in ("high yield", "investment grade", "credit", "bond", "securitization"))

    if credit >= 4 and f >= 1:
        label, why = "credit", f"{credit} credit-specific mentions alongside fundamental toolkit"
    elif q >= 3 and f >= 2:
        label, why = "hybrid", f"{q} quantitative and {f} fundamental skill families evidenced"
    elif q >= 3:
        label, why = "quantitative", f"{q} quantitative skill families evidenced"
    elif f >= 1:
        label, why = "fundamental", f"{f} fundamental skill families, {q} quantitative"
    else:
        return AgentResult(name="", status="partial", output=None,
                           warnings=["insufficient signal to place on the quant/fundamental axis"])

    conf = 0.7
    if llm_item and llm_item.get("label") == label:
        conf = 0.93
        why += "; LLM classification agrees"
    elif llm_item and llm_item.get("label"):
        conf = 0.5
        why += f"; LLM said '{llm_item['label']}' -- disagreement, routed to review"
    return AgentResult(name="", confidence=conf,
                       output=Classification(label=label, confidence=conf, rationale=why))


@subagent("classify.feeder_path", AGENT, "1.1")
def feeder_path(text: str, employment: list, llm_item: dict | None = None) -> AgentResult:
    """Identify the pipeline this candidate entered finance through.

    Recruiters at a pod shop think in feeder paths -- 'two years IBD then a pod seat'
    is a recognisable shape with known strengths. Scoring weights the EARLIEST
    substantive roles, because the feeder path is about origin, not current seat.
    """
    low = tx.norm(text)
    scores: Counter = Counter()
    hits: dict[str, list[str]] = {}
    for key, spec in tx.FEEDER_PATHS.items():
        found = [s for s in spec["signals"] if s in low]
        if found:
            scores[key] += len(found)
            hits[key] = found[:4]
    real = [e for e in employment if not e.is_internship]
    for e in reversed(real[-3:]):                    # earliest roles carry the most weight
        tier = e.employer_tier or "unknown"
        for key, spec in tx.FEEDER_PATHS.items():
            if tier in spec.get("tiers", []):
                scores[key] += 2.5
                hits.setdefault(key, []).append(f"early employer tier: {tx.TIER_DISPLAY.get(tier, tier)}")
    if not scores:
        return AgentResult(name="", status="partial", output=None,
                           warnings=["no recognisable feeder path"])
    label, sc = scores.most_common(1)[0]
    total = sum(scores.values()) or 1
    conf = round(min(0.92, 0.35 + 0.65 * sc / total), 3)
    why = f"signals: {', '.join(hits.get(label, [])[:4])}"
    if llm_item and llm_item.get("label") == label:
        conf = round(min(0.95, conf + 0.12), 3)
        why += "; LLM agrees"
    return AgentResult(name="", confidence=conf,
                       output=Classification(label=label, confidence=conf, rationale=why,
                                             triggers=hits.get(label, [])[:6],
                                             low_support=sc < 2))
