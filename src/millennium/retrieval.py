"""Agent 5 -- Search. Query understanding, hybrid retrieval, evidence-bearing results.

Fusion uses Reciprocal Rank Fusion rather than a weighted score blend. RRF combines
*ranks*, so it needs no calibration between BM25 (unbounded, corpus-dependent) and
cosine (bounded, [-1,1]) -- two scales that cannot be meaningfully added. The ablation
table in the app reports dense-only, lexical-only and fused side by side rather than
asserting the hybrid is better.
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

from . import taxonomy as tx
from .agents.base import AgentResult, subagent
from .config import SETTINGS
from .index import SearchIndex
from .llm import LLMClient, LLMUnavailable
from .prompts import query_prompt

AGENT = "search"


@dataclass
class ParsedQuery:
    semantic_text: str = ""
    must_have: dict = field(default_factory=dict)
    preferences: dict = field(default_factory=dict)
    exclusions: dict = field(default_factory=dict)
    interpretation: str = ""
    method: str = "rule"

    def is_empty(self) -> bool:
        return not (self.semantic_text.strip() or any(self.must_have.values())
                    or any(self.preferences.values()))


@dataclass
class Hit:
    candidate_id: str
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    matched_chunks: list[dict] = field(default_factory=list)
    explain: str = ""


# --------------------------------------------------------------- query parsing
# The rule parser is not a stub -- it is the guaranteed path. The LLM parser is an
# enhancement that runs when a key is available, and its output is validated against
# the same closed vocabulary, so it can only ever produce labels the rule parser could.
_NEG = re.compile(r"\b(no|not|without|exclude|excluding|avoid|non)\b[\s\-]*([a-z+#/&' ]{3,40})", re.I)
_YEARS = re.compile(r"(\d+)\s*\+?\s*(?:-\s*(\d+)\s*)?(?:years?|yrs?|y)\b", re.I)
_MIN_YEARS = re.compile(r"(?:at least|minimum|min\.?|over|more than|\bgt\b)\s*(\d+)", re.I)
_MAX_YEARS = re.compile(r"(?:at most|maximum|max\.?|under|less than|up to|below)\s*(\d+)", re.I)
_MUST = re.compile(r"\b(must|required|require|mandatory|essential|minimum|at least|"
                   r"only consider|strictly)\b", re.I)
_PREFER = re.compile(r"\b(prefer|preferred|preferable|nice to have|a plus|bonus|"
                     r"ideally|desirable|would be good|open to)\b", re.I)
# Clause boundaries. A requisition states its hard and soft requirements in separate
# sentences or bullets, so that is the unit at which must/prefer must be decided.
_CLAUSE = re.compile(r"[.;\n\r]+|\s+[-•*]\s+")

_REGION_WORDS = {
    "americas": ["us", "usa", "u.s.", "united states", "america", "americas", "new york",
                 "nyc", "boston", "chicago", "greenwich", "brazil", "canada", "latam"],
    "emea": ["europe", "european", "emea", "uk", "london", "france", "paris", "germany",
             "frankfurt", "switzerland", "zurich", "dubai", "middle east"],
    "apac": ["apac", "asia", "asia-pacific", "asia pacific", "hong kong", "singapore",
             "india", "mumbai", "china", "japan", "tokyo", "australia"],
}


def _scan_vocab(text: str) -> dict:
    """Map free text onto the closed taxonomies. Shared by both query parsers."""
    low = tx.norm(text)
    found = {"strategies": [], "sectors": [], "skills": [], "geo_regions": [],
             "certifications": [], "degree_levels": [], "employer_tiers": [],
             "feeder_paths": [], "languages": []}
    for label, spec in tx.STRATEGIES.items():
        if any(t in low for t in spec["triggers"]):
            found["strategies"].append(label)
    for label, spec in tx.SECTORS.items():
        if any(t in low for t in spec["triggers"]):
            found["sectors"].append(label)
    for canon, spec in tx.SKILLS.items():
        if any(len(a) > 2 and a in low for a in spec["aliases"]):
            found["skills"].append(canon)
    for region, words in _REGION_WORDS.items():
        if any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", low) for w in words):
            found["geo_regions"].append(region)
    for canon, spec in tx.CERTIFICATIONS.items():
        if any(a in low for a in spec["aliases"]):
            found["certifications"].append(canon)
    for lvl_pat, lvl in tx.DEGREE_LEVELS:
        if re.search(lvl_pat, low):
            found["degree_levels"].append(lvl)
    for tier in tx.FIRM_TIERS:
        if tier.replace("_", " ") in low:
            found["employer_tiers"].append(tier)
    for fp, spec in tx.FEEDER_PATHS.items():
        # Match the display name, the key, or any of the path's own signal phrases --
        # a recruiter writes "no banking background", not "no ibd_analyst_program".
        if (fp.replace("_", " ") in low or tx.norm(spec["display"]) in low
                or any(len(sig) > 5 and sig in low for sig in spec["signals"])):
            found["feeder_paths"].append(fp)
    if re.search(r"(?<!\w)banking(?!\w)", low):
        found["feeder_paths"].append("ibd_analyst_program")
    for lang in tx.LANGUAGE_NAMES:
        if re.search(rf"(?<!\w){lang}(?!\w)", low):
            found["languages"].append(lang.title())
    return {k: sorted(set(v)) for k, v in found.items()}


@subagent("search.parse_query_rules", AGENT, "1.4")
def parse_query_rules(query: str) -> AgentResult:
    """Deterministic query understanding. Always available, never the untested fallback.

    Must-have vs preference is decided **per clause**, not per query. An earlier version
    treated a single "must have" anywhere as making every matched term mandatory, which
    on a real requisition — *"Must have 3-7 years ... CFA preferred"* — promoted the
    explicitly-preferred CFA to a hard gate and excluded the entire pool. Since
    over-marking must-haves silently empties a candidate list, ambiguous clauses default
    to preferences.
    """
    q = query or ""

    # Negations first, per clause, and their spans removed before positive scanning so
    # "no banking background" cannot also register banking as something desirable.
    exclusions: dict[str, list] = {}
    stripped = q
    for m in _NEG.finditer(q):
        for k, v in _scan_vocab(m.group(2)).items():
            exclusions.setdefault(k, []).extend(v)
        stripped = stripped.replace(m.group(0), " ")
    exclusions = {k: sorted(set(v)) for k, v in exclusions.items() if v}

    must: dict[str, list] = {}
    prefs: dict[str, list] = {}
    hard_clauses = soft_clauses = 0

    for clause in _CLAUSE.split(stripped):
        clause = clause.strip()
        if len(clause) < 3:
            continue
        found = _scan_vocab(clause)
        if not any(found.values()):
            continue
        is_must = bool(_MUST.search(clause))
        is_pref = bool(_PREFER.search(clause))
        # An explicit "preferred" wins over a "must" in the same clause: a sentence
        # reading "must be strong; a CFA is preferred" is stating one of each.
        target, counted = (prefs, "soft") if (is_pref or not is_must) else (must, "hard")
        hard_clauses += counted == "hard"
        soft_clauses += counted == "soft"
        for k, v in found.items():
            target.setdefault(k, []).extend(v)

    for block in (must, prefs):
        for k in list(block):
            block[k] = sorted({x for x in block[k] if x not in exclusions.get(k, [])})
    # A term stated as mandatory somewhere does not need to also be a preference.
    for k, v in must.items():
        if k in prefs:
            prefs[k] = [x for x in prefs[k] if x not in v]

    # Experience bounds: hard only when their own clause says so.
    min_y = max_y = None
    min_hard = max_hard = False
    for clause in _CLAUSE.split(stripped):
        hard = bool(_MUST.search(clause)) and not _PREFER.search(clause)
        if (m := _MIN_YEARS.search(clause)):
            min_y, min_hard = float(m.group(1)), hard
        if (m := _MAX_YEARS.search(clause)):
            max_y, max_hard = float(m.group(1)), hard
        if min_y is None and max_y is None and (m := _YEARS.search(clause)):
            min_y, min_hard = float(m.group(1)), hard
            if m.group(2):
                max_y, max_hard = float(m.group(2)), hard

    must["min_years"] = min_y if min_hard else None
    must["max_years"] = max_y if max_hard else None
    must.setdefault("countries", [])
    prefs.setdefault("countries", [])
    if min_y is not None and not min_hard:
        prefs["soft_min_years"] = min_y

    n_hard = sum(len(v) for v in must.values() if isinstance(v, list))
    n_soft = sum(len(v) for v in prefs.values() if isinstance(v, list))
    parts = [f"Rule parser: {n_hard} hard requirement(s), {n_soft} preference(s)"]
    if any(exclusions.values()):
        parts.append(f"{sum(len(v) for v in exclusions.values())} exclusion(s)")
    if min_y is not None or max_y is not None:
        band = f"{min_y if min_y is not None else 'any'}–{max_y if max_y is not None else 'any'}y"
        parts.append(f"experience {band}" + (" (hard)" if min_hard or max_hard else " (soft)"))
    parts.append("must/prefer decided per clause; ambiguous clauses default to "
                 "preferences, because over-gating silently empties a pool")

    return AgentResult(name="", output=ParsedQuery(
        semantic_text=q.strip(), must_have=must, preferences=prefs,
        exclusions=exclusions, interpretation="; ".join(parts), method="rule"),
        confidence=0.8)


@subagent("search.parse_query_llm", AGENT, "1.2")
def parse_query_llm(client: LLMClient, query: str) -> AgentResult:
    """LLM query understanding, with the rule parser as a guaranteed fallback.

    Output is intersected with the closed vocabulary, so a hallucinated label is
    dropped rather than silently becoming a filter nobody can satisfy.
    """
    if not SETTINGS.flags.enable_llm_query_parse:
        return AgentResult(name="", status="skipped", output=None)
    try:
        system, msgs, hint = query_prompt(query)
        r = client.complete_json(system, msgs, hint, stage="query")
        d = r.data if isinstance(r.data, dict) else {}
    except (LLMUnavailable, Exception) as e:  # noqa: BLE001
        return AgentResult(name="", status="failed", output=None,
                           warnings=[f"LLM query parsing unavailable ({type(e).__name__}); "
                                     f"using the deterministic rule parser"])

    vocab = {"strategies": set(tx.STRATEGIES), "sectors": set(tx.SECTORS),
             "skills": set(tx.SKILLS), "certifications": set(tx.CERTIFICATIONS),
             "employer_tiers": set(tx.FIRM_TIERS), "feeder_paths": set(tx.FEEDER_PATHS),
             "geo_regions": {"americas", "emea", "apac"},
             "degree_levels": {"phd", "mba", "masters", "bachelors", "professional", "secondary"}}

    def clean(block) -> dict:
        block = block if isinstance(block, dict) else {}
        out = {}
        dropped = []
        for k, v in block.items():
            if isinstance(v, list):
                allowed = vocab.get(k)
                kept = [x for x in v if isinstance(x, str) and (allowed is None or x in allowed)]
                dropped += [x for x in v if isinstance(x, str) and allowed and x not in allowed]
                out[k] = kept
            else:
                out[k] = v
        return out

    pq = ParsedQuery(
        semantic_text=str(d.get("semantic_text") or query),
        must_have=clean(d.get("must_have")), preferences=clean(d.get("preferences")),
        exclusions=clean(d.get("exclusions")),
        interpretation=str(d.get("interpretation") or ""), method="llm")
    return AgentResult(name="", output=pq, confidence=0.9,
                       tokens_in=r.tokens_in, tokens_out=r.tokens_out,
                       cost_usd=r.cost_usd, cached=r.cached)


def understand_query(query: str, client: LLMClient | None = None) -> tuple[ParsedQuery, AgentResult]:
    """LLM first when enabled and reachable, deterministic rules otherwise."""
    if client is not None and SETTINGS.flags.enable_llm_query_parse:
        r = parse_query_llm(client, query)
        if r.ok and r.output is not None:
            return r.output, r
    r = parse_query_rules(query)
    return r.output, r


# ------------------------------------------------------------------- retrieval
@subagent("search.rrf_fuse", AGENT, "1.1")
def rrf_fuse(rankings: dict[str, list[tuple[str, float]]], k: int = 60) -> AgentResult:
    """score(d) = sum_r 1/(k + rank_r(d)). Rank-based, so no score calibration needed."""
    fused: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for source, ranked in rankings.items():
        for i, (cid, _s) in enumerate(ranked, start=1):
            fused[cid] += 1.0 / (k + i)
            ranks[cid][source] = i
    order = sorted(fused.items(), key=lambda x: -x[1])
    return AgentResult(name="", output=[(cid, sc, ranks[cid]) for cid, sc in order])


@subagent("search.retrieve", AGENT, "1.3")
def retrieve(index: SearchIndex, query: str, mode: str = "hybrid",
             top_k: int | None = None) -> AgentResult:
    """Run one or both retrievers, fuse, then aggregate chunk hits to candidates.

    Aggregation takes each candidate's BEST chunk rather than summing, so a long CV
    with many mediocre chunks cannot outrank a short one with a perfect match. The
    chunks that fired are carried through as evidence for the UI.
    """
    top_k = top_k or SETTINGS.retrieval.top_k_final
    t0 = time.perf_counter()
    rankings: dict[str, list[tuple[str, float]]] = {}
    if mode in ("dense", "hybrid"):
        rankings["dense"] = index.dense_search(query, SETTINGS.retrieval.top_k_dense)
    if mode in ("lexical", "hybrid"):
        rankings["lexical"] = index.lexical_search(query, SETTINGS.retrieval.top_k_lexical)

    fused = rrf_fuse(rankings, SETTINGS.retrieval.rrf_k).output or []

    by_cand: dict[str, Hit] = {}
    for chunk_id, score, ranks in fused:
        ch = index.chunks.get(chunk_id)
        if ch is None:
            continue
        h = by_cand.get(ch.candidate_id)
        if h is None:
            h = Hit(candidate_id=ch.candidate_id, score=score,
                    dense_rank=ranks.get("dense"), lexical_rank=ranks.get("lexical"))
            by_cand[ch.candidate_id] = h
        else:
            h.score = max(h.score, score)          # best chunk wins, hits do not sum
        if len(h.matched_chunks) < 3:
            h.matched_chunks.append({
                "kind": ch.kind, "label": ch.label, "text": ch.text[:280],
                "char_start": ch.char_start, "char_end": ch.char_end,
                "dense_rank": ranks.get("dense"), "lexical_rank": ranks.get("lexical")})

    hits = sorted(by_cand.values(), key=lambda h: -h.score)[:top_k]
    for h in hits:
        parts = []
        if h.dense_rank:
            parts.append(f"semantic #{h.dense_rank}")
        if h.lexical_rank:
            parts.append(f"keyword #{h.lexical_rank}")
        h.explain = " + ".join(parts) or "no direct match"
    ms = (time.perf_counter() - t0) * 1000
    return AgentResult(name="", output=hits, latency_ms=int(ms),
                       confidence=1.0 if hits else 0.0)


# --------------------------------------------------------------------- filters
def _has_any(have: list[str], want: list[str]) -> bool:
    return bool(set(map(str.lower, have)) & set(map(str.lower, want)))


def apply_filters(profiles: list, pq: ParsedQuery) -> tuple[list, list[dict], dict]:
    """Hard gating. Returns (kept, excluded_with_reasons, caveats_by_candidate_id).

    Three rules, in order of how often they are got wrong:

    1. **A preference never eliminates.** It only scores.
    2. **Unknown never eliminates either.** If a requisition demands a CFA and a CV
       simply never mentions certifications, that candidate is *unverified*, not
       *unqualified*. Excluding them is how a strong candidate disappears because of a
       formatting quirk in their resume. They are kept, and the unmet-but-unknown
       requirement is returned as a caveat the UI shows and a reviewer can resolve in
       thirty seconds. Only a candidate with a KNOWN value that fails to match is gated.
    3. **Everything gated is visible**, with its reason. Silently shrinking a pool is
       how a search "finds nobody" without anyone learning why.
    """
    kept, excluded = [], []
    caveats: dict[str, list[str]] = {}
    mh = pq.must_have or {}
    ex = pq.exclusions or {}

    for p in profiles:
        sc = p.scorable()
        reasons: list[str] = []
        unknowns: list[str] = []

        checks = [
            ("strategies", [s for s in sc.strategies], "strategy"),
            ("sectors", [s for s in sc.sectors], "sector"),
            ("skills", [s.canonical for s in sc.skills], "skill"),
            ("certifications", sc.certifications, "certification"),
            ("degree_levels", sc.degree_levels, "degree"),
            ("employer_tiers", sc.employer_tiers, "employer tier"),
            ("languages", sc.languages, "language"),
            ("geo_regions", [sc.geo_region] if sc.geo_region else [], "region"),
            ("countries", [sc.geography] if sc.geography else [], "country"),
            ("feeder_paths", [sc.feeder_path] if sc.feeder_path else [], "feeder path"),
        ]
        for key, have, label in checks:
            known = [h for h in have if h]
            want = [w for w in (mh.get(key) or []) if w]
            if want and not _has_any(known, want):
                if known:
                    reasons.append(f"must-have {label}: {', '.join(want)} "
                                   f"(has {', '.join(sorted(known)[:4])})")
                else:
                    unknowns.append(f"{label} not stated in the CV, so "
                                    f"'{', '.join(want)}' could not be confirmed")
            drop = [w for w in (ex.get(key) or []) if w]
            if drop and _has_any(known, drop):
                reasons.append(f"excluded {label}: "
                               f"{', '.join(sorted(set(known) & set(drop)))}")

        y = sc.years_experience
        if (m := mh.get("min_years")) is not None:
            if y is None:
                unknowns.append(f"total experience could not be derived, so the "
                                f"{m:g}y minimum could not be confirmed")
            elif y < float(m):
                reasons.append(f"must-have min {m:g}y experience (has {y:.1f}y)")
        if (m := mh.get("max_years")) is not None and y is not None and y > float(m):
            reasons.append(f"must-have max {m:g}y experience (has {y:.1f}y)")
        if (m := mh.get("min_seniority")) is not None and sc.seniority_level is not None \
                and sc.seniority_level < int(m):
            reasons.append(f"must-have seniority >= L{m} (is L{sc.seniority_level})")

        if reasons:
            excluded.append({"candidate_id": p.candidate_id, "reasons": reasons,
                             "unverified": unknowns})
        else:
            kept.append(p)
            if unknowns:
                caveats[p.candidate_id] = unknowns
    return kept, excluded, caveats


@subagent("search.similar", AGENT, "1.0")
def similar_candidates(index: SearchIndex, profile, k: int = 5) -> AgentResult:
    """'More like this' over the profile-level text, excluding the candidate itself."""
    text = profile.searchable_text()[:1500]
    raw = index.store.search(index.embedder.encode_one(text), k * 6)
    seen: dict[str, float] = {}
    for chunk_id, score in raw:
        ch = index.chunks.get(chunk_id)
        if ch is None or ch.candidate_id == profile.candidate_id:
            continue
        seen[ch.candidate_id] = max(seen.get(ch.candidate_id, 0.0), score)
    out = sorted(seen.items(), key=lambda x: -x[1])[:k]
    return AgentResult(name="", output=out)
