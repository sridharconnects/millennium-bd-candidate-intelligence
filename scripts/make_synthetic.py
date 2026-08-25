#!/usr/bin/env python3
"""Generate a labelled synthetic corpus for the scalability benchmark.

Why procedural rather than LLM-generated:

* **Determinism.** Seeded, so the benchmark curve is reproducible by anyone who checks
  out the repo. An LLM-generated corpus would differ every run and cost money to
  regenerate, which makes the published latency numbers unverifiable.
* **Grid control.** The point of this corpus is to cover the full
  geography x strategy x sector x seniority space evenly. Sampling that grid directly
  guarantees coverage; prompting for it only hopes for it.
* **Honesty.** These records exist to measure index build time and search latency at
  scale. They are never used to measure extraction accuracy -- that is what the ten
  real, hand-labelled resumes are for -- so LLM-authored prose would add cost without
  adding validity.

Every record is stamped `is_synthetic=True`, is labelled SYNTHETIC in the UI, and is
excluded from every accuracy metric.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from millennium import taxonomy as tx                     # noqa: E402
from millennium.config import SETTINGS                    # noqa: E402
from millennium.schema import (Certification, Classification, DateRange,  # noqa: E402
                               EducationEntry, EmploymentEntry, Evidence,
                               LanguageEntry, ProvenanceRecord, QualityReport,
                               CandidateProfile, SensitiveAttributes, SkillEntry,
                               Tracked, stable_id)

TITLES = {1: ["Summer Analyst", "Intern", "Trainee"],
          2: ["Junior Analyst", "Research Associate"],
          3: ["Analyst", "Research Analyst", "Desk Analyst"],
          4: ["Senior Analyst", "Investment Analyst", "Associate"],
          5: ["Lead Analyst", "Vice President"],
          6: ["Portfolio Manager", "Director"],
          7: ["Head of Research", "Chief Investment Officer"]}

CITY_BY_REGION = {
    "americas": [("New York", "United States"), ("Boston", "United States"),
                 ("Chicago", "United States"), ("Greenwich", "United States"),
                 ("Sao Paulo", "Brazil"), ("Toronto", "Canada")],
    "emea": [("London", "United Kingdom"), ("Paris", "France"), ("Frankfurt", "Germany"),
             ("Zurich", "Switzerland"), ("Dubai", "UAE")],
    "apac": [("Hong Kong", "Hong Kong"), ("Singapore", "Singapore"), ("Mumbai", "India"),
             ("Tokyo", "Japan"), ("Sydney", "Australia")],
}

UNIS = ["State University", "Metropolitan Institute", "Northern Business School",
        "Riverside College", "Capital University", "Harborview Institute of Technology"]
FIRST = ["Alex", "Jordan", "Sam", "Riley", "Casey", "Morgan", "Taylor", "Avery",
         "Quinn", "Rowan", "Skyler", "Emerson", "Reese", "Finley", "Harper"]
LAST = ["Okafor", "Nakamura", "Silva", "Haddad", "Kowalski", "Lindqvist", "Moreau",
        "Bianchi", "Novak", "Ferreira", "Andersen", "Dubois", "Meier", "Karlsson"]

BULLETS = {
    "equity_long_short": ["Managed a fundamental long/short book with disciplined position sizing and sector hedges",
                          "Generated single-name alpha through primary research and expert network calls"],
    "quantitative_research": ["Researched and backtested systematic alpha signals across a multi-factor framework",
                              "Built and maintained a factor model with performance attribution"],
    "credit_long_short": ["Covered high yield and investment grade issuers with three-statement credit models",
                          "Priced corporate bonds and pitched relative value ideas to the desk"],
    "derivatives_pricing": ["Developed pricing library components in C++ for exotic payoffs",
                            "Implemented Greeks computation and scenario analysis for trading desks"],
    "global_macro": ["Analysed rates, FX and sovereign risk across developed and emerging markets",
                     "Constructed discretionary macro trade expressions with defined risk"],
    "event_driven": ["Analysed catalyst-driven situations including spin-offs and restructurings",
                     "Modelled deal spreads and downside scenarios for announced transactions"],
}
DEFAULT_BULLETS = ["Produced initiation reports and quarterly earnings analysis for institutional clients",
                   "Built financial models including DCF and comparable company analysis"]

SKILL_POOL = {
    "quantitative": ["python", "cpp", "sql", "machine_learning", "time_series", "statistics",
                     "backtesting", "kdb", "matlab", "r_lang"],
    "fundamental": ["financial_modelling", "equity_research", "due_diligence", "bloomberg",
                    "factset", "excel", "capital_iq"],
    "credit": ["financial_modelling", "bloomberg", "statistics", "excel", "refinitiv"],
    "hybrid": ["python", "financial_modelling", "sql", "statistics", "bloomberg",
               "equity_research", "backtesting"],
}


def _tracked(value, doc_id: str, pos: int, conf: float = 0.9) -> Tracked:
    """Synthetic records still carry evidence objects so that every downstream code
    path -- the evidence viewer, the integrity test, the coverage metric -- exercises
    the same shape it would on a real document."""
    snippet = str(value)
    return Tracked(value=value, normalized_value=value, confidence=conf,
                   extraction_method="llm", validation_status="verified",
                   evidence=[Evidence(doc_id=doc_id, char_start=pos,
                                      char_end=pos + len(snippet), snippet=snippet)])


def make_one(rng: random.Random, i: int) -> CandidateProfile:
    region = rng.choice(list(CITY_BY_REGION))
    city, country = rng.choice(CITY_BY_REGION[region])
    approach = rng.choices(["fundamental", "quantitative", "hybrid", "credit"],
                           weights=[0.42, 0.26, 0.20, 0.12])[0]
    n_strat = rng.randint(1, 3)
    strategies = rng.sample(list(tx.STRATEGIES), n_strat)
    sectors = rng.sample(list(tx.SECTORS), rng.randint(1, 3))
    level = rng.choices([2, 3, 4, 5, 6], weights=[0.15, 0.30, 0.30, 0.18, 0.07])[0]
    n_roles = rng.randint(1, 4)
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    doc_id = f"synth{i:05d}"
    cid = stable_id(doc_id, "synthetic", SETTINGS.schema_version)

    text_parts = [name, f"{city}, {country}"]
    employment: list[EmploymentEntry] = []
    end_year = date.today().year
    pos = 0
    for r in range(n_roles):
        dur = rng.randint(12, 44)
        start_year = end_year - dur // 12
        tier = rng.choice(list(tx.FIRM_TIERS))
        employer = rng.choice(tx.FIRM_TIERS[tier])
        canon, real_tier = tx.canonical_employer(employer)
        lvl = max(1, level - r)
        title = rng.choice(TITLES[lvl])
        bullets = BULLETS.get(strategies[0], DEFAULT_BULLETS)
        block = f"{title} — {canon}, {city}  {start_year}-{end_year}\n" + "\n".join(bullets)
        text_parts.append(block)
        employment.append(EmploymentEntry(
            employer_raw=_tracked(canon, doc_id, pos), employer_canonical=canon,
            employer_tier=real_tier,
            title_raw=_tracked(title, doc_id, pos + 30), title_normalized=title,
            seniority_level=lvl, location=_tracked(f"{city}, {country}", doc_id, pos + 60),
            dates=DateRange(
                start=_tracked(f"{start_year}-{rng.randint(1,12):02d}", doc_id, pos + 90),
                end=(Tracked(value="present", normalized_value="present", confidence=0.9,
                             validation_status="verified",
                             evidence=[Evidence(doc_id=doc_id, char_start=pos + 100,
                                                char_end=pos + 107, snippet="present")])
                     if r == 0 else _tracked(f"{end_year}-{rng.randint(1,12):02d}", doc_id, pos + 100)),
                is_current=(r == 0)),
            is_internship=(lvl == 1),
            highlights=[_tracked(b, doc_id, pos + 140 + 90 * j, 0.85)
                        for j, b in enumerate(bullets)]))
        # Normalise the stored start/end to ISO so downstream date logic works.
        employment[-1].dates.start.normalized_value = f"{start_year}-01"
        employment[-1].dates.end.normalized_value = "present" if r == 0 else f"{end_year}-01"
        end_year = start_year
        pos += len(block) + 2

    grad = end_year - rng.randint(0, 2)
    degree = rng.choice(["Bachelor of Science in Economics", "Master of Science in Finance",
                         "MBA", "Master of Financial Engineering", "PhD in Statistics"])
    education = [EducationEntry(
        institution=_tracked(rng.choice(UNIS), doc_id, pos),
        degree_raw=_tracked(degree, doc_id, pos + 30),
        degree_level=tx.degree_level(degree),
        field_of_study=_tracked(rng.choice(["Finance", "Economics", "Statistics",
                                            "Engineering"]), doc_id, pos + 60),
        graduation_year=_tracked(grad, doc_id, pos + 90))]

    skills = [SkillEntry(canonical=s, category=tx.SKILLS[s]["category"],
                         depth=rng.choices(["core", "applied", "mentioned"],
                                           weights=[0.25, 0.45, 0.30])[0],
                         surface_forms=[s],
                         evidence=[Evidence(doc_id=doc_id, char_start=pos + 200,
                                            char_end=pos + 200 + len(s), snippet=s)])
              for s in rng.sample(SKILL_POOL[approach],
                                  min(len(SKILL_POOL[approach]), rng.randint(3, 7)))]

    certs = []
    if rng.random() < 0.32:
        certs.append(Certification(name=_tracked("CFA", doc_id, pos + 300),
                                   canonical="cfa",
                                   status=rng.choice(["charterholder", "level_ii", "level_i"])))

    langs = [LanguageEntry(language="English", proficiency="fluent")]
    if rng.random() < 0.4:
        langs.append(LanguageEntry(language=rng.choice(["Mandarin", "French", "Spanish",
                                                        "German", "Hindi", "Portuguese"]),
                                   proficiency=rng.choice(["native", "conversational"])))

    raw = "\n\n".join(text_parts)
    years = round(sum((e.dates.duration_months.value or 24) for e in employment) / 12, 1) \
        if any(e.dates.duration_months.is_known for e in employment) else \
        round(rng.uniform(1.5, 18.0), 1)

    p = CandidateProfile(
        candidate_id=cid, doc_id=doc_id, raw_text=raw,
        sensitive=SensitiveAttributes(
            full_name=_tracked(name, doc_id, 0),
            email=_tracked(f"{name.split()[0].lower()}.{name.split()[1].lower()}@example.invalid",
                           doc_id, 20)),
        headline=_tracked(f"{employment[0].title_raw.value} at {employment[0].employer_canonical}",
                          doc_id, 40),
        location_current=_tracked(f"{city}, {country}", doc_id, 12),
        employment=employment, education=education, skills=skills,
        certifications=certs, languages=langs,
        years_experience=Tracked.derived(years, 0.9, "synthetic generator"),
        geography=Classification(label=country, confidence=0.95, rationale="synthetic"),
        geo_region=Classification(label=region, confidence=0.95, rationale="synthetic"),
        seniority=Classification(label=f"L{level}", confidence=0.9, rationale="synthetic"),
        quant_fundamental=Classification(label=approach, confidence=0.9, rationale="synthetic"),
        feeder_path=Classification(label=rng.choice(list(tx.FEEDER_PATHS)), confidence=0.8,
                                   rationale="synthetic"),
        strategies=[Classification(label=s, confidence=round(rng.uniform(0.6, 0.95), 2),
                                   rationale="synthetic") for s in strategies],
        sectors=[Classification(label=s, confidence=round(rng.uniform(0.6, 0.95), 2),
                                rationale="synthetic") for s in sectors],
        quality=QualityReport(extraction_quality=round(rng.uniform(0.75, 0.98), 3),
                              completeness=round(rng.uniform(0.6, 1.0), 3),
                              evidence_coverage=round(rng.uniform(0.8, 1.0), 3)),
        provenance=ProvenanceRecord(
            source_file=f"SYNTHETIC_{i:05d}.generated", file_sha256=doc_id * 4,
            text_sha256=doc_id * 4, file_type="synthetic",
            extractor="scripts/make_synthetic.py", schema_version=SETTINGS.schema_version,
            taxonomy_version=tx.TAXONOMY_VERSION, is_synthetic=True))
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260822)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    profiles = [make_one(rng, i) for i in range(args.n)]
    out = SETTINGS.paths.synthetic / "synthetic_candidates.json"
    out.write_text(json.dumps({
        "WARNING": "SYNTHETIC DATA — generated for scalability benchmarking only. "
                   "Never used for accuracy evaluation. Not real candidates.",
        "generated_by": "scripts/make_synthetic.py", "seed": args.seed,
        "count": len(profiles),
        "candidates": [json.loads(p.model_dump_json()) for p in profiles]},
        indent=1), encoding="utf8")

    from collections import Counter
    print(f"Wrote {len(profiles)} synthetic profiles -> {out.relative_to(ROOT)} "
          f"({out.stat().st_size/1e6:.1f} MB)")
    for dim, fn in (("region", lambda p: p.geo_region.label),
                    ("approach", lambda p: p.quant_fundamental.label),
                    ("seniority", lambda p: p.seniority.label)):
        print(f"  {dim:10s} {dict(Counter(fn(p) for p in profiles).most_common())}")
    print(f"  strategies {len({c.label for p in profiles for c in p.strategies})}/"
          f"{len(tx.STRATEGIES)} covered; sectors "
          f"{len({c.label for p in profiles for c in p.sectors})}/{len(tx.SECTORS)} covered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
