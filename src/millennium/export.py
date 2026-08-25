"""Deliverable #2: parsed resume data as JSON and CSV.

Three shapes, because they answer different questions:

* `candidates.json`  -- full fidelity, every field with its evidence and status. This
  is the contract other systems integrate against.
* `candidates.csv`   -- one row per candidate, flattened for Excel. Unknown values are
  written as an empty cell and a companion `*_status` column says WHY: 'abstained'
  (we saw a claim we could not prove) reads very differently from 'missing' (the CV
  never said). Collapsing those two into a blank is the standard way this data gets
  quietly misread.
* `employment.csv` / `education.csv` / `skills.csv` -- long-form for pivoting.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from . import taxonomy as tx
from .config import SETTINGS
from .schema import CandidateProfile, Tracked


def _status(t: Tracked) -> str:
    if t.is_known:
        return t.validation_status
    return "abstained" if t.validation_status == "abstained" else "missing"


def _v(t: Tracked):
    return t.normalized_value if t.is_known and t.normalized_value is not None else (
        t.value if t.is_known else "")


def flat_row(p: CandidateProfile, include_pii: bool = True) -> dict:
    cur = p.current_role()
    highest = None
    order = {"phd": 5, "mba": 4, "professional": 4, "masters": 3, "bachelors": 2, "secondary": 1}
    for e in p.education:
        if e.degree_level and (highest is None or order.get(e.degree_level, 0) > order.get(highest, 0)):
            highest = e.degree_level

    row = {
        "candidate_id": p.candidate_id,
        "source_file": p.provenance.source_file if p.provenance else "",
        "is_synthetic": bool(p.provenance and p.provenance.is_synthetic),
        "headline": _v(p.headline), "headline_status": _status(p.headline),
        "location": _v(p.location_current), "location_status": _status(p.location_current),
        "country": p.geography.label if p.geography else "",
        "region": tx.display("region", p.geo_region.label, "") if p.geo_region else "",
        "region_confidence": round(p.geo_region.confidence, 3) if p.geo_region else "",
        "years_experience": _v(p.years_experience),
        "years_experience_status": _status(p.years_experience),
        "years_relevant": _v(p.years_relevant_experience),
        "seniority_level": p.seniority.label if p.seniority else "",
        "seniority_title": (tx.display("seniority", p.seniority.label, "")
                            if p.seniority and p.seniority.label.startswith("L") else ""),
        "current_employer": (cur.employer_canonical or cur.employer_raw.display(""))
                            if cur else "",
        "current_employer_tier": tx.display("tier", cur.employer_tier, "") if cur else "",
        "current_title": cur.title_raw.display("") if cur else "",
        "current_tenure_months": _v(p.current_tenure_months),
        "approach": p.quant_fundamental.label if p.quant_fundamental else "",
        "feeder_path": (tx.display("feeder", p.feeder_path.label)
                        if p.feeder_path else ""),
        "strategies": "; ".join(tx.display("strategy", c.label) for c in p.strategies),
        "strategies_confidence": "; ".join(f"{c.confidence:.2f}" for c in p.strategies),
        "sectors": "; ".join(tx.display("sector", c.label) for c in p.sectors),
        "skills_core": "; ".join(s.canonical for s in p.skills if s.depth == "core"),
        "skills_all": "; ".join(s.canonical for s in p.skills),
        "certifications": "; ".join(
            f"{tx.display("certification", c.canonical)}"
            + (f" ({c.status})" if c.status else "") for c in p.certifications if c.canonical),
        "languages": "; ".join(
            f"{l.language}" + (f" ({l.proficiency})" if l.proficiency else "")
            for l in p.languages),
        "highest_degree": highest or "",
        "institutions": "; ".join(e.institution.display("") for e in p.education
                                  if e.institution.is_known),
        "n_roles": len(p.employment),
        "n_employment_gaps": len(p.employment_gaps),
        "completeness": p.quality.completeness,
        "evidence_coverage": p.quality.evidence_coverage,
        "extraction_quality": p.quality.extraction_quality,
        "abstentions": p.quality.abstention_count,
        "needs_human_review": p.quality.needs_human_review,
        "review_reasons": " | ".join(p.quality.review_reasons),
        "validation_flags": " | ".join(p.quality.validation_flags),
        "injection_flags": "; ".join(p.provenance.injection_flags) if p.provenance else "",
        "near_duplicate_of": "; ".join(p.provenance.near_duplicate_of) if p.provenance else "",
        "llm_model": p.provenance.llm_model if p.provenance else "",
        "cost_usd": p.provenance.cost_usd if p.provenance else 0.0,
        "schema_version": p.provenance.schema_version if p.provenance else "",
    }
    if include_pii:
        row |= {
            "full_name": _v(p.sensitive.full_name),
            "full_name_status": _status(p.sensitive.full_name),
            "email": _v(p.sensitive.email), "email_status": _status(p.sensitive.email),
            "phone": _v(p.sensitive.phone), "phone_status": _status(p.sensitive.phone),
        }
    return row


def _write_csv(path: Path, rows: list[dict]) -> Path:
    if not rows:
        path.write_text("")
        return path
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf8") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    return path


def export_all(profiles: list[CandidateProfile], out_dir: Path | None = None,
               manifest: dict | None = None, include_pii: bool = True) -> dict[str, Path]:
    out = Path(out_dir or SETTINGS.paths.exports)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema_version": SETTINGS.schema_version,
        "taxonomy_version": tx.TAXONOMY_VERSION,
        "count": len(profiles),
        "manifest": manifest or {},
        "candidates": [json.loads(p.model_dump_json(exclude={"raw_text"})) for p in profiles],
    }
    pj = out / "candidates.json"
    pj.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf8")
    written["candidates.json"] = pj

    written["candidates.csv"] = _write_csv(
        out / "candidates.csv", [flat_row(p, include_pii) for p in profiles])

    emp_rows = []
    for p in profiles:
        for i, e in enumerate(p.employment):
            emp_rows.append({
                "candidate_id": p.candidate_id, "seq": i,
                "employer_raw": e.employer_raw.display(""),
                "employer_canonical": e.employer_canonical or "",
                "employer_tier": tx.display("tier", e.employer_tier, ""),
                "title": e.title_raw.display(""),
                "seniority_level": e.seniority_level or "",
                "location": e.location.display(""),
                "start": e.dates.start.normalized_value or "",
                "end": e.dates.end.normalized_value or "",
                "start_status": _status(e.dates.start), "end_status": _status(e.dates.end),
                "duration_months": _v(e.dates.duration_months),
                "is_current": e.dates.is_current, "is_internship": e.is_internship,
                "n_highlights": len(e.highlights),
                "evidence_page": (e.employer_raw.evidence[0].page
                                  if e.employer_raw.evidence else ""),
                "evidence_char_start": (e.employer_raw.evidence[0].char_start
                                        if e.employer_raw.evidence else ""),
            })
    written["employment.csv"] = _write_csv(out / "employment.csv", emp_rows)

    edu_rows = []
    for p in profiles:
        for e in p.education:
            edu_rows.append({
                "candidate_id": p.candidate_id,
                "institution": e.institution.display(""), "degree": e.degree_raw.display(""),
                "degree_level": e.degree_level or "", "field": e.field_of_study.display(""),
                "graduation_year": _v(e.graduation_year), "gpa": e.gpa_raw.display(""),
                "location": e.location.display(""), "honors": "; ".join(e.honors),
                "institution_status": _status(e.institution),
            })
    written["education.csv"] = _write_csv(out / "education.csv", edu_rows)

    skill_rows = [{"candidate_id": p.candidate_id, "skill": s.canonical,
                   "category": s.category, "depth": s.depth,
                   "surface_forms": "; ".join(s.surface_forms),
                   "n_evidence": len(s.evidence)}
                  for p in profiles for s in p.skills]
    written["skills.csv"] = _write_csv(out / "skills.csv", skill_rows)

    _CTX_PAD = 80

    def _context(p: CandidateProfile, ev) -> str:
        """A readable excerpt around the span, for a reviewer scanning the CSV without
        the app open. Built here from `raw_text`, not stored on `Evidence` itself --
        `Evidence.snippet` is defined to be the exact matched text (see classification.
        _ev and validate.verify_span), and conflating the two previously let a genuine
        span-integrity bug (padded text mislabelled as an exact match) go unnoticed."""
        if not p.raw_text:
            return ev.snippet[:200]
        lo = max(0, ev.char_start - _CTX_PAD)
        hi = min(len(p.raw_text), ev.char_end + _CTX_PAD)
        return p.raw_text[lo:hi].replace("\n", " ")[:260]

    ev_rows = [{"candidate_id": p.candidate_id, "doc_id": ev.doc_id, "page": ev.page or "",
                "char_start": ev.char_start, "char_end": ev.char_end,
                "match_kind": ev.match_kind, "match_score": ev.match_score,
                "snippet": ev.snippet[:200].replace("\n", " "),
                "context": _context(p, ev)}
               for p in profiles for ev in p.all_evidence()]
    written["evidence.csv"] = _write_csv(out / "evidence.csv", ev_rows)
    return written
