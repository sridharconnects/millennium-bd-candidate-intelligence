"""LLM-backed advisory helpers for app pages.

These features are intentionally advisory. They can summarize, draft, and suggest
next checks, but they do not approve a review, delete a candidate, or make a hiring
decision. Every helper falls back to deterministic content so the app remains usable
in DEMO_MODE when no cached response exists for a new prompt.
"""
from __future__ import annotations

import json
from typing import Any

from millennium import taxonomy as tx


def _display(t, default: str = "") -> str:
    return str(t.display(default)) if hasattr(t, "display") else str(t or default)


def _role_text(role) -> str:
    if not role:
        return ""
    return " at ".join(x for x in [
        _display(role.title_raw),
        role.employer_canonical or _display(role.employer_raw),
    ] if x)


def candidate_context(p, *, blind: bool = False) -> dict[str, Any]:
    cur = p.current_role()
    return {
        "candidate": p.display_name(blind),
        "headline": _display(p.headline),
        "summary": _display(p.summary),
        "current_role": _role_text(cur),
        "years_experience": _display(p.years_experience, "unknown"),
        "region": tx.display("region", p.geo_region.label, "") if p.geo_region else "",
        "strategies": [tx.display("strategy", c.label, c.label) for c in p.strategies],
        "sectors": [tx.display("sector", c.label, c.label) for c in p.sectors],
        "skills_core": [s.canonical for s in p.skills if s.depth == "core"][:12],
        "skills_applied": [s.canonical for s in p.skills if s.depth == "applied"][:12],
        "quality": {
            "completeness": p.quality.completeness,
            "evidence_coverage": p.quality.evidence_coverage,
            "abstention_count": p.quality.abstention_count,
            "needs_human_review": p.quality.needs_human_review,
            "review_reasons": p.quality.review_reasons[:8],
            "validation_flags": p.quality.validation_flags[:8],
        },
        "employment": [
            {
                "role": _role_text(e),
                "dates": f"{e.dates.start.display('?')} to {e.dates.end.display('?')}",
                "highlights": [_display(h) for h in e.highlights[:3]],
            }
            for e in p.employment[:6]
        ],
        "education": [
            {
                "institution": _display(e.institution),
                "degree": _display(e.degree_raw),
                "field": _display(e.field_of_study),
                "year": _display(e.graduation_year),
            }
            for e in p.education[:4]
        ],
    }


def _ask_json(client, *, stage: str, system: str, payload: dict, schema: str) -> tuple[dict | None, str | None]:
    try:
        r = client.complete_json(
            system,
            [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            schema,
            stage=stage,
            max_tokens=1100,
        )
        data = r.data if isinstance(r.data, dict) else None
        if data:
            data["_llm_generated"] = True
            return data, None
    except Exception as e:  # noqa: BLE001 - UI falls back and explains the reason.
        return None, type(e).__name__
    return None, "empty_response"


def _fallback_candidate(ctx: dict, error: str | None) -> dict:
    strengths = []
    for label, values in (("Strategy", ctx["strategies"]), ("Sector", ctx["sectors"]),
                          ("Core skill", ctx["skills_core"])):
        strengths.extend(f"{label}: {v}" for v in values[:3])
    watchouts = list(ctx["quality"]["review_reasons"] or ctx["quality"]["validation_flags"])
    if ctx["quality"]["abstention_count"]:
        watchouts.append(f"{ctx['quality']['abstention_count']} field(s) need source confirmation")
    return {
        "_llm_generated": False,
        "_notice": f"LLM unavailable ({error}); showing deterministic brief.",
        "executive_summary": ctx["summary"] or ctx["headline"] or "Candidate profile is ready for recruiter review.",
        "strengths": strengths[:6] or ["Profile has structured, searchable candidate data."],
        "watchouts": watchouts[:5] or ["No validation flags recorded."],
        "interview_questions": [
            "Which role or sector exposure is most relevant to the current mandate?",
            "What parts of the resume should be verified before outreach?",
        ],
        "best_fit_mandates": ctx["strategies"][:3] or ["Open mandate fit to be determined by requisition scoring."],
    }


def candidate_brief(client, p, pool, *, blind: bool = False) -> dict:
    ctx = candidate_context(p, blind=blind)
    ctx["pool_context"] = {"pool_size": len(pool)}
    data, error = _ask_json(
        client,
        stage="candidate_brief",
        schema='{"executive_summary":"","strengths":[],"watchouts":[],"interview_questions":[],"best_fit_mandates":[]}',
        system=(
            "You are a hedge-fund BD talent intelligence analyst. Use only the JSON "
            "candidate context provided. Do not invent facts. Do not make a hiring "
            "decision. Return strict JSON with executive_summary, strengths, watchouts, "
            "interview_questions, and best_fit_mandates."
        ),
        payload=ctx,
    )
    return data or _fallback_candidate(ctx, error)


def review_plan(client, p, fields: dict, *, blind: bool = False) -> dict:
    ctx = candidate_context(p, blind=blind)
    ctx["review_fields"] = [
        {
            "label": label,
            "path": path,
            "current_value": _display(t),
            "status": getattr(t, "validation_status", ""),
            "notes": getattr(t, "notes", [])[:4],
        }
        for label, (path, t) in fields.items()
    ]
    data, error = _ask_json(
        client,
        stage="review_plan",
        schema='{"priority":"","what_is_wrong":[],"fields_to_check":[],"recommended_next_actions":[],"reviewer_note":""}',
        system=(
            "You are a resume data QA reviewer. Use only the supplied candidate and "
            "field metadata. Identify what is wrong or uncertain, what field should be "
            "checked first, and what evidence is needed. Do not invent corrected values. "
            "Return strict JSON."
        ),
        payload=ctx,
    )
    if data:
        return data
    flagged = [
        f"{label}: {t.validation_status}"
        for label, (_path, t) in fields.items()
        if getattr(t, "validation_status", "") in {"abstained", "conflicted", "unverified"}
    ]
    return {
        "_llm_generated": False,
        "_notice": f"LLM unavailable ({error}); showing deterministic review plan.",
        "priority": "High" if p.quality.needs_human_review else "Normal",
        "what_is_wrong": p.quality.review_reasons[:5] or p.quality.validation_flags[:5]
        or ["No specific issue recorded."],
        "fields_to_check": flagged[:6] or ["Check source evidence for the selected field."],
        "recommended_next_actions": [
            "Open the source evidence for the selected field.",
            "Correct only values supported by the document, otherwise approve the abstention.",
        ],
        "reviewer_note": "Do not fill blanks by inference; leave unproven facts abstained.",
    }


def analytics_memo(client, pool, dq: dict, dist: dict, gaps: dict) -> dict:
    payload = {
        "pool_size": len(pool),
        "data_quality": dq,
        "top_regions": list((dist.get("region") or {}).items())[:8],
        "top_strategies": list((dist.get("strategy") or {}).items())[:8],
        "top_sectors": list((dist.get("sector") or {}).items())[:8],
        "coverage_gaps": (gaps.get("gaps") or [])[:12],
        "strongest_cells": (gaps.get("strongest_cells") or [])[:8],
    }
    data, error = _ask_json(
        client,
        stage="analytics_memo",
        schema='{"headline":"","market_read":[],"coverage_risks":[],"recommended_searches":[],"data_quality_actions":[]}',
        system=(
            "You are a BD recruiting analytics lead. Read the aggregate pool data and "
            "write an executive memo. Focus on where the pool can fill mandates, where "
            "coverage is thin, and what searches to run next. Return strict JSON."
        ),
        payload=payload,
    )
    if data:
        return data
    return {
        "_llm_generated": False,
        "_notice": f"LLM unavailable ({error}); showing deterministic analytics memo.",
        "headline": f"{len(pool)} candidate(s) in the current pool.",
        "market_read": [
            f"{k}: {v}" for k, v in list((dist.get("strategy") or {}).items())[:5]
        ] or ["No strategy distribution available."],
        "coverage_risks": [
            f"{g.get('label')} ({g.get('dimension')}) has {g.get('count')} candidate(s)"
            for g in (gaps.get("gaps") or [])[:5]
        ] or ["No coverage gaps reported."],
        "recommended_searches": [
            "Run a requisition match for the thinnest coverage cell.",
            "Filter by evidence coverage before sharing candidates externally.",
        ],
        "data_quality_actions": [
            f"Review {dq.get('needs_review', 0)} flagged record(s).",
            f"Resolve {dq.get('total_abstentions', 0)} abstained field(s) where important.",
        ],
    }


def shortlist_memo(client, sl: dict, byid: dict, store) -> dict:
    rows = []
    for cid, entry in sl.items():
        p = byid.get(cid)
        if not p:
            continue
        comms = store.list_communications(cid)
        rows.append({
            "candidate": p.display_name(False),
            "headline": _display(p.headline),
            "years": _display(p.years_experience, "unknown"),
            "region": tx.display("region", p.geo_region.label, "") if p.geo_region else "",
            "strategies": [tx.display("strategy", c.label, c.label) for c in p.strategies],
            "sectors": [tx.display("sector", c.label, c.label) for c in p.sectors],
            "note": entry.get("note", ""),
            "tags": entry.get("tags", ""),
            "basis": entry.get("basis", ""),
            "touchpoints": [c["kind"] for c in comms[:5]],
            "needs_review": p.quality.needs_human_review,
        })
    data, error = _ask_json(
        client,
        stage="shortlist_memo",
        schema='{"slate_summary":"","top_reasons":[],"risks":[],"next_actions":[],"decision_pack_note":""}',
        system=(
            "You are a BD recruiter preparing a shortlist decision pack. Use only the "
            "provided slate data. Summarize strengths, risks, and next actions. Do not "
            "rank by protected attributes or make final hiring decisions. Return strict JSON."
        ),
        payload={"shortlist": rows},
    )
    if data:
        return data
    return {
        "_llm_generated": False,
        "_notice": f"LLM unavailable ({error}); showing deterministic slate memo.",
        "slate_summary": f"{len(rows)} candidate(s) are currently shortlisted.",
        "top_reasons": [r["basis"] or r["headline"] for r in rows[:5] if (r["basis"] or r["headline"])]
        or ["No shortlist rationale has been written yet."],
        "risks": [
            f"{r['candidate']} needs review before sharing"
            for r in rows if r["needs_review"]
        ] or ["No review blockers recorded on the slate."],
        "next_actions": [
            "Add recruiter notes for each shortlisted candidate.",
            "Draft or log outreach for candidates without touchpoints.",
        ],
        "decision_pack_note": "Use this memo as recruiter context, not as an automated decision.",
    }
