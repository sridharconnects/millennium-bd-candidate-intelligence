"""App-wide chat assistant: a real Claude tool-calling loop over the live workspace.

This is not a FAQ bot bolted onto the sidebar. Every tool below reads or writes the
exact same `st.session_state` keys a human clicking through the UI would touch --
`page`, `query`, `pending_filters`, `shortlist`, `selected` -- so an action the
assistant takes is indistinguishable, downstream, from the user having done it
themselves. It reuses the app's own established mechanisms (the saved-search
`pending_filters` hook, the "Open" button's set-then-`st.rerun()` navigation pattern)
rather than inventing a parallel path that could drift out of sync with them.

Two things are deliberately NOT delegated to the model:
  * Candidate resolution ("open Ryan's profile") is done by fuzzy name/id matching in
    Python (RapidFuzz, already a dependency), not by asking the model to know a
    candidate_id -- it never does, and guessing one would silently open the wrong
    person's record.
  * Nothing the assistant does is irreversible. It can navigate, search, filter, and
    manage the shortlist; it cannot delete a candidate, correct a field, or approve a
    review -- those stay behind an explicit human click on the Review page, matching
    the platform's standing claim that a human approves every shortlist and every
    correction.

Like the LangChain orchestrator, this needs a live API key -- a chat reply is
inherently unpredictable, so there is nothing to replay from the deterministic
DEMO_MODE cache. It fails cleanly and explains why when no key is configured.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz, process, utils as fuzz_utils

from . import taxonomy as tx
from .config import SETTINGS

MAX_TOOL_ROUNDS = 8   # hard cap: a misbehaving loop must not run away with API cost.
# Raised from 5 after a live test: "who is most senior in APAC" legitimately needs
# get_pool_summary + search + several per-candidate lookups, and 5 rounds cut it off
# mid-comparison. list_candidates (below) also cut the round count needed per query by
# letting the model compare several people in one call instead of one-by-one.


class AssistantUnavailable(RuntimeError):
    """No live key configured. Mirrors llm.LLMUnavailable's messaging pattern."""


SYSTEM = """You are the assistant embedded in the Millennium BD candidate intelligence \
platform, a hedge-fund BD recruiting tool. You act on the SAME live workspace the \
person you're talking to is looking at -- when you navigate or search, their screen \
changes with you.

Rules:
- Use tools to answer anything about candidates, the pool, or to take an action. Never \
invent a candidate fact, a count, or a name -- call get_pool_summary or \
get_candidate_summary and read the real result.
- Ground every claim in a tool result. If a tool can't find what's asked, say so \
plainly rather than guessing.
- You can navigate pages, run searches, set filters, and manage the shortlist. You \
CANNOT delete a candidate, correct a field, or approve a review -- those require a \
human's own click on the Review page, by design; if asked, say so and point at that \
page.
- Keep replies short: 1-4 sentences. This is a sidebar chat, not a report.
- When you take an action (navigate, search, filter, shortlist), say what you did in \
one clause, not a paragraph -- "Opened Ryan Patel's profile." not an essay."""

TOOLS = [
    {"name": "navigate", "description":
     "Switch the workspace to a different page. Use this when the person asks to see "
     "a specific page (\"take me to analytics\", \"open the requisition workspace\").",
     "input_schema": {"type": "object", "properties": {
         "page": {"type": "string", "enum": ["Search", "Candidate", "Requisition",
                                             "Shortlist", "Intake", "Review",
                                             "Analytics", "System"]}},
         "required": ["page"]}},
    {"name": "search_candidates", "description":
     "Run a natural-language candidate search, exactly as if typed into the Search "
     "bar, and switch to the Search page to show results.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string", "description":
                   "the search text, e.g. 'healthcare equity long/short in APAC'"}},
         "required": ["query"]}},
    {"name": "set_filters", "description":
     "Apply filters on the Search page's filter rail without changing the query text. "
     "Only pass facets the person actually mentioned; omit everything else.",
     "input_schema": {"type": "object", "properties": {
         "region": {"type": "array", "items": {"type": "string"}},
         "strategy": {"type": "array", "items": {"type": "string"}},
         "sector": {"type": "array", "items": {"type": "string"}},
         "skill": {"type": "array", "items": {"type": "string"}},
         "min_years": {"type": "number"}, "max_years": {"type": "number"}},
         "required": []}},
    {"name": "open_candidate", "description":
     "Open one candidate's full profile page by name (fuzzy match is fine, e.g. "
     "'Ryan' or 'Zara Al Rashid').",
     "input_schema": {"type": "object", "properties": {
         "name_or_id": {"type": "string"}}, "required": ["name_or_id"]}},
    {"name": "add_to_shortlist", "description":
     "Add a candidate to the shortlist, with an optional recruiter note.",
     "input_schema": {"type": "object", "properties": {
         "name_or_id": {"type": "string"}, "note": {"type": "string"}},
         "required": ["name_or_id"]}},
    {"name": "remove_from_shortlist", "description": "Remove a candidate from the shortlist.",
     "input_schema": {"type": "object", "properties": {
         "name_or_id": {"type": "string"}}, "required": ["name_or_id"]}},
    {"name": "get_pool_summary", "description":
     "Read real, current pool-wide statistics -- counts by region, strategy, sector, "
     "seniority, review queue size, abstention counts. Use this before answering any "
     "question about the pool as a whole.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "get_candidate_summary", "description":
     "Read one candidate's real profile summary -- experience, strategies, sectors, "
     "seniority, completeness, review status. Use this before answering any question "
     "about a specific person.",
     "input_schema": {"type": "object", "properties": {
         "name_or_id": {"type": "string"}}, "required": ["name_or_id"]}},
    {"name": "list_shortlist", "description": "Read who is currently on the shortlist.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "list_candidates", "description":
     "List candidates with their key comparison fields (name, region, seniority, "
     "years of experience, strategies) in one call. Use this for ANY question that "
     "compares or ranks more than one person -- 'most senior in APAC', 'who has the "
     "most credit experience' -- instead of calling get_candidate_summary once per "
     "person, which wastes turns on a question this answers in one call. Filters are "
     "optional; omit all of them to list the whole pool.",
     "input_schema": {"type": "object", "properties": {
         "region": {"type": "string", "description": "e.g. 'Asia-Pacific', 'Americas', 'Europe / EMEA'"},
         "strategy": {"type": "string"}, "sector": {"type": "string"},
         "min_years": {"type": "number"}}, "required": []}},
]


# --------------------------------------------------------------------- resolution
def _resolve_candidate(name_or_id: str, pool: list) -> tuple[Any | None, str]:
    """Exact id match first, then fuzzy name match. Never guesses silently past a
    weak match -- a low-confidence result is reported as such so the model can ask
    the person to disambiguate rather than confidently opening the wrong record."""
    if not name_or_id:
        return None, "no name given"
    for p in pool:
        if p.candidate_id == name_or_id:
            return p, "exact id match"
    names = {p.candidate_id: p.display_name(False) for p in pool}
    # Real bug this caught: without case normalisation, "Ryan" scored 36% against
    # "RYAN PATEL" (an all-caps name in the actual corpus) but 60% against an
    # unrelated "Omar El-Hassan" -- confidently resolving to the wrong person. Case
    # normalisation alone fixes it (90% vs 60%).
    match = process.extractOne(name_or_id, names, scorer=fuzz.WRatio,
                               processor=fuzz_utils.default_process)
    if match is None:
        return None, f"no candidate resembling {name_or_id!r} found"
    _display, score, cid = match
    if score < 60:
        return None, (f"no confident match for {name_or_id!r} "
                      f"(closest was {_display!r} at {score:.0f}%)")
    by_id = {p.candidate_id: p for p in pool}
    return by_id[cid], f"matched {name_or_id!r} -> {_display!r} ({score:.0f}%)"


def _display(p) -> str:
    return p.display_name(False)


# -------------------------------------------------------------------------- tools
def _tool_navigate(inp: dict, pool: list, store) -> dict:
    import streamlit as st
    page = inp.get("page")
    st.session_state.page = page
    return {"ok": True, "navigated_to": page}


def _tool_search_candidates(inp: dict, pool: list, store) -> dict:
    import streamlit as st
    q = inp.get("query", "")
    st.session_state.query = q
    st.session_state.page = "Search"
    return {"ok": True, "query": q, "note": "the Search page will show live results"}


def _tool_set_filters(inp: dict, pool: list, store) -> dict:
    import streamlit as st
    filters = {}
    for k in ("region", "strategy", "sector", "skill"):
        if inp.get(k):
            filters[k] = inp[k]
    if inp.get("min_years") is not None or inp.get("max_years") is not None:
        filters["years"] = [inp.get("min_years", 0.0), inp.get("max_years", 30.0)]
    if not filters:
        return {"ok": False, "error": "no recognised filter values were supplied"}
    st.session_state.pending_filters = filters
    st.session_state.page = "Search"
    return {"ok": True, "filters_applied": filters}


def _tool_open_candidate(inp: dict, pool: list, store) -> dict:
    import streamlit as st
    p, why = _resolve_candidate(inp.get("name_or_id", ""), pool)
    if p is None:
        return {"ok": False, "error": why}
    st.session_state.selected = p.candidate_id
    st.session_state.page = "Candidate"
    return {"ok": True, "opened": _display(p), "resolution": why}


def _tool_add_to_shortlist(inp: dict, pool: list, store) -> dict:
    import streamlit as st
    # Real bug a live test caught: this assumed app.py's own startup code had already
    # run `st.session_state.setdefault("shortlist", {})`, which is true inside the
    # real app but not guaranteed for a tool called any other way -- the assistant
    # reported the resulting AttributeError back to the user as an honest "technical
    # error" rather than silently failing, which is the right degradation, but the
    # right FIX is for the tool not to depend on unrelated init code having run first.
    st.session_state.setdefault("shortlist", {})
    p, why = _resolve_candidate(inp.get("name_or_id", ""), pool)
    if p is None:
        return {"ok": False, "error": why}
    st.session_state.shortlist[p.candidate_id] = {
        "note": inp.get("note", ""), "tags": "", "source": "assistant", "basis": ""}
    return {"ok": True, "added": _display(p), "resolution": why}


def _tool_remove_from_shortlist(inp: dict, pool: list, store) -> dict:
    import streamlit as st
    st.session_state.setdefault("shortlist", {})
    p, why = _resolve_candidate(inp.get("name_or_id", ""), pool)
    if p is None:
        return {"ok": False, "error": why}
    removed = st.session_state.shortlist.pop(p.candidate_id, None)
    return {"ok": removed is not None, "removed": _display(p),
           "was_on_shortlist": removed is not None}


def _tool_list_shortlist(inp: dict, pool: list, store) -> dict:
    import streamlit as st
    st.session_state.setdefault("shortlist", {})
    by_id = {p.candidate_id: p for p in pool}
    return {"count": len(st.session_state.shortlist),
           "candidates": [_display(by_id[cid]) for cid in st.session_state.shortlist
                          if cid in by_id]}


def _tool_get_pool_summary(inp: dict, pool: list, store) -> dict:
    from collections import Counter
    region = Counter()
    strategy = Counter()
    sector = Counter()
    seniority = Counter()
    for p in pool:
        if p.geo_region:
            region[tx.display("region", p.geo_region.label)] += 1
        for c in p.strategies:
            strategy[tx.display("strategy", c.label)] += 1
        for c in p.sectors:
            sector[tx.display("sector", c.label)] += 1
        if p.seniority:
            seniority[p.seniority.label] += 1
    return {
        "total_candidates": len(pool),
        "needs_review": sum(1 for p in pool if p.quality.needs_human_review),
        "by_region": dict(region.most_common()),
        "by_strategy": dict(strategy.most_common(8)),
        "by_sector": dict(sector.most_common(8)),
        "by_seniority": dict(seniority.most_common()),
        "mean_completeness": round(sum(p.quality.completeness for p in pool) / max(1, len(pool)), 3),
    }


def _tool_get_candidate_summary(inp: dict, pool: list, store) -> dict:
    p, why = _resolve_candidate(inp.get("name_or_id", ""), pool)
    if p is None:
        return {"ok": False, "error": why}
    cur = p.current_role()
    return {
        "ok": True, "resolution": why, "name": _display(p),
        "current_role": (f"{cur.title_raw.display('—')} at "
                         f"{cur.employer_canonical or cur.employer_raw.display('—')}"
                         if cur else "unknown"),
        "years_experience": p.years_experience.display("unknown"),
        "seniority": p.seniority.label if p.seniority else "unknown",
        "region": tx.display("region", p.geo_region.label, "unknown") if p.geo_region else "unknown",
        "strategies": [tx.display("strategy", c.label) for c in p.strategies],
        "sectors": [tx.display("sector", c.label) for c in p.sectors],
        "completeness": p.quality.completeness,
        "evidence_coverage": p.quality.evidence_coverage,
        "needs_human_review": p.quality.needs_human_review,
        "review_reasons": p.quality.review_reasons,
    }


def _tool_list_candidates(inp: dict, pool: list, store) -> dict:
    region = (inp.get("region") or "").strip().lower()
    strategy = (inp.get("strategy") or "").strip().lower()
    sector = (inp.get("sector") or "").strip().lower()
    min_years = inp.get("min_years")

    rows = []
    for p in pool:
        p_region = tx.display("region", p.geo_region.label, "") if p.geo_region else ""
        p_strategies = [tx.display("strategy", c.label) for c in p.strategies]
        p_sectors = [tx.display("sector", c.label) for c in p.sectors]
        if region and region not in p_region.lower():
            continue
        if strategy and not any(strategy in s.lower() for s in p_strategies):
            continue
        if sector and not any(sector in s.lower() for s in p_sectors):
            continue
        yrs = p.years_experience.value
        if min_years is not None and (yrs is None or yrs < min_years):
            continue
        rows.append({
            "name": _display(p), "region": p_region or "unknown",
            "years_experience": yrs if yrs is not None else "unknown",
            "seniority": p.seniority.label if p.seniority else "unknown",
            "strategies": p_strategies, "sectors": p_sectors,
            "needs_review": p.quality.needs_human_review,
        })
    return {"count": len(rows), "candidates": rows}


_DISPATCH = {
    "navigate": _tool_navigate, "search_candidates": _tool_search_candidates,
    "set_filters": _tool_set_filters, "open_candidate": _tool_open_candidate,
    "add_to_shortlist": _tool_add_to_shortlist,
    "remove_from_shortlist": _tool_remove_from_shortlist,
    "list_shortlist": _tool_list_shortlist,
    "get_pool_summary": _tool_get_pool_summary,
    "get_candidate_summary": _tool_get_candidate_summary,
    "list_candidates": _tool_list_candidates,
}


def execute_tool(name: str, tool_input: dict, pool: list, store) -> dict:
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"ok": False, "error": f"unknown tool {name!r}"}
    try:
        return fn(tool_input or {}, pool, store)
    except Exception as e:  # noqa: BLE001 -- a tool failure must not crash the chat
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------------------- the loop
@dataclass
class TurnResult:
    reply: str
    actions: list[str] = field(default_factory=list)
    navigated: bool = False


def _get_client():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise AssistantUnavailable(
            "ANTHROPIC_API_KEY is not set. The assistant makes live, unpredictable "
            "replies and cannot replay from the deterministic demo cache the way the "
            "rest of the app can -- export a key to use it.")
    import anthropic
    return anthropic.Anthropic(api_key=key, timeout=30)


def run_turn(messages: list[dict], pool: list, store) -> TurnResult:
    """`messages` is the full Anthropic-format conversation so far, INCLUDING the new
    user turn already appended by the caller. Mutates it in place with every
    assistant/tool_result block produced this turn, and returns the final reply plus
    a plain-English log of what actions were actually taken.
    """
    client = _get_client()
    actions: list[str] = []
    navigated = False

    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.messages.create(
            model=SETTINGS.llm.model, max_tokens=700, temperature=0.2,
            system=SYSTEM, tools=TOOLS, messages=messages)

        messages.append({"role": "assistant", "content": resp.content})

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            return TurnResult(reply=text or "(no reply)", actions=actions,
                              navigated=navigated)

        results = []
        for tu in tool_uses:
            out = execute_tool(tu.name, tu.input, pool, store)
            if out.get("ok", True):
                if tu.name in ("navigate", "search_candidates", "set_filters",
                               "open_candidate"):
                    navigated = True
                actions.append(f"{tu.name}({json.dumps(tu.input, default=str)})")
            results.append({"type": "tool_result", "tool_use_id": tu.id,
                            "content": json.dumps(out, default=str),
                            "is_error": not out.get("ok", True)})
        messages.append({"role": "user", "content": results})

    return TurnResult(reply="I took several steps but didn't reach a final answer in "
                            "time -- try a more specific request.",
                      actions=actions, navigated=navigated)
