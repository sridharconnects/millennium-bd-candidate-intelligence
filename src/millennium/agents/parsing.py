"""Agent 2 -- Parsing. The required LLM-via-API path, with a rule layer beside it.

Order of operations, and why:

1. **Segment first, cheaply.** Sections are found by rules and layout before any LLM
   call. Segmentation is a formatting problem, not a language problem, and doing it in
   Python keeps prompts small and their failure modes local.
2. **LLM extracts, in three targeted passes.** One giant prompt degrades on long
   documents; three focused ones each get a short task and a short schema. Identity /
   employment / profile also fail independently, which is what lets a single bad pass
   degrade to abstained fields instead of losing the whole candidate.
3. **Rules cross-check, they do not replace.** The case study requires LLM parsing, so
   the LLM is the primary path. Regex runs alongside it on the handful of fields where
   regex is genuinely better -- email, phone, dates, degree level, certifications --
   purely as a second opinion. Agreement raises confidence; disagreement lowers it and
   routes the field to human review.
4. **A fourth pass runs only on conflicts.** Adjudication is expensive and usually
   unnecessary, so it is conditional on there being something to adjudicate.
"""
from __future__ import annotations

import re

from .. import prompts, taxonomy as tx
from ..config import SETTINGS
from ..llm import LLMClient, LLMUnavailable
from ..schema import (Certification, DateRange, EducationEntry, EmploymentEntry,
                      Evidence, LanguageEntry, Tracked)
from ..validate import (check_email, check_phone, parse_date, parse_duration, verify_span)
from .base import AgentResult, subagent

AGENT = "parsing"

# Section headings observed across the supplied corpus, plus common synonyms.
SECTION_PATTERNS: list[tuple[str, str]] = [
    ("experience", r"(work(ing)?\s+experience|professional\s+experience|experience|"
                   r"employment|academic\s+profile|career\s+history)"),
    ("education", r"(education(al)?(\s+qualification)?|academic\s+background|qualifications?)"),
    ("skills", r"(skills?|technical\s+skills?|it\s+skills?|database\s+skills?|"
               r"computer\s+skills?|quantitive\s+skills?|quantitative\s+skills?|competencies)"),
    ("certifications", r"(certifications?|licen[cs]es?|certifications?/licen[cs]es?)"),
    ("summary", r"(profile\s+overview|summary|professional\s+summary|objective|about)"),
    ("projects", r"(key\s+projects?|projects?|selected\s+transaction)"),
    ("activities", r"(activities|extra[\s-]?curricular|leadership|volunteer|interests|"
                   r"scholastic\s+achievements?|awards?)"),
    ("personal", r"(personal\s+information|personal\s+details|additional\s+information|"
                 r"languages?)"),
]


@subagent("parse.segment_sections", AGENT, "1.2")
def segment_sections(text: str) -> AgentResult:
    """Locate section boundaries by rules and layout, before any model call.

    A heading is recognised by shape, not just by keyword: short line, mostly capitals
    or title case, no terminal punctuation. That is what lets this work on the
    table-derived headings ('EDUCATIONAL QUALIFICATION', 'ACADEMIC PROFILE') that this
    corpus is full of.
    """
    lines = text.split("\n")
    bounds, pos = [], 0
    for ln in lines:
        bounds.append((pos, pos + len(ln), ln))
        pos += len(ln) + 1

    found: list[tuple[str, int, str]] = []
    for start, _end, ln in bounds:
        s = ln.strip().strip("|").strip()
        if not s or len(s) > 60 or s.endswith((".", ",", ";", ":")) and len(s) > 40:
            continue
        letters = [c for c in s if c.isalpha()]
        if not letters:
            continue
        shouty = sum(c.isupper() for c in letters) / len(letters) > 0.7
        # A short line that is *entirely* a section name counts as a heading whatever
        # its capitalisation. Omar's CV uses sentence case ("Work experience"), which
        # a title-case-or-shouty test rejects — and losing the experience section on a
        # CV that is mostly work history is not a small loss.
        norm = tx.norm(re.sub(r"[^A-Za-z\s/&-]", " ", s))
        short_exact = len(s) <= 34 and s[0].isupper()
        if not (s.istitle() or shouty or short_exact):
            continue
        for name, pattern in SECTION_PATTERNS:
            if re.fullmatch(rf"\s*{pattern}\s*", norm):
                found.append((name, start, s))
                break

    sections: dict[str, list[int]] = {}
    for i, (name, start, _label) in enumerate(found):
        end = found[i + 1][1] if i + 1 < len(found) else len(text)
        if name in sections:                       # a repeated heading extends the span
            sections[name][1] = max(sections[name][1], end)
        else:
            sections[name] = [start, end]
    return AgentResult(
        name="", output=sections, confidence=min(1.0, len(sections) / 4),
        warnings=[] if sections else ["no section headings recognised; the whole "
                                      "document is treated as one block"])


# ------------------------------------------------------------------- LLM passes
def _llm_pass(client: LLMClient, builder, stage: str, *args) -> AgentResult:
    try:
        system, msgs, hint = builder(*args)
        r = client.complete_json(system, msgs, hint, stage=stage)
        return AgentResult(name="", output=r.data, confidence=1.0 if not r.cached else 1.0,
                           tokens_in=r.tokens_in, tokens_out=r.tokens_out,
                           cost_usd=r.cost_usd, cached=r.cached, latency_ms=r.latency_ms,
                           warnings=[f"replayed from cache"] if r.cached else [])
    except LLMUnavailable as e:
        return AgentResult(name="", status="failed", output=None, confidence=0.0,
                           errors=[str(e)],
                           warnings=["LLM unavailable -- all fields from this pass abstain"])


@subagent("parse.llm_identity", AGENT, "1.3")
def llm_identity(client: LLMClient, text: str) -> AgentResult:
    """LLM pass 1: identity, contact, education, certifications, languages."""
    return _llm_pass(client, prompts.identity_prompt, "identity", text)


@subagent("parse.llm_employment", AGENT, "1.3")
def llm_employment(client: LLMClient, text: str) -> AgentResult:
    """LLM pass 2: employment history with per-entry dates and attributed highlights."""
    return _llm_pass(client, prompts.employment_prompt, "employment", text)


@subagent("parse.llm_profile", AGENT, "1.3")
def llm_profile(client: LLMClient, text: str) -> AgentResult:
    """LLM pass 3: strategy / sector / skills / feeder-path classification."""
    return _llm_pass(client, prompts.profile_prompt, "profile", text)


# --------------------------------------------------- rule-baseline alternative
# NOT the case-study's required path -- see extract_rules.py for why it exists.
# Output shape is identical to the LLM passes, including verbatim quotes, so the same
# span-verification and merge code runs unchanged and an ungrounded rule guess is
# discarded on exactly the same terms as an ungrounded model guess.
@subagent("parse.rule_identity", AGENT, "1.1")
def rule_identity(text: str, sections: dict | None = None) -> AgentResult:
    """Rule baseline: identity, contact, education, certifications, languages."""
    from ..extract_rules import identity_extract
    return AgentResult(name="", output=identity_extract(text, sections or {}),
                       confidence=0.6,
                       warnings=["rule baseline — NOT the LLM API path"])


@subagent("parse.rule_employment", AGENT, "1.1")
def rule_employment(text: str, sections: dict | None = None) -> AgentResult:
    """Rule baseline: employment entries from section and line structure."""
    from ..extract_rules import employment_extract
    return AgentResult(name="", output=employment_extract(text, sections or {}),
                       confidence=0.55,
                       warnings=["rule baseline — NOT the LLM API path"])


@subagent("parse.rule_profile", AGENT, "1.1")
def rule_profile(text: str, sections: dict | None = None) -> AgentResult:
    """Rule baseline: lexical strategy/sector/skill classification."""
    from ..extract_rules import profile_extract
    return AgentResult(name="", output=profile_extract(text, sections or {}),
                       confidence=0.5,
                       warnings=["rule baseline — NOT the LLM API path"])


@subagent("parse.llm_adjudicate", AGENT, "1.1")
def llm_adjudicate(client: LLMClient, text: str, conflicts: list[dict]) -> AgentResult:
    """LLM pass 4 (conditional): resolve rule-vs-LLM disagreements, or decline to."""
    if not conflicts:
        return AgentResult(name="", status="skipped", output={"resolutions": []})
    return _llm_pass(client, prompts.adjudication_prompt, "adjudicate", text, conflicts)


# ------------------------------------------------------------------- rule layer
EMAIL_RX = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)*")
PHONE_RX = re.compile(r"(?:(?<=\D)|^)(\+?\d[\d\s\-().]{7,18}\d)(?=\D|$)")
YEAR_RX = re.compile(r"\b(19|20)\d{2}\b")


@subagent("parse.rule_contacts", AGENT, "1.2")
def rule_contacts(text: str, doc_id: str, header_footer: str = "") -> AgentResult:
    """High-precision regex extraction, used only as a second opinion on the LLM.

    Handles two corpus-specific traps: an ISSN in a publication citation that matches
    a phone pattern, and an agency watermark in the header that is provenance rather
    than candidate data.
    """
    warnings: list[str] = []
    emails = []
    for m in EMAIL_RX.finditer(text):
        v = m.group(0).rstrip(".")
        ok, why = check_email(v)
        emails.append({"value": v, "start": m.start(), "end": m.start() + len(v),
                       "valid": ok, "note": why})
    phones = []
    for m in PHONE_RX.finditer(text):
        v = m.group(1).strip()
        ctx = text[max(0, m.start() - 60):m.end() + 20]
        ok, why = check_phone(v, ctx)
        if not ok and "ISSN" in why:
            warnings.append(f"rejected {v!r} as a phone number: {why}")
            continue
        phones.append({"value": v, "start": m.start(1), "end": m.end(1),
                       "valid": ok, "note": why})
    certs = [{"canonical": c, "status": st, "surface": sf, "start": s, "end": e}
             for c, st, sf, s, e in tx.match_certifications(text)]

    agency = None
    if header_footer:
        hf = header_footer.strip()
        if hf and not re.fullmatch(r"[\d\s\-/]+", hf) and len(hf) < 120:
            agency = hf
            warnings.append(f"document carries an agency/source watermark: {hf!r} -- "
                            f"recorded as provenance, excluded from candidate fields")

    langs = []
    low = tx.norm(text)
    for lang in tx.LANGUAGE_NAMES:
        for m in re.finditer(rf"\b{lang}\b", low):
            window = low[m.start():m.start() + 90]
            prof = next((p for p in tx.PROFICIENCY if p in window), None)
            langs.append({"language": lang.title(), "proficiency": prof,
                          "start": m.start(), "end": m.end()})
            break

    return AgentResult(
        name="", output={"emails": emails, "phones": phones, "certifications": certs,
                         "languages": langs, "agency_watermark": agency,
                         "years": sorted({m.group(0) for m in YEAR_RX.finditer(text)})},
        confidence=0.95, warnings=warnings)


# --------------------------------------------------------------- merge + verify
def _track(value, quote: str | None, text: str, doc_id: str, page=None,
           method="llm", conf: float = 0.8, normalized=None) -> Tracked:
    """Wrap a model-proposed value, grounding it or abstaining.

    This is the single choke point where an unprovable claim is destroyed. Everything
    the UI later displays as fact passed through here.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return Tracked.missing()
    ev = verify_span(quote, text, doc_id, page, SETTINGS.span_fuzzy_threshold) if quote else None
    if ev is None:
        return Tracked(value=None, confidence=0.0, extraction_method=method,
                       validation_status="abstained",
                       notes=[f"proposed value {str(value)[:60]!r} discarded: its quote "
                              f"could not be located in the source document"])
    penalty = {"exact": 0.0, "normalized": 0.03, "fuzzy": 0.12}[ev.match_kind]
    return Tracked(value=value, normalized_value=normalized if normalized is not None else value,
                   confidence=round(max(0.0, conf - penalty), 3), evidence=[ev],
                   extraction_method=method, validation_status="verified")


def _g(d, key, default=None):
    """Model output is untrusted in shape as well as content."""
    if not isinstance(d, dict):
        return default
    v = d.get(key, default)
    return v if v is not None else default


def _vq(node) -> tuple:
    """Unpack a {'value':..,'quote':..} node, tolerating a bare scalar."""
    if isinstance(node, dict):
        return _g(node, "value"), _g(node, "quote", "")
    return node, ""


@subagent("parse.merge_identity", AGENT, "1.3")
def merge_identity(ident: dict | None, rules: dict, text: str, doc_id: str) -> AgentResult:
    """Ground identity/contact/education/cert fields and cross-check against rules."""
    ident = ident or {}
    conflicts: list[dict] = []
    warnings: list[str] = []

    name_v, name_q = _vq(_g(ident, "full_name"))
    email_v, email_q = _vq(_g(ident, "email"))
    phone_v, phone_q = _vq(_g(ident, "phone"))

    name = _track(name_v, name_q, text, doc_id, conf=0.9)
    email = _track(email_v, email_q, text, doc_id, conf=0.9)
    phone = _track(phone_v, phone_q, text, doc_id, conf=0.9)

    # --- cross-check: email
    rule_emails = [e for e in rules.get("emails", [])]
    if email.is_known:
        ok, why = check_email(str(email.value))
        if not ok:
            email.confidence = round(email.confidence * 0.5, 3)
            email.notes.append(f"format check failed: {why}")
            warnings.append(f"contact quality: {why}")
        if rule_emails and tx.norm(str(email.value)) != tx.norm(rule_emails[0]["value"]):
            conflicts.append({"field": "email", "rule": rule_emails[0]["value"],
                              "llm": email.value})
    elif rule_emails:
        r0 = rule_emails[0]
        email = Tracked(value=r0["value"], normalized_value=r0["value"],
                        confidence=0.9 if r0["valid"] else 0.45,
                        extraction_method="rule", validation_status="verified",
                        evidence=[Evidence(doc_id=doc_id, char_start=r0["start"],
                                           char_end=r0["end"], snippet=r0["value"])],
                        notes=[r0["note"]])

    rule_phones = [p for p in rules.get("phones", []) if p["valid"]]
    if not phone.is_known and rule_phones:
        p0 = rule_phones[0]
        phone = Tracked(value=p0["value"], normalized_value=p0["value"], confidence=0.88,
                        extraction_method="rule", validation_status="verified",
                        evidence=[Evidence(doc_id=doc_id, char_start=p0["start"],
                                           char_end=p0["end"], snippet=p0["value"])])

    # --- education
    education: list[EducationEntry] = []
    for e in _g(ident, "education", []) or []:
        inst_v, inst_q = _vq(_g(e, "institution"))
        deg_v, deg_q = _vq(_g(e, "degree_raw"))
        fos_v, fos_q = _vq(_g(e, "field_of_study"))
        yr_v, yr_q = _vq(_g(e, "graduation_year"))
        gpa_v, gpa_q = _vq(_g(e, "gpa_raw"))
        loc_v, loc_q = _vq(_g(e, "location"))
        entry = EducationEntry(
            institution=_track(inst_v, inst_q, text, doc_id, conf=0.9),
            degree_raw=_track(deg_v, deg_q, text, doc_id, conf=0.9),
            field_of_study=_track(fos_v, fos_q, text, doc_id, conf=0.85),
            graduation_year=_track(_int(yr_v), yr_q, text, doc_id, conf=0.85),
            gpa_raw=_track(gpa_v, gpa_q, text, doc_id, conf=0.85),
            location=_track(loc_v, loc_q, text, doc_id, conf=0.8),
            honors=[h for h in (_g(e, "honors", []) or []) if isinstance(h, str)],
        )
        # Rule cross-check on degree level: a closed vocabulary regex beats the model here.
        entry.degree_level = tx.degree_level(f"{deg_v or ''} {fos_v or ''}")
        education.append(entry)

    # --- certifications: rules lead, the model supplies the year
    certs: list[Certification] = []
    llm_certs = {tx.norm(str(_vq(_g(c, "name"))[0] or "")): c
                 for c in (_g(ident, "certifications", []) or [])}
    for rc in rules.get("certifications", []):
        yr = Tracked.missing()
        for key, lc in llm_certs.items():
            if rc["canonical"].replace("_", " ") in key or key in rc["canonical"]:
                yv, yq = _vq(_g(lc, "year"))
                yr = _track(_int(yv), yq, text, doc_id, conf=0.85)
                break
        certs.append(Certification(
            name=Tracked(value=tx.CERTIFICATIONS[rc["canonical"]]["display"],
                         normalized_value=rc["canonical"], confidence=0.93,
                         extraction_method="rule", validation_status="verified",
                         evidence=[Evidence(doc_id=doc_id, char_start=rc["start"],
                                            char_end=rc["end"], snippet=rc["surface"])]),
            canonical=rc["canonical"], status=rc["status"], year=yr))
    # De-duplicate: 'CFA' and 'CFA Charterholder' are one credential.
    seen: dict[str, Certification] = {}
    for c in certs:
        prev = seen.get(c.canonical or "")
        if prev is None or (c.status and not prev.status):
            seen[c.canonical or ""] = c
    certs = list(seen.values())

    # --- languages: union of rule hits and model output, model wins on proficiency
    langs: dict[str, LanguageEntry] = {}
    for l in rules.get("languages", []):
        langs[l["language"].lower()] = LanguageEntry(language=l["language"],
                                                     proficiency=l["proficiency"])
    for l in _g(ident, "languages", []) or []:
        if not isinstance(l, dict):
            continue
        nm = str(_g(l, "language", "")).strip()
        if not nm:
            continue
        ev = verify_span(_g(l, "quote", ""), text, doc_id)
        cur = langs.get(nm.lower()) or LanguageEntry(language=nm.title())
        cur.proficiency = _g(l, "proficiency") or cur.proficiency
        cur.evidence = [ev] if ev else cur.evidence
        langs[nm.lower()] = cur

    loc_v, loc_q = _vq(_g(ident, "location_current"))
    head_v, head_q = _vq(_g(ident, "headline"))
    summ_v, summ_q = _vq(_g(ident, "summary"))
    marital_v, marital_q = _vq(_g(ident, "marital_status"))
    addr_v, addr_q = _vq(_g(ident, "home_address"))
    auth_v, auth_q = _vq(_g(ident, "work_authorization"))

    return AgentResult(name="", output={
        "full_name": name, "email": email, "phone": phone,
        "home_address": _track(addr_v, addr_q, text, doc_id, conf=0.85),
        "marital_status": _track(marital_v, marital_q, text, doc_id, conf=0.85),
        "location_current": _track(loc_v, loc_q, text, doc_id, conf=0.85),
        "headline": _track(head_v, head_q, text, doc_id, conf=0.8),
        "summary": _track(summ_v, summ_q, text, doc_id, conf=0.8),
        "work_authorization": _track(auth_v, auth_q, text, doc_id, conf=0.8),
        "education": education, "certifications": certs,
        "languages": list(langs.values()),
        "conflicts": conflicts,
        "suspicious": _g(ident, "suspicious_content", []) or [],
    }, confidence=0.9 if name.is_known else 0.55, warnings=warnings)


def _int(v):
    try:
        return int(str(v).strip()[:4])
    except (TypeError, ValueError):
        return None


@subagent("parse.merge_employment", AGENT, "1.3")
def merge_employment(emp: dict | None, text: str, doc_id: str) -> AgentResult:
    """Ground each employment entry, normalise dates, canonicalise employers."""
    entries: list[EmploymentEntry] = []
    warnings: list[str] = []
    for e in (_g(emp or {}, "employment", []) or []):
        er_v, er_q = _vq(_g(e, "employer_raw"))
        ti_v, ti_q = _vq(_g(e, "title_raw"))
        lo_v, lo_q = _vq(_g(e, "location"))
        st_v, st_q = _vq(_g(e, "start"))
        en_v, en_q = _vq(_g(e, "end"))
        du_v, du_q = _vq(_g(e, "duration_text"))

        employer = _track(er_v, er_q, text, doc_id, conf=0.92)
        title = _track(ti_v, ti_q, text, doc_id, conf=0.9)
        if not employer.is_known and not title.is_known:
            warnings.append("dropped an employment entry: neither employer nor title "
                            "could be grounded in the document")
            continue

        s_norm, s_note = parse_date(st_v)
        e_norm, e_note = parse_date(en_v)
        start = _track(st_v, st_q, text, doc_id, conf=0.88, normalized=s_norm)
        end = _track(en_v, en_q, text, doc_id, conf=0.88, normalized=e_norm)

        months = None
        if s_norm and e_norm:
            from ..validate import months_between
            months = months_between(s_norm, e_norm)
        dur_months = parse_duration(du_v)
        dur_track = Tracked.missing()
        if months is not None:
            dur_track = Tracked.derived(months, 0.9, f"{s_norm} -> {e_norm}")
        elif dur_months is not None:
            dur_track = _track(dur_months, du_q, text, doc_id, conf=0.8, normalized=dur_months)
            if dur_track.is_known:
                dur_track.notes.append("tenure stated as a duration with no dates; "
                                       "absolute dates are unknown, not assumed")

        canon, tier = tx.canonical_employer(str(employer.value or ""))
        level, _why = tx.title_to_level(str(title.value or ""), tier)
        highlights = []
        for h in (_g(e, "highlights", []) or []):
            hv, hq = _vq(h)
            t = _track(hv, hq, text, doc_id, conf=0.85)
            if t.is_known:
                highlights.append(t)

        entries.append(EmploymentEntry(
            employer_raw=employer, employer_canonical=canon, employer_tier=tier,
            title_raw=title, title_normalized=str(title.value or "").strip() or None,
            seniority_level=level, location=_track(lo_v, lo_q, text, doc_id, conf=0.8),
            dates=DateRange(start=start, end=end,
                            is_current=(str(e_norm or "").lower() == "present"),
                            duration_months=dur_track),
            is_internship=bool(_g(e, "is_internship", False)),
            is_volunteer=bool(_g(e, "is_volunteer", False)),
            highlights=highlights))

    # Most recent first, undated entries last -- the UI and seniority logic assume this.
    def sort_key(x: EmploymentEntry):
        s = x.dates.start.normalized_value or x.dates.start.value
        if x.dates.is_current:
            return (0, "9999")
        return (1, str(s or "0000"))
    entries.sort(key=sort_key, reverse=False)
    entries.sort(key=lambda x: (not x.dates.is_current,
                                -(int(str(x.dates.start.normalized_value or "0")[:4]) or 0)))
    return AgentResult(name="", output=entries, confidence=0.9 if entries else 0.2,
                       warnings=warnings,
                       errors=[] if entries else ["no employment entries could be grounded"])
