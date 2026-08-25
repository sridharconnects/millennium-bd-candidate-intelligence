"""A deterministic, rule-only extractor producing the same JSON shapes as the LLM passes.

WHAT THIS IS AND IS NOT
-----------------------
This is **not** the case-study's required parsing path. The case study mandates
"parse resume data ... using LLM models via API", and `agents/parsing.py` does exactly
that. This module exists for two other reasons, both real:

1. **A published baseline.** The README promises to "break out rule vs LLM
   performance". You cannot report that comparison without actually running a rule
   extractor over the same documents and scoring it against the same gold set. This is
   that extractor. Where it beats the model — and on dates and emails it usually does —
   that is the finding, and it gets reported rather than buried.

2. **A zero-cost CI path.** Every stage downstream of extraction (span grounding,
   classification, validation, export, indexing, retrieval, matching) can be exercised
   end to end on the ten real resumes without an API key and without spending money on
   every test run.

HONESTY CONSTRAINTS, enforced downstream:
  * Profiles produced this way are stamped `llm_model=None` and
    `extractor="rule-baseline-v1"` in provenance. Nothing pretends to be an API call.
  * Fields are stamped `extraction_method="rule"`, so the UI and the evaluation report
    them separately.
  * Output still carries verbatim quotes, so span verification runs exactly as it does
    on model output. An ungrounded rule guess is discarded on the same terms.

It is a baseline, not a competitor: it has no semantics, so it cannot resolve the
attribution traps (a company named inside a bullet) that the LLM prompt handles
explicitly. Its scores are expected to be lower, and that gap is the point.
"""
from __future__ import annotations

import re

from . import taxonomy as tx
from .validate import parse_date

VERSION = "rule-baseline-v1"

EMAIL_RX = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)*")
PHONE_RX = re.compile(r"(?:(?<=\D)|^)(\+?\d[\d\s\-().]{7,18}\d)(?=\D|$)")

MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
YEAR = r"(?:19|20)\d{2}"
SHORT_YR = r"'?\d{2}"
_TOKEN = rf"(?:{MONTH}[\s.\-']*(?:{YEAR}|{SHORT_YR})|{MONTH}\s*[-']\s*{SHORT_YR}|\d{{1,2}}/{YEAR}|{YEAR})"
PRESENT = r"(?:Present|present|Current|current|till date|till now|Now|now|ongoing|Date)"
# A date range: two tokens joined by a dash/to, or a token followed by 'Present'.
RANGE_RX = re.compile(
    rf"({_TOKEN})\s*(?:[-–—]|to|To|\bthrough\b)\s*({_TOKEN}|{PRESENT})|"
    rf"(?:Since|since|From|from)\s+({_TOKEN})|"
    rf"({_TOKEN})\s*[-–—]\s*({PRESENT})")
DURATION_RX = re.compile(r"(\d+\s*years?(?:\s*\d+\s*months?)?|\d+\s*months?)", re.I)

BULLET_START = re.compile(r"^\s*(?:[•▪◦‣·§\-*]|\d+[.)])\s+")
SEPARATORS = re.compile(r"\s+[–—|]\s+|\s{2,}|\t+|,\s(?=[A-Z][a-z]+,?\s*[A-Z]{2}\b)")

INTERN_RX = re.compile(r"\b(intern|internship|trainee|summer analyst)\b", re.I)
DEGREE_RX = re.compile(
    r"\b(ph\.?d|mba|pgdm|pgp|m\.?b\.?b\.?s|m\.?tech|m\.?com|m\.?s\.?c?\b|master[’']?s?|"
    r"b\.?s\.?c?\b|b\.?a\b|b\.?tech|b\.?com|b\.?m\.?s|bba|bachelor|"
    r"diplôme d[’']ingénieur|diplome d[’']ingenieur|hsc|ssc|xii std|x std)", re.I)
INSTITUTION_RX = re.compile(
    r"\b(university|universite|université|college|school|institute|institut|academy|"
    r"insa|iit|iim|polytechnic)\b", re.I)


TITLE_WORDS = re.compile(
    r"\b(analyst|associate|manager|director|intern|trainee|vice president|vp|"
    r"president|officer|engineer|developer|strategist|consultant|researcher|"
    r"partner|head|chief|specialist|assistant|representative)\b", re.I)
ORG_WORDS = re.compile(
    r"\b(inc|llc|ltd|limited|plc|pvt|corp|corporation|group|holdings?|partners?|"
    r"capital|management|securities|bank|advisors?|associates|company|co|&|"
    r"university|college|school|institute)\b", re.I)


def _looks_like_org(s: str) -> bool:
    """Distinguish an employer line from a job-title line.

    Three independent signals, any of which is enough: an organisational suffix, a
    fully-capitalised line (how most of this corpus writes employers), or the simple
    absence of any job-title noun.
    """
    s = s.strip()
    if not s or len(s) > 90:
        return False
    if TITLE_WORDS.search(s) and not ORG_WORDS.search(s):
        return False
    letters = [c for c in s if c.isalpha()]
    shouty = letters and sum(c.isupper() for c in letters) / len(letters) > 0.75
    return bool(ORG_WORDS.search(s) or shouty or not TITLE_WORDS.search(s))


def _prev_header(lines, i: int) -> str | None:
    """Nearest preceding non-bullet, undated, short line."""
    for j in range(i - 1, max(-1, i - 4), -1):
        cand = lines[j][2].strip()
        if not cand or BULLET_START.match(lines[j][2]):
            continue
        if RANGE_RX.search(cand) or len(cand) > 90:
            return None
        return cand.split("|")[0].strip(" ,")
    return None


def _lines_with_offsets(text: str) -> list[tuple[int, int, str]]:
    out, pos = [], 0
    for ln in text.split("\n"):
        out.append((pos, pos + len(ln), ln))
        pos += len(ln) + 1
    return out


def _vq(value, quote):
    """The {value, quote} node the merge layer expects."""
    return {"value": value, "quote": quote}


def _section(text: str, sections: dict, name: str) -> tuple[int, int]:
    span = sections.get(name)
    return (span[0], span[1]) if span else (0, len(text))


# --------------------------------------------------------------------- identity
def identity_extract(text: str, sections: dict | None = None) -> dict:
    sections = sections or {}
    lines = _lines_with_offsets(text)
    non_empty = [(a, b, ln) for a, b, ln in lines if ln.strip()]

    # Name: the first substantive line, minus trailing credentials. Real CVs lead with
    # the name; the exceptions in this corpus lead with a section heading, which the
    # length and keyword guards below reject.
    name = name_q = None
    for _a, _b, ln in non_empty[:4]:
        cand = ln.strip()
        if len(cand) > 60 or EMAIL_RX.search(cand) or PHONE_RX.search(cand):
            continue
        if re.search(r"(experience|education|profile|summary|skills|resume|curriculum)",
                     cand, re.I):
            continue
        if not re.match(r"^[A-Za-z][A-Za-z.,'’\-\s()]+$", cand):
            continue
        name_q = cand
        name = re.sub(r",?\s*(CFA|CPA|FRM|CAIA|PhD|MBA|MD)\b\.?", "", cand,
                      flags=re.I).strip(" ,")
        break

    m = EMAIL_RX.search(text)
    email = _vq(m.group(0).rstrip("."), m.group(0)) if m else _vq(None, "")

    phone = _vq(None, "")
    for pm in PHONE_RX.finditer(text):
        ctx = text[max(0, pm.start() - 60):pm.end() + 20]
        if re.search(r"\b(issn|isbn)\b", ctx, re.I):
            continue
        digits = len(re.findall(r"\d", pm.group(1)))
        if 7 <= digits <= 15:
            phone = _vq(pm.group(1).strip(), pm.group(1))
            break

    # Location: the first geography token that appears near the top of the document.
    loc = _vq(None, "")
    for country, _region, surface, s, _e in tx.match_geography(text[:900]):
        for a, b, ln in lines:
            if a <= s < b:
                loc = _vq(country, ln.strip()[:120])
                break
        if loc["value"]:
            break

    marital = _vq(None, "")
    mm = re.search(r"Marital\s+Status\s*[-:–]\s*(\w+)", text, re.I)
    if mm:
        marital = _vq(mm.group(1), mm.group(0))

    education = _education_extract(text, sections)
    certs = [{"name": _vq(tx.CERTIFICATIONS[c]["display"], surface), "year": _vq(None, "")}
             for c, _st, surface, _s, _e in tx.match_certifications(text)]
    seen, uniq = set(), []
    for c in certs:
        if c["name"]["value"] in seen:
            continue
        seen.add(c["name"]["value"])
        uniq.append(c)

    languages = []
    low = tx.norm(text)
    for lang in tx.LANGUAGE_NAMES:
        mm = re.search(rf"(?<!\w){lang}(?!\w)", low)
        if not mm:
            continue
        window = low[mm.start():mm.start() + 90]
        prof = next((p for p in tx.PROFICIENCY if p in window), None)
        idx = low.find(lang)
        quote = text[max(0, idx):idx + len(lang)]
        languages.append({"language": lang.title(), "proficiency": prof, "quote": quote})

    headline = _vq(None, "")
    emp = employment_extract(text, sections).get("employment") or []
    if emp:
        t = emp[0]["title_raw"]["value"]
        e = emp[0]["employer_raw"]["value"]
        if t and e:
            headline = _vq(f"{t} at {e}", emp[0]["title_raw"]["quote"])

    return {
        "full_name": _vq(name, name_q or ""), "email": email, "phone": phone,
        "home_address": _vq(None, ""), "location_current": loc,
        "headline": headline, "summary": _vq(None, ""),
        "marital_status": marital, "work_authorization": _vq(None, ""),
        "education": education, "certifications": uniq, "languages": languages,
        "suspicious_content": [],
    }


def _education_extract(text: str, sections: dict) -> list[dict]:
    lo, hi = _section(text, sections, "education")
    block = text[lo:hi]
    out = []
    for _a, _b, ln in _lines_with_offsets(block):
        s = ln.strip()
        if not s or BULLET_START.match(s):
            continue
        has_deg = DEGREE_RX.search(s)
        has_inst = INSTITUTION_RX.search(s)
        if not (has_deg or has_inst):
            continue
        # Table rows arrive as 'Year | Degree | Institute | Result'.
        parts = [p.strip() for p in s.split("|")] if "|" in s else \
            [p.strip() for p in SEPARATORS.split(s) if p and p.strip()]
        inst = next((p for p in parts if INSTITUTION_RX.search(p)), None)
        deg = next((p for p in parts if DEGREE_RX.search(p)), None)
        if not (inst or deg):
            continue
        yrs = re.findall(YEAR, s)
        year = int(yrs[-1]) if yrs else None
        gpa = None
        gm = re.search(r"(\d\.\d{1,2}\s*/\s*\d(?:\.\d)?|\d{2,3}(?:\.\d+)?\s*%)", s)
        if gm:
            gpa = gm.group(1)
        out.append({
            "institution": _vq(inst, inst or s[:90]),
            "degree_raw": _vq(deg, deg or s[:90]),
            "field_of_study": _vq(None, ""),
            "graduation_year": _vq(year, str(year) if year else ""),
            "gpa_raw": _vq(gpa, gpa or ""),
            "location": _vq(None, ""), "honors": [],
        })
    return out[:10]


# ------------------------------------------------------------------ employment
def employment_extract(text: str, sections: dict | None = None) -> dict:
    sections = sections or {}
    lo, hi = _section(text, sections, "experience")
    block = text[lo:hi]
    lines = _lines_with_offsets(block)
    entries: list[dict] = []

    for i, (_a, _b, raw) in enumerate(lines):
        ln = raw.strip()
        if not ln or BULLET_START.match(raw):
            continue
        m = RANGE_RX.search(ln)
        dur = None
        if not m and (len(ln) <= 110 or "|" in ln):
            dur = DURATION_RX.search(ln)
        if not m and not dur:
            continue

        date_txt = m.group(0) if m else dur.group(0)
        head = ln[:ln.find(date_txt)].strip(" \t|–—-,")
        # Table rows: 'Organization | Axis Mutual Fund | Duration | 2 months'
        if "|" in ln:
            cells = [c.strip() for c in ln.split("|") if c.strip()]
            head = cells[1] if len(cells) > 1 else head

        employer = title = None
        if head:
            parts = [p.strip(" ,") for p in SEPARATORS.split(head) if p and p.strip()]
            parts = [p for p in parts if len(p) > 1]
            if len(parts) >= 2:
                employer, title = parts[0], parts[1]
            elif parts:
                one = parts[0]
                prev = _prev_header(lines, i)
                # A single fragment on a dated line is usually the job title, with the
                # employer on the line above -- the dominant layout in this corpus.
                if prev and _looks_like_org(prev) and not _looks_like_org(one):
                    employer, title = prev, one
                else:
                    employer = one

        # A header carrying only dates takes its employer from the line above and its
        # title from the line below -- the two-line layout most of this corpus uses.
        if not employer:
            employer = _prev_header(lines, i)
        if not title:
            for j in range(i + 1, min(len(lines), i + 3)):
                nxt = lines[j][2].strip()
                if not nxt or RANGE_RX.search(nxt):
                    continue
                if len(nxt) < 80 and not BULLET_START.match(lines[j][2]):
                    title = nxt.split("|")[0].strip(" ,")
                    break
        if not employer and not title:
            continue

        start = end = None
        if m:
            g = [x for x in m.groups() if x]
            if len(g) >= 2:
                start, end = g[0], g[1]
            elif g:
                start, end = g[0], "present"
            if end and re.match(rf"^{PRESENT}$", str(end)):
                end = "present"
        sv = parse_date(start)[0] if start else None
        ev = parse_date(end)[0] if end else None

        # Collect the bullets that belong to this header.
        highlights = []
        for j in range(i + 1, len(lines)):
            nxt_raw, nxt = lines[j][2], lines[j][2].strip()
            if not nxt:
                continue
            if RANGE_RX.search(nxt) and len(nxt) < 120:
                break
            if len(nxt) > 45:
                highlights.append({"value": nxt[:400], "quote": nxt[:200]})
            if len(highlights) >= 5:
                break

        entries.append({
            "employer_raw": _vq(employer, employer or ""),
            "title_raw": _vq(title, title or ""),
            "location": _vq(None, ""),
            "start": _vq(sv, start or ""), "end": _vq(ev, end or ""),
            "duration_text": _vq(dur.group(0) if dur else None,
                                 dur.group(0) if dur else ""),
            "is_internship": bool(INTERN_RX.search(f"{title or ''} {employer or ''}")),
            "highlights": highlights,
        })

    # Drop entries whose employer is obviously a heading rather than a company.
    entries = [e for e in entries
               if not re.fullmatch(r"(?i)\s*(experience|education|professional experience"
                                   r"|work(ing)? experience|key projects)\s*",
                                   str(e["employer_raw"]["value"] or ""))]
    return {"employment": entries[:12], "suspicious_content": []}


# --------------------------------------------------------------------- profile
def profile_extract(text: str, sections: dict | None = None) -> dict:
    """Lexical-only classification. Deliberately thin: `agents/classification.py`
    already runs the same trigger matching and treats this as the second opinion."""
    def _agg(finder, table, key):
        seen: dict[str, dict] = {}
        for label, surface, s, e in finder(text):
            d = seen.setdefault(label, {"label": label, "confidence": 0.0,
                                        "rationale": "lexical trigger", "quote": surface,
                                        "low_support": False, "_n": 0})
            d["_n"] += 1
        out = []
        for d in seen.values():
            n = d.pop("_n")
            d["confidence"] = round(min(0.8, 0.35 + 0.15 * n), 3)
            d["low_support"] = n < 2
            d["rationale"] = f"{n} lexical trigger(s) in the document"
            out.append(d)
        return sorted(out, key=lambda d: -d["confidence"])

    skills = []
    for canon, surface, _s, _e in tx.find_skills(text):
        if not any(x["name"] == canon for x in skills):
            skills.append({"name": canon, "depth": "mentioned", "quote": surface})

    geo = None
    hits = tx.match_geography(text)
    if hits:
        geo = {"value": hits[0][0], "quote": hits[0][2]}

    return {
        "strategies": _agg(tx.find_strategies, tx.STRATEGIES, "strategy"),
        "sectors": _agg(tx.find_sectors, tx.SECTORS, "sector"),
        "quant_fundamental": {"label": None, "confidence": 0.0, "rationale": "", "quote": ""},
        "feeder_path": {"label": None, "confidence": 0.0, "rationale": "", "quote": ""},
        "geography_primary": geo or {"value": None, "quote": ""},
        "skills": skills,
        "suspicious_content": [],
    }
