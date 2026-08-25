"""Verification: turn model claims into either grounded facts or abstentions.

The pipeline's central guarantee is implemented here. `verify_span` takes the quote
the model supplied for a field and tries to locate it in the raw document text by
three progressively looser strategies. If none succeeds, the caller discards the
value and marks the field `abstained`.

Why three strategies rather than exact-only: models reliably reproduce content but
unreliably reproduce whitespace, ligatures, and smart quotes -- especially in text we
ourselves repaired during ingestion. Exact-only would abstain on correct answers.
Fuzzy-only would accept paraphrase, which defeats the point. The ladder accepts real
quotes and rejects invented ones, and every Evidence records which rung it landed on
so a reviewer can see how strong the match was.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date

from rapidfuzz import fuzz

from .schema import Evidence

# --------------------------------------------------------------------- normalise
_PUNCT = dict.fromkeys(map(ord, "'‘’\"“”`–—―"), None)


def _fold(s: str) -> tuple[str, list[int]]:
    """Lowercase, strip accents/punctuation, collapse whitespace.

    Returns the folded string plus an index map so a match found in folded space can
    be reported as offsets into the ORIGINAL text -- evidence must point at real
    characters a reviewer can highlight, not at a normalised shadow copy.
    """
    out: list[str] = []
    idx: list[int] = []
    prev_space = True
    for i, ch in enumerate(s):
        d = unicodedata.normalize("NFKD", ch)
        d = "".join(c for c in d if not unicodedata.combining(c))
        d = d.translate(_PUNCT)
        if not d:
            continue
        for c in d.lower():
            if c.isspace():
                if prev_space:
                    continue
                out.append(" ")
                idx.append(i)
                prev_space = True
            else:
                out.append(c)
                idx.append(i)
                prev_space = False
    return "".join(out), idx


def verify_span(quote: str, text: str, doc_id: str, page: int | None = None,
                threshold: float = 0.92) -> Evidence | None:
    """Locate `quote` in `text`. Returns None when the quote cannot be grounded.

    None is the signal to abstain. It is deliberately the only failure mode: there is
    no 'trust it anyway' path, because that path is exactly how a fabricated employer
    reaches a recruiter's screen.
    """
    if not quote or not text:
        return None
    q = quote.strip()
    if len(q) < 3:
        return None

    # Rung 1: exact.
    i = text.find(q)
    if i != -1:
        return Evidence(doc_id=doc_id, page=page, char_start=i, char_end=i + len(q),
                        snippet=text[i:i + len(q)], match_kind="exact", match_score=1.0)

    ftext, fmap = _fold(text)
    fq, _ = _fold(q)
    if len(fq) < 3:
        return None

    # Rung 2: normalised exact (whitespace / ligature / smart-quote insensitive).
    j = ftext.find(fq)
    if j != -1:
        s, e = fmap[j], fmap[min(j + len(fq) - 1, len(fmap) - 1)] + 1
        return Evidence(doc_id=doc_id, page=page, char_start=s, char_end=e,
                        snippet=text[s:e], match_kind="normalized", match_score=1.0)

    # Rung 3: fuzzy, with the alignment window so we can still emit real offsets.
    al = fuzz.partial_ratio_alignment(fq, ftext, score_cutoff=threshold * 100)
    if al is None:
        return None
    score = fuzz.partial_ratio(fq, ftext) / 100.0
    if score < threshold:
        return None
    ds, de = al.dest_start, al.dest_end
    if de <= ds or ds >= len(fmap):
        return None
    s, e = fmap[ds], fmap[min(de - 1, len(fmap) - 1)] + 1
    return Evidence(doc_id=doc_id, page=page, char_start=s, char_end=e,
                    snippet=text[s:e], match_kind="fuzzy", match_score=round(score, 4))


# ------------------------------------------------------------------------ dates
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MONTHS.update({"sept": 9, "june": 6, "july": 7, "mayr": 5})  # 'Mayr'23' appears in the corpus
_PRESENT = re.compile(r"\b(present|current|till date|till now|now|ongoing|today)\b", re.I)


def parse_date(raw: str | None) -> tuple[str | None, str]:
    """Loose date text -> ('YYYY-MM' | 'YYYY' | 'present' | None, note)."""
    if raw is None:
        return None, "empty"
    s = str(raw).strip()
    if not s:
        return None, "empty"
    if _PRESENT.search(s):
        return "present", "present marker"
    s2 = s.replace("’", "'").replace("‘", "'")

    m = re.search(r"\b(19|20)\d{2}-(0[1-9]|1[0-2])\b", s2)
    if m:
        return m.group(0), "iso"

    m = re.search(r"([A-Za-z]{3,9})[\s.\-']*'?(\d{2,4})", s2)
    if m:
        mon = _MONTHS.get(m.group(1)[:4].lower()) or _MONTHS.get(m.group(1)[:3].lower())
        if mon:
            yr = int(m.group(2))
            yr += 2000 if yr < 50 else (1900 if yr < 100 else 0)
            if 1950 <= yr <= date.today().year + 2:
                return f"{yr:04d}-{mon:02d}", "month-year"

    m = re.search(r"\b(0?[1-9]|1[0-2])/((19|20)\d{2})\b", s2)
    if m:
        return f"{int(m.group(2)):04d}-{int(m.group(1)):02d}", "numeric"

    m = re.search(r"\b((19|20)\d{2})\b", s2)
    if m:
        return m.group(1), "year-only"
    return None, f"unparseable: {s[:40]!r}"


def parse_duration(text: str | None) -> int | None:
    """'8 years 10 months' / '2 months' / '10 months' -> months. None if absent.

    Needed because several entries in the corpus give a tenure with no dates at all;
    the alternative is to invent dates, which we never do.
    """
    if not text:
        return None
    t = str(text).lower()
    y = re.search(r"(\d+(?:\.\d+)?)\s*(?:year|yr)", t)
    mo = re.search(r"(\d+)\s*(?:month|mo\b)", t)
    if not y and not mo:
        return None
    return int(round((float(y.group(1)) * 12 if y else 0) + (int(mo.group(1)) if mo else 0)))


def _to_months(d: str | None, today: date | None = None) -> int | None:
    if not d:
        return None
    today = today or date.today()
    if d == "present":
        return today.year * 12 + today.month
    if re.fullmatch(r"\d{4}", d):
        return int(d) * 12 + 6            # mid-year when only a year is known
    m = re.fullmatch(r"(\d{4})-(\d{2})", d)
    return int(m.group(1)) * 12 + int(m.group(2)) if m else None


def months_between(start: str | None, end: str | None) -> int | None:
    a, b = _to_months(start), _to_months(end)
    if a is None or b is None:
        return None
    return max(0, b - a)


# ------------------------------------------------------------------ plausibility
EMAIL_RE = re.compile(r"^[\w.+\-]+@[\w\-]+(\.[\w\-]+)+$")
PHONE_DIGITS = re.compile(r"\d")
ISSN_ISBN = re.compile(r"\b(issn|isbn)\s*:?\s*[\d\-]+", re.I)


def check_email(value: str | None) -> tuple[bool, str]:
    if not value:
        return False, "missing"
    v = value.strip()
    if EMAIL_RE.match(v):
        return True, "well-formed"
    if "@" in v:
        return False, f"malformed address (no valid domain): {v!r}"
    return False, f"not an email address: {v!r}"


def check_phone(value: str | None, context: str = "") -> tuple[bool, str]:
    """Rejects the classic false positive: an ISSN/ISBN that looks like a phone."""
    if not value:
        return False, "missing"
    v = value.strip()
    if ISSN_ISBN.search(context or v):
        return False, "looks like an ISSN/ISBN identifier, not a phone number"
    n = len(PHONE_DIGITS.findall(v))
    if n < 7:
        return False, f"only {n} digits -- too short for a phone number"
    if n > 15:
        return False, f"{n} digits -- exceeds E.164 maximum"
    return True, f"plausible ({n} digits)"


# ------------------------------------------------------------------ consistency
def find_overlaps(spans: list[tuple[str, str | None, str | None, bool]]) -> list[dict]:
    """Concurrent non-internship roles. Overlap is common and legitimate (a role change
    at the same firm, a part-time research post during a masters), so this is reported
    as context for a reviewer, never as an error."""
    out = []
    real = [(lbl, s, e) for lbl, s, e, is_intern in spans if not is_intern and s and e]
    for i in range(len(real)):
        for j in range(i + 1, len(real)):
            la, sa, ea = real[i]
            lb, sb, eb = real[j]
            a0, a1 = _to_months(sa), _to_months(ea)
            b0, b1 = _to_months(sb), _to_months(eb)
            if None in (a0, a1, b0, b1):
                continue
            ov = min(a1, b1) - max(a0, b0)
            if ov > 1:
                out.append({"a": la, "b": lb, "overlap_months": ov,
                            "note": "concurrent roles -- verify whether these were "
                                    "simultaneous, sequential, or a title change"})
    return out


def find_gaps(spans: list[tuple[str, str | None, str | None, bool]],
              min_months: int = 6) -> list[dict]:
    """Employment gaps >= min_months, between consecutive dated roles."""
    dated = sorted(
        [(lbl, _to_months(s), _to_months(e)) for lbl, s, e, _ in spans if s and e],
        key=lambda x: (x[1] if x[1] is not None else 0))
    dated = [d for d in dated if d[1] is not None and d[2] is not None]
    out = []
    covered_to = None
    prev_label = None
    for lbl, s, e in dated:
        if covered_to is not None and s - covered_to >= min_months:
            out.append({"after": prev_label, "before": lbl, "months": s - covered_to,
                        "from": f"{covered_to // 12}-{covered_to % 12 or 12:02d}",
                        "to": f"{s // 12}-{s % 12 or 12:02d}"})
        if covered_to is None or e > covered_to:
            covered_to, prev_label = e, lbl
    return out


def total_experience_months(spans: list[tuple[str, str | None, str | None, bool]],
                            include_internships: bool = False) -> tuple[int | None, str]:
    """Union of employment intervals -- NOT a sum, so concurrent roles are not
    double-counted. Returns None when nothing is dated (the model is never asked to
    estimate this; an unknown total is reported as unknown)."""
    ivs = []
    for lbl, s, e, is_intern in spans:
        if is_intern and not include_internships:
            continue
        a, b = _to_months(s), _to_months(e)
        if a is None or b is None or b < a:
            continue
        ivs.append((a, b))
    if not ivs:
        return None, "no dated employment entries"
    ivs.sort()
    merged = [list(ivs[0])]
    for a, b in ivs[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    total = sum(b - a for a, b in merged)
    return total, f"union of {len(merged)} non-overlapping interval(s) from {len(ivs)} role(s)"


def detect_contradictions(profile) -> list[str]:
    """Cross-field consistency. Each message names both sides so a reviewer can act."""
    flags: list[str] = []
    grad_years = [e.graduation_year.value for e in profile.education
                  if e.graduation_year.is_known and e.degree_level not in ("secondary",)]
    first_starts = [_to_months(e.dates.start.value) for e in profile.employment
                    if e.dates.start.is_known and not e.is_internship]
    first_starts = [x for x in first_starts if x]

    if grad_years and first_starts:
        first_degree = min(grad_years)
        if min(first_starts) < (first_degree - 1) * 12:
            flags.append(
                f"first non-internship role starts before the earliest degree completed "
                f"({first_degree}) -- verify whether the role predates study or the "
                f"degree year is wrong")

    if profile.years_experience.is_known and grad_years:
        yrs_since = date.today().year - min(grad_years)
        if profile.years_experience.value > yrs_since + 2:
            flags.append(
                f"derived experience ({profile.years_experience.value:.1f}y) exceeds time "
                f"since first degree ({yrs_since}y)")

    for e in profile.employment:
        s, en = e.dates.start.value, e.dates.end.value
        a, b = _to_months(s), _to_months(en)
        if a and b and b < a:
            flags.append(f"{e.employer_raw.display()}: end date precedes start date ({s} -> {en})")
        if a and b and (b - a) > 45 * 12:
            flags.append(f"{e.employer_raw.display()}: implausible tenure of {(b - a) // 12} years")

    seen: set[tuple] = set()
    for e in profile.employment:
        k = (str(e.employer_canonical).lower(), str(e.title_normalized).lower(),
             e.dates.start.value)
        if k in seen and any(k):
            flags.append(f"duplicate employment entry: {e.employer_raw.display()} / {e.title_raw.display()}")
        seen.add(k)

    edu_seen: set[tuple] = set()
    for e in profile.education:
        k = (str(e.institution.value).lower(), str(e.degree_raw.value).lower(),
             e.graduation_year.value)
        if k in edu_seen and any(k):
            flags.append(f"duplicate education entry: {e.institution.display()} — {e.degree_raw.display()}")
        edu_seen.add(k)

    return flags
