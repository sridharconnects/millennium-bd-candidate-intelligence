"""The data contract.

Design thesis: a recruiting tool is only useful if a recruiter can check its work.
So no scalar is ever stored bare. Every extracted value is wrapped in `Tracked[T]`,
which carries the verbatim source span the value came from, how it was obtained, and
whether that span was independently verified against the raw document text.

If a span cannot be located in the source, the value is *discarded* and the field is
marked `abstained`. In a hiring product a blank is strictly better than a fabricated
employer, so abstention is a success state and is reported as a first-class metric.
"""
from __future__ import annotations

import hashlib
from datetime import date
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

T = TypeVar("T")

ExtractionMethod = Literal["rule", "llm", "hybrid", "human", "derived"]
ValidationStatus = Literal[
    "verified",        # span located in source text
    "unverified",      # value present, span not checked (never shown as fact in UI)
    "abstained",       # model proposed a value, span did not verify -> value dropped
    "conflicted",      # rule and LLM disagreed
    "human_corrected", # a reviewer overrode it
    "derived",         # computed in Python from verified fields
]


class Evidence(BaseModel):
    """A pointer back into the source document. This is the unit of trust."""
    model_config = ConfigDict(frozen=True)

    doc_id: str
    page: int | None = None
    char_start: int
    char_end: int
    snippet: str
    match_kind: Literal["exact", "normalized", "fuzzy"] = "exact"
    match_score: float = 1.0

    def belongs_to(self, doc_id: str) -> bool:
        return self.doc_id == doc_id


class Tracked(BaseModel, Generic[T]):
    """A value plus its provenance. `value` is None whenever status == 'abstained'."""
    value: T | None = None
    normalized_value: T | None = None
    confidence: float = 0.0
    evidence: list[Evidence] = Field(default_factory=list)
    extraction_method: ExtractionMethod = "llm"
    validation_status: ValidationStatus = "unverified"
    notes: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @property
    def is_known(self) -> bool:
        """True only if we have a value we are willing to display as fact.

        Note the distinction the UI relies on: `abstained` (we saw a claim but could
        not prove it) is a different user-facing state from `missing` (the document
        never mentioned it). Both are unknown; only one is interesting.
        """
        return self.value is not None and self.validation_status != "abstained"

    def display(self, fallback: str = "—") -> str:
        if not self.is_known:
            return fallback
        v = self.normalized_value if self.normalized_value is not None else self.value
        return str(v)

    @classmethod
    def abstain(cls, reason: str, method: ExtractionMethod = "llm") -> "Tracked[T]":
        return cls(value=None, confidence=0.0, extraction_method=method,
                   validation_status="abstained", notes=[reason])

    @classmethod
    def missing(cls) -> "Tracked[T]":
        return cls(value=None, confidence=0.0, validation_status="unverified",
                   notes=["not present in document"])

    @classmethod
    def derived(cls, value: T, confidence: float, basis: str) -> "Tracked[T]":
        return cls(value=value, normalized_value=value, confidence=confidence,
                   extraction_method="derived", validation_status="derived",
                   notes=[f"computed from verified fields: {basis}"])


# ----------------------------------------------------------------------------- 
# Sensitive attributes are physically segregated from the scoring surface.
# `CandidateProfile` holds a `sensitive` block; the scorer's signature accepts only
# `ScorableProfile`, which structurally lacks it. The fairness claim is therefore a
# property of the type system rather than a promise in a README.
# -----------------------------------------------------------------------------
class SensitiveAttributes(BaseModel):
    """Never passed to any scoring, ranking, or embedding function. Enforced by
    tests/test_fairness.py, which asserts the scorer cannot even accept this type."""
    full_name: Tracked = Field(default_factory=Tracked)
    email: Tracked = Field(default_factory=Tracked)
    phone: Tracked = Field(default_factory=Tracked)
    home_address: Tracked = Field(default_factory=Tracked)
    date_of_birth: Tracked = Field(default_factory=Tracked)
    gender_markers: list[str] = Field(default_factory=list)
    marital_status: Tracked = Field(default_factory=Tracked)
    nationality: Tracked = Field(default_factory=Tracked)
    photo_present: bool = False
    personal_interests: list[str] = Field(default_factory=list)


class DateRange(BaseModel):
    # NOTE ON TYPING: fields below are declared as the unparametrised `Tracked`, not
    # `Tracked[str]`. Pydantic v2 validates a parametrised generic field strictly --
    # a `Tracked[str]` field rejects both `Tracked[int]` and a bare `Tracked`. Since
    # extraction yields heterogeneous value types at runtime (year -> int, employer ->
    # str, tenure -> int), pinning T per field would force brittle parametrisation at
    # every construction site for no safety gain. The generic parameter remains on the
    # class, where it is useful for annotating helpers, and each field's intended value
    # type is documented in a trailing comment.
    start: Tracked = Field(default_factory=Tracked)   # ISO 'YYYY-MM' or 'YYYY'
    end: Tracked = Field(default_factory=Tracked)
    is_current: bool = False
    duration_months: Tracked = Field(default_factory=Tracked)


class EmploymentEntry(BaseModel):
    employer_raw: Tracked = Field(default_factory=Tracked)
    employer_canonical: str | None = None
    employer_tier: str | None = None          # see taxonomy.FIRM_TIERS
    title_raw: Tracked = Field(default_factory=Tracked)
    title_normalized: str | None = None
    seniority_level: int | None = None        # 1..7, see taxonomy.SENIORITY_LEVELS
    location: Tracked = Field(default_factory=Tracked)
    dates: DateRange = Field(default_factory=DateRange)
    is_internship: bool = False
    # Found by a real hallucination: Ryan Patel co-founded a non-profit ("Global
    # Education Alliance, Co-Founder, Jan 2017-Present") which is structurally
    # identical to a job entry -- a title, dates, an org name -- so the extractor
    # correctly parsed it as one. It should not count as investment-relevant
    # employment (tier, years-of-experience, employer distributions), but it IS real,
    # useful CV content a recruiter would want to see, so it is kept and flagged
    # rather than dropped. Same pattern as is_internship; see current_role()'s
    # docstring and validate.dates() for how it's excluded from totals.
    is_volunteer: bool = False
    highlights: list[Tracked] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)


class EducationEntry(BaseModel):
    institution: Tracked = Field(default_factory=Tracked)
    degree_raw: Tracked = Field(default_factory=Tracked)
    degree_level: str | None = None           # bachelors|masters|mba|phd|professional|secondary
    field_of_study: Tracked = Field(default_factory=Tracked)
    graduation_year: Tracked = Field(default_factory=Tracked)
    gpa_raw: Tracked = Field(default_factory=Tracked)
    location: Tracked = Field(default_factory=Tracked)
    honors: list[str] = Field(default_factory=list)


class SkillEntry(BaseModel):
    canonical: str
    surface_forms: list[str] = Field(default_factory=list)
    category: str = "other"                   # programming|analytics|finance|tools|domain
    depth: Literal["mentioned", "applied", "core"] = "mentioned"
    evidence: list[Evidence] = Field(default_factory=list)


class Certification(BaseModel):
    name: Tracked = Field(default_factory=Tracked)
    canonical: str | None = None
    status: str | None = None                 # charterholder|level_i|level_ii|level_iii|registered
    year: Tracked = Field(default_factory=Tracked)


class LanguageEntry(BaseModel):
    language: str
    proficiency: str | None = None            # native|fluent|professional|conversational|basic
    evidence: list[Evidence] = Field(default_factory=list)


class Classification(BaseModel):
    """Every label carries the rule or exemplar that fired, plus its support."""
    label: str
    confidence: float = 0.0
    rationale: str = ""
    triggers: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    low_support: bool = False


class QualityReport(BaseModel):
    extraction_quality: float = 0.0     # how clean was the text layer
    completeness: float = 0.0           # share of core fields known
    evidence_coverage: float = 0.0      # share of known fields with a verified span
    abstention_count: int = 0
    conflict_count: int = 0
    validation_flags: list[str] = Field(default_factory=list)
    needs_human_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)


class ProvenanceRecord(BaseModel):
    source_file: str
    file_sha256: str
    text_sha256: str
    file_type: str
    page_count: int | None = None
    ingested_at: str = ""
    extractor: str = ""
    schema_version: str = ""
    taxonomy_version: str = ""
    pipeline_run_id: str = ""
    llm_model: str | None = None
    cost_usd: float = 0.0
    is_synthetic: bool = False
    injection_flags: list[str] = Field(default_factory=list)
    near_duplicate_of: list[str] = Field(default_factory=list)


class ScorableProfile(BaseModel):
    """Exactly the surface the matching engine is allowed to see.

    Constructed by `CandidateProfile.scorable()`. It deliberately has no field that
    could carry a protected attribute -- no name, no email, no address, no marital
    status, no nationality, no hobbies.
    """
    candidate_id: str
    years_experience: float | None = None
    seniority_level: int | None = None
    geography: str | None = None
    geo_region: str | None = None
    strategies: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    skills: list[SkillEntry] = Field(default_factory=list)
    degree_levels: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    employer_tiers: list[str] = Field(default_factory=list)
    employers_canonical: list[str] = Field(default_factory=list)
    feeder_path: str | None = None
    quant_fundamental: str | None = None
    data_quality: float = 0.0
    searchable_text: str = ""


class CandidateProfile(BaseModel):
    candidate_id: str
    doc_id: str
    sensitive: SensitiveAttributes = Field(default_factory=SensitiveAttributes)

    headline: Tracked = Field(default_factory=Tracked)
    summary: Tracked = Field(default_factory=Tracked)
    location_current: Tracked = Field(default_factory=Tracked)
    work_authorization: Tracked = Field(default_factory=Tracked)

    employment: list[EmploymentEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    skills: list[SkillEntry] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    languages: list[LanguageEntry] = Field(default_factory=list)

    # Derived in Python from verified fields only -- never asked of the model.
    years_experience: Tracked = Field(default_factory=Tracked)
    years_relevant_experience: Tracked = Field(default_factory=Tracked)
    current_tenure_months: Tracked = Field(default_factory=Tracked)
    employment_gaps: list[dict] = Field(default_factory=list)

    geography: Classification | None = None
    geo_region: Classification | None = None
    seniority: Classification | None = None
    quant_fundamental: Classification | None = None
    feeder_path: Classification | None = None
    strategies: list[Classification] = Field(default_factory=list)
    sectors: list[Classification] = Field(default_factory=list)

    quality: QualityReport = Field(default_factory=QualityReport)
    provenance: ProvenanceRecord | None = None
    raw_text: str = ""
    sections: dict[str, list[int]] = Field(default_factory=dict)  # name -> [start, end]

    # ---------------- derived views ----------------
    def scorable(self) -> ScorableProfile:
        """The only legal input to any scoring function."""
        return ScorableProfile(
            candidate_id=self.candidate_id,
            years_experience=self.years_experience.value,
            seniority_level=_lvl(self.seniority),
            geography=self.geography.label if self.geography else None,
            geo_region=self.geo_region.label if self.geo_region else None,
            strategies=[c.label for c in self.strategies],
            sectors=[c.label for c in self.sectors],
            skills=self.skills,
            degree_levels=[e.degree_level for e in self.education if e.degree_level],
            certifications=[c.canonical for c in self.certifications if c.canonical],
            languages=[l.language for l in self.languages],
            employer_tiers=[e.employer_tier for e in self.employment
                           if e.employer_tier and not e.is_volunteer],
            employers_canonical=[e.employer_canonical for e in self.employment
                                if e.employer_canonical and not e.is_volunteer],
            feeder_path=self.feeder_path.label if self.feeder_path else None,
            quant_fundamental=self.quant_fundamental.label if self.quant_fundamental else None,
            data_quality=self.quality.completeness,
            searchable_text=self.searchable_text(),
        )

    def searchable_text(self) -> str:
        """Non-sensitive text used for embedding and lexical indexing.

        Name and contact details are intentionally excluded so that neither the
        embedding nor the BM25 index can key on a protected attribute.
        """
        bits: list[str] = []
        if self.headline.is_known:
            bits.append(str(self.headline.value))
        if self.summary.is_known:
            bits.append(str(self.summary.value))
        for e in self.employment:
            bits.append(" ".join(filter(None, [
                e.title_raw.value or "", e.employer_raw.value or "",
                e.location.value or "", " ".join(s.value or "" for s in e.highlights)])))
        for e in self.education:
            bits.append(" ".join(filter(None, [
                e.degree_raw.value or "", e.field_of_study.value or "",
                e.institution.value or ""])))
        bits.append(" ".join(s.canonical for s in self.skills))
        bits.append(" ".join(c.canonical or "" for c in self.certifications))
        return "\n".join(b for b in bits if b.strip())

    def current_role(self) -> "EmploymentEntry | None":
        """The best-defensible 'current' role, or None when picking one would be a guess.

        Found by a real hallucination, not a hypothetical: on a live LLM parse of
        Viktor Sharat's CV, three call sites each independently did
        `employment[0] if employment else None` as a fallback when no entry was marked
        `is_current`. Viktor's CV states every tenure as a bare duration ("8 years 10
        months") with zero absolute dates, so `employment[0]` was whatever order the
        model happened to list roles in -- not a sorted "most recent first" -- and the
        fallback silently presented "Axis Mutual Fund, Research Analyst" (a 2-month
        stint) as his current employer. That is precisely the class of confident-but-
        wrong claim this whole system exists to refuse.

        The fix: a non-volunteer entry explicitly marked current wins over a volunteer
        one marked current (a paid job and an ongoing non-profit co-founder role can
        both legitimately say "Present" -- see is_volunteer -- and the paid one is what
        "current employer" means to a recruiter). Failing that, the most recent entry
        by a KNOWN start date is used; and when NO entry has a known date, this returns
        None so the caller must display "unknown" rather than guess.
        """
        current = [e for e in self.employment if e.dates.is_current]
        professional_current = [e for e in current if not e.is_volunteer]
        if professional_current:
            return professional_current[0]
        if current:
            return current[0]
        dated = [e for e in self.employment if not e.is_volunteer
                and (e.dates.start.normalized_value or e.dates.start.value)]
        if not dated:
            return None
        return max(dated, key=lambda e: str(e.dates.start.normalized_value
                                            or e.dates.start.value or "0000"))

    def display_name(self, blind: bool = False) -> str:
        if blind or not self.sensitive.full_name.is_known:
            return f"Candidate {self.candidate_id[:8].upper()}"
        return str(self.sensitive.full_name.value)

    def all_evidence(self) -> list[Evidence]:
        out: list[Evidence] = []

        def walk(obj):
            if isinstance(obj, Evidence):
                out.append(obj)
            elif isinstance(obj, BaseModel):
                for v in obj.__dict__.values():
                    walk(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    walk(v)
        walk(self)
        return out

    def all_tracked(self) -> list[Tracked]:
        """Every `Tracked` field anywhere in the profile, sensitive block included.

        The public counterpart to `all_evidence` -- same generic walk, one level up the
        type hierarchy. Used by completeness scoring (validate.completeness) and by the
        review/candidate UI to answer "what did the pipeline abstain on and why", so
        both stay in sync with the schema by construction rather than by convention.
        """
        out: list[Tracked] = []

        def walk(obj):
            if isinstance(obj, Tracked):
                out.append(obj)
            elif isinstance(obj, BaseModel):
                for v in obj.__dict__.values():
                    walk(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    walk(v)
        walk(self)
        return out


def _lvl(c: Classification | None) -> int | None:
    """Classification labels are produced as f"L{level}" (e.g. "L3"), never
    "L_3" -- splitting on "_" left the leading "L" on the number and made
    `int(...)` raise every time, so `seniority_level` was silently always
    None and the must_have min_seniority gate in retrieval.py never gated."""
    if not c:
        return None
    try:
        return int(c.label[1:]) if c.label.startswith("L") else None
    except ValueError:
        return None


def stable_id(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()[:16]
