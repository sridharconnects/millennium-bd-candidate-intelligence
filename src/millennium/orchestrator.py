"""Pipeline orchestration: an explicit state machine over stages, per-document.

Design choices that matter:

* **Stages are explicit and persisted.** A crashed run resumes from the last completed
  stage rather than restarting. At ten documents that is a convenience; at fifty
  thousand it is the difference between a re-run costing minutes and costing a day.
* **Parallel across documents, sequential within one.** Documents are independent, so
  they thread cleanly. Stages inside a document have hard data dependencies, so
  parallelising them would only add failure modes.
* **Failure is contained per document.** One malformed file produces a degraded
  profile with abstained fields and a recorded error. It never takes down the batch.
* **Every run emits a manifest** -- model, versions, cost, timings, git state -- so any
  artefact in the repo can be traced back to the exact configuration that produced it.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import taxonomy as tx
from .agents import classification as C
from .agents import ingestion as I
from .agents import parsing as P
from .agents import validation as V
from .agents.base import AgentResult, run_subagent
from .config import SETTINGS
from .llm import LLMClient
from .sanitize import redact_pii
from .schema import (CandidateProfile, ProvenanceRecord, QualityReport, SensitiveAttributes,
                     Tracked, stable_id)

STAGES = ["ingest", "sanitize", "parse", "merge", "classify", "validate", "finalize"]


@dataclass
class DocResult:
    source_file: str
    profile: CandidateProfile | None = None
    trace: list[AgentResult] = field(default_factory=list)
    status: str = "ok"
    error: str | None = None
    stage_ms: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0

    def trace_rows(self) -> list[dict]:
        rows = []
        for r in self.trace:
            for f in r.flatten():
                rows.append({"subagent": f.name, "status": f.status,
                             "confidence": round(f.confidence, 3), "ms": f.latency_ms,
                             "cached": f.cached, "cost_usd": round(f.cost_usd or 0.0, 6),
                             "warnings": len(f.warnings), "errors": len(f.errors)})
        return rows


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=3, cwd=SETTINGS.paths.root).stdout.strip() or "no-git"
    except Exception:
        return "no-git"


class Pipeline:
    def __init__(self, client: LLMClient | None = None, run_id: str | None = None,
                 max_workers: int = 4, extractor: str = "llm"):
        """`extractor` selects the parsing path.

        'llm' is the case study's required path and the default. 'rules' runs the
        deterministic baseline in extract_rules.py instead -- used to publish the
        rule-vs-LLM comparison and to exercise every downstream stage in CI without an
        API key. Profiles from the 'rules' path are stamped with a null llm_model and a
        distinct extractor id, so the two are never confused in an artefact.
        """
        if extractor not in ("llm", "rules"):
            raise ValueError(f"extractor must be 'llm' or 'rules', got {extractor!r}")
        self.extractor = extractor
        self.client = client or LLMClient()
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.max_workers = max_workers
        self.log: list[str] = []
        self.state_dir = SETTINGS.paths.artifacts / "state" / self.run_id
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _say(self, msg: str) -> None:
        self.log.append(redact_pii(msg))

    # ------------------------------------------------------------------ per doc
    def process(self, path: Path, corpus: list[tuple[str, str, str]] | None = None,
                is_synthetic: bool = False) -> DocResult:
        path = Path(path)
        res = DocResult(source_file=path.name)
        t_stage = time.perf_counter()

        # ---- stage: ingest -------------------------------------------------
        r_type = run_subagent("ingest.detect_type", path)
        res.trace.append(r_type)
        if not r_type.ok:
            res.status, res.error = "failed", (r_type.errors or ["unsupported"])[0]
            return res

        r_ext = run_subagent("ingest.extract", path, is_synthetic)
        res.trace.append(r_ext)
        doc = r_ext.output
        if doc is None or not doc.text.strip():
            res.status = "failed"
            res.error = "no extractable text (document may need OCR)"
            return res
        res.stage_ms["ingest"] = int((time.perf_counter() - t_stage) * 1000)

        r_lang = run_subagent("ingest.language", doc.text)
        r_qual = run_subagent("ingest.quality", doc)
        r_dup = run_subagent("ingest.near_duplicate", doc, corpus or [])
        res.trace += [r_lang, r_qual, r_dup]

        # ---- stage: sanitize ----------------------------------------------
        t_stage = time.perf_counter()
        r_inj = run_subagent("ingest.injection_scan", doc, path,
                             SETTINGS.flags.enable_injection_scan)
        res.trace.append(r_inj)
        safe_text = (r_inj.output or {}).get("text") or doc.text
        inj_flags = (r_inj.output or {}).get("flags", [])
        res.stage_ms["sanitize"] = int((time.perf_counter() - t_stage) * 1000)

        # ---- stage: parse (LLM) -------------------------------------------
        t_stage = time.perf_counter()
        r_sec = run_subagent("parse.segment_sections", doc.text)
        r_rules = run_subagent("parse.rule_contacts", doc.text, doc.doc_id,
                               doc.header_footer_text)
        # The LLM sees the SANITISED text; span verification runs against the ORIGINAL
        # so that evidence offsets a reviewer clicks still point at the real document.
        if self.extractor == "llm":
            r_id = run_subagent("parse.llm_identity", self.client, safe_text)
            r_emp = run_subagent("parse.llm_employment", self.client, safe_text)
            r_prof = run_subagent("parse.llm_profile", self.client, safe_text)
        else:
            sections = r_sec.output or {}
            r_id = run_subagent("parse.rule_identity", safe_text, sections)
            r_emp = run_subagent("parse.rule_employment", safe_text, sections)
            r_prof = run_subagent("parse.rule_profile", safe_text, sections)
        res.trace += [r_sec, r_rules, r_id, r_emp, r_prof]
        res.stage_ms["parse"] = int((time.perf_counter() - t_stage) * 1000)

        if not any(r.ok for r in (r_id, r_emp, r_prof)):
            res.status = "failed"
            res.error = (r_id.errors or ["LLM parsing unavailable"])[0]
            return res

        # ---- stage: merge + ground ----------------------------------------
        t_stage = time.perf_counter()
        r_mid = run_subagent("parse.merge_identity", r_id.output, r_rules.output,
                             doc.text, doc.doc_id)
        r_memp = run_subagent("parse.merge_employment", r_emp.output, doc.text, doc.doc_id)
        res.trace += [r_mid, r_memp]
        ident = r_mid.output or {}
        employment = r_memp.output or []

        # Conditional adjudication pass -- only if there is a real conflict.
        conflicts = ident.get("conflicts", [])
        if conflicts:
            r_adj = run_subagent("parse.llm_adjudicate", self.client, safe_text, conflicts)
            res.trace.append(r_adj)
            for r in ((r_adj.output or {}).get("resolutions") or []):
                fld = r.get("field")
                if fld in ident and r.get("winner") in ("rule", "llm") and r.get("value"):
                    t: Tracked = ident[fld]
                    t.value = r["value"]
                    t.normalized_value = r["value"]
                    t.validation_status = "verified"
                    t.notes.append(f"adjudicated in favour of {r['winner']}: {r.get('reason','')}")
                elif fld in ident:
                    ident[fld].validation_status = "conflicted"
                    ident[fld].notes.append("rule and model disagree; unresolved -- sent to review")
        res.stage_ms["merge"] = int((time.perf_counter() - t_stage) * 1000)

        # ---- assemble profile ---------------------------------------------
        cid = stable_id(doc.file_sha256, SETTINGS.schema_version)
        prof = CandidateProfile(
            candidate_id=cid, doc_id=doc.doc_id, raw_text=doc.text,
            sections={k: v for k, v in (r_sec.output or {}).items()},
            headline=ident.get("headline") or Tracked.missing(),
            summary=ident.get("summary") or Tracked.missing(),
            location_current=ident.get("location_current") or Tracked.missing(),
            work_authorization=ident.get("work_authorization") or Tracked.missing(),
            employment=employment,
            education=ident.get("education", []),
            certifications=ident.get("certifications", []),
            languages=ident.get("languages", []),
            sensitive=SensitiveAttributes(
                full_name=ident.get("full_name") or Tracked.missing(),
                email=ident.get("email") or Tracked.missing(),
                phone=ident.get("phone") or Tracked.missing(),
                home_address=ident.get("home_address") or Tracked.missing(),
                marital_status=ident.get("marital_status") or Tracked.missing(),
            ),
        )

        # ---- stage: classify ----------------------------------------------
        t_stage = time.perf_counter()
        pdata = r_prof.output or {}
        r_sk = run_subagent("classify.skills", doc.text, doc.doc_id, pdata.get("skills"))
        prof.skills = r_sk.output or []
        r_st = run_subagent("classify.strategy", doc.text, doc.doc_id, pdata.get("strategies"))
        r_se = run_subagent("classify.sector", doc.text, doc.doc_id, pdata.get("sectors"))
        r_geo = run_subagent("classify.geography", doc.text, doc.doc_id, employment,
                             pdata.get("geography_primary"))
        r_qp = run_subagent("classify.quant_profile", doc.text, prof.skills,
                            pdata.get("quant_fundamental"))
        r_fp = run_subagent("classify.feeder_path", doc.text, employment,
                            pdata.get("feeder_path"))
        prof.strategies = r_st.output or []
        prof.sectors = r_se.output or []
        if r_geo.output and r_geo.output[0]:
            prof.geography, prof.geo_region = r_geo.output
        prof.quant_fundamental = r_qp.output
        prof.feeder_path = r_fp.output
        res.trace += [r_sk, r_st, r_se, r_geo, r_qp, r_fp]
        res.stage_ms["classify"] = int((time.perf_counter() - t_stage) * 1000)

        # ---- stage: validate ----------------------------------------------
        t_stage = time.perf_counter()
        r_dates = run_subagent("validate.dates", prof)
        dd = r_dates.output or {}
        prof.years_experience = dd.get("years_experience") or Tracked.missing()
        prof.years_relevant_experience = dd.get("years_relevant") or Tracked.missing()
        prof.current_tenure_months = dd.get("current_tenure") or Tracked.missing()
        prof.employment_gaps = dd.get("gaps", [])

        r_sen = run_subagent("classify.seniority", employment, prof.years_experience.value)
        prof.seniority = r_sen.output

        r_span = run_subagent("validate.spans", prof)
        r_cons = run_subagent("validate.consistency", prof)
        r_comp = run_subagent("validate.completeness", prof)
        q: QualityReport = r_comp.output or QualityReport()
        q.extraction_quality = doc.extraction_quality
        q.validation_flags = list(q.validation_flags) + list(r_cons.output or [])
        prof.quality = q

        # Graceful degradation is correct, but silence is not. A failed subagent in a
        # core stage previously vanished into a default QualityReport, which reported
        # 0% completeness for every candidate and looked like a data problem rather
        # than a crash. Surface it where a human will see it.
        broken = [f.name for r in res.trace for f in r.flatten()
                  if f.status == "failed" and f.name.split(".")[0] in
                  ("parse", "validate", "classify")]
        if broken:
            prof.quality.validation_flags.append(
                f"PIPELINE DEGRADED: subagent(s) failed — {', '.join(sorted(set(broken)))}; "
                f"fields derived from them are unreliable")

        r_rev = run_subagent("validate.route_review", prof, doc.extraction_quality, inj_flags)
        prof.quality.needs_human_review = (r_rev.output or {}).get("needs_review", False)
        prof.quality.review_reasons = (r_rev.output or {}).get("reasons", [])
        if broken:
            prof.quality.needs_human_review = True
            prof.quality.review_reasons.append(
                f"a pipeline stage failed ({len(set(broken))} subagent(s))")
        res.trace += [r_dates, r_sen, r_span, r_cons, r_comp, r_rev]
        res.stage_ms["validate"] = int((time.perf_counter() - t_stage) * 1000)

        # ---- stage: finalize ----------------------------------------------
        cost = sum(f.cost_usd or 0.0 for r in res.trace for f in r.flatten())
        prof.provenance = ProvenanceRecord(
            source_file=doc.source_file, file_sha256=doc.file_sha256,
            text_sha256=doc.text_sha256, file_type=doc.file_type,
            page_count=doc.page_count,
            ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            extractor=(f"millennium.ingest/{'pdf' if doc.file_type=='pdf' else 'docx'}"
                       + (f" + parse:{self.extractor}"
                          if self.extractor != "llm" else " + parse:llm")),
            schema_version=SETTINGS.schema_version, taxonomy_version=tx.TAXONOMY_VERSION,
            pipeline_run_id=self.run_id,
            # Null when no model was called. Provenance never implies an API call that
            # did not happen.
            llm_model=SETTINGS.llm.model if self.extractor == "llm" else None,
            cost_usd=round(cost, 6),
            is_synthetic=is_synthetic, injection_flags=inj_flags,
            near_duplicate_of=[d["label"] for d in (r_dup.output or {}).get("duplicates", [])],
        )
        if (agency := (r_rules.output or {}).get("agency_watermark")):
            prof.provenance.injection_flags = prof.provenance.injection_flags
            prof.quality.validation_flags.append(f"sourced via agency: {agency}")

        res.profile = prof
        res.cost_usd = cost
        res.status = "partial" if (prof.quality.needs_human_review or
                                   any(r.status == "failed" for r in res.trace)) else "ok"
        (self.state_dir / f"{cid}.json").write_text(
            prof.model_dump_json(indent=1), encoding="utf8")
        self._say(f"[{res.status}] {path.name}: {len(prof.employment)} roles, "
                  f"{len(prof.skills)} skills, completeness {prof.quality.completeness:.0%}, "
                  f"{prof.quality.abstention_count} abstentions, ${cost:.4f}")
        return res

    # ------------------------------------------------------------------ batch
    def run(self, paths: list[Path], is_synthetic: bool = False,
            progress=None) -> tuple[list[CandidateProfile], list[DocResult], dict]:
        t0 = time.perf_counter()
        paths = [Path(p) for p in paths]

        # Pre-extract text once so near-duplicate detection sees the whole corpus.
        corpus: list[tuple[str, str, str]] = []
        for p in paths:
            r = run_subagent("ingest.extract", p, is_synthetic)
            if r.output is not None:
                corpus.append((r.output.doc_id, p.name, r.output.text))

        results: list[DocResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self.process, p, corpus, is_synthetic): p for p in paths}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    results.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    results.append(DocResult(source_file=futs[fut].name, status="failed",
                                             error=f"{type(exc).__name__}: {exc}"))
                if progress:
                    progress(i, len(paths), futs[fut].name)

        results.sort(key=lambda r: r.source_file)
        profiles = [r.profile for r in results if r.profile]
        elapsed = time.perf_counter() - t0
        manifest = {
            "run_id": self.run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": _git_sha(),
            "python": platform.python_version(),
            "schema_version": SETTINGS.schema_version,
            "taxonomy_version": tx.TAXONOMY_VERSION,
            "extractor": self.extractor,
            "llm_provider": SETTINGS.llm.provider if self.extractor == "llm" else None,
            "llm_model": SETTINGS.llm.model if self.extractor == "llm" else None,
            "NOTE": (None if self.extractor == "llm" else
                     "RULE BASELINE RUN — no LLM API was called. This is not the "
                     "case study's required parsing path; it is the published "
                     "rule-vs-LLM baseline and the zero-cost CI path."),
            "demo_mode": self.client.demo_mode,
            "documents": len(paths),
            "succeeded": len(profiles),
            "failed": [{"file": r.source_file, "error": r.error}
                       for r in results if r.status == "failed"],
            "needs_review": sum(1 for p in profiles if p.quality.needs_human_review),
            "elapsed_s": round(elapsed, 2),
            "throughput_docs_per_min": round(len(paths) / max(elapsed, 1e-6) * 60, 1),
            "llm_calls": self.client.usage.calls,
            "llm_cache_hits": self.client.usage.cache_hits,
            "tokens_in": self.client.usage.tokens_in,
            "tokens_out": self.client.usage.tokens_out,
            "cost_usd": round(self.client.usage.cost_usd, 5),
            "cost_per_doc_usd": round(self.client.usage.cost_usd / max(1, len(paths)), 5),
            "per_stage_llm": self.client.usage.per_stage,
            "is_synthetic": is_synthetic,
        }
        (SETTINGS.paths.artifacts / f"manifest_{self.run_id}.json").write_text(
            json.dumps(manifest, indent=2))
        return profiles, results, manifest
