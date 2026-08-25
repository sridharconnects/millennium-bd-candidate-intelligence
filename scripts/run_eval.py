#!/usr/bin/env python3
"""Evaluation harness: extraction accuracy, retrieval ablation, fairness audit.

Everything here is measured against artefacts that were labelled by hand BEFORE the
pipeline was tuned against them (data/gold/). The numbers this writes are what the
System page renders -- there is no path where the app displays a metric that was not
computed here from real output.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from millennium import app_data, taxonomy as tx            # noqa: E402
from millennium.config import SETTINGS, ScoreWeights       # noqa: E402
from millennium.index import HashingEmbedder, build_embedder, build_index  # noqa: E402
from millennium.retrieval import parse_query_rules, retrieve  # noqa: E402
from millennium.scoring import rank                        # noqa: E402

GOLD = json.loads((SETTINGS.paths.gold / "gold_labels.json").read_text())["candidates"]
QUERIES = json.loads((SETTINGS.paths.gold / "retrieval_queries.json").read_text())["queries"]


# ------------------------------------------------------------------ extraction
def _norm(s) -> str:
    return tx.norm(str(s or "")).replace(".", "").replace(",", "").strip()


def _prf(pred: set, gold: set) -> tuple[float, float, float]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _scalar_match(pred, gold) -> bool:
    if gold is None:
        return pred is None or pred == ""
    if pred is None or pred == "":
        return False
    return _norm(pred) == _norm(gold) or _norm(gold) in _norm(pred) or _norm(pred) in _norm(gold)


def eval_extraction(profiles) -> dict:
    by_file = {p.provenance.source_file: p for p in profiles if p.provenance}
    scalar_fields = ["full_name", "email", "phone", "country", "region",
                     "current_employer", "current_title"]
    set_fields = {"employers": lambda p: {_norm(e) for e in p.scorable().employers_canonical},
                  "degree_levels": lambda p: {e.degree_level for e in p.education if e.degree_level},
                  "certifications": lambda p: {c.canonical for c in p.certifications if c.canonical},
                  "languages": lambda p: {_norm(l.language) for l in p.languages},
                  "skills_expected": lambda p: {s.canonical for s in p.skills},
                  "sectors_expected": lambda p: {c.label for c in p.sectors}}

    per_field: dict[str, dict] = {}
    hallucinations: list[dict] = []
    abstentions = tot_fields = evid_ok = evid_tot = 0
    schema_ok = 0
    rule_vs_llm: dict[str, dict] = {}
    years_err: list[float] = []

    for fname, gold in GOLD.items():
        p = by_file.get(fname)
        if p is None:
            continue
        schema_ok += 1

        for f in scalar_fields:
            g = gold.get(f)
            pred = _pred_scalar(p, f)
            d = per_field.setdefault(f, {"field": f, "n": 0, "correct": 0,
                                         "abstained": 0, "wrong": 0})
            d["n"] += 1
            tot_fields += 1
            t = _tracked_for(p, f)
            if t is not None and not t.is_known and t.validation_status == "abstained":
                abstentions += 1
                d["abstained"] += 1
                # Abstaining where the gold value is genuinely null is CORRECT.
                if g is None:
                    d["correct"] += 1
                continue
            if _scalar_match(pred, g):
                d["correct"] += 1
            else:
                d["wrong"] += 1
                # A hallucination is a confident value where the truth is 'nothing'.
                if g is None and pred:
                    hallucinations.append({"file": fname, "field": f, "predicted": pred,
                                           "gold": None})

        for f, getter in set_fields.items():
            g = {_norm(x) if f in ("employers", "languages") else x
                 for x in (gold.get(f) or [])}
            pred = getter(p)
            if f == "employers":
                pred = {_norm(x) for x in pred}
            pr, rc, f1 = _prf(pred, g)
            d = per_field.setdefault(f, {"field": f, "n": 0, "p": [], "r": [], "f1": []})
            d["n"] += 1
            d["p"].append(pr)
            d["r"].append(rc)
            d["f1"].append(f1)
            # Any predicted employer that the gold set explicitly forbids is a
            # hard attribution error, which is the failure mode that matters most.
            if f == "employers":
                for bad in gold.get("must_not_contain", []):
                    if _norm(bad) in pred:
                        hallucinations.append({"file": fname, "field": "employers",
                                               "predicted": bad,
                                               "gold": "TRAP: named in a bullet, not an employer"})

        gy = gold.get("years_experience")
        if gy is not None and p.years_experience.is_known:
            years_err.append(abs(float(p.years_experience.value) - float(gy)))

        for t in _all_tracked(p):
            if t.is_known:
                evid_tot += 1
                evid_ok += int(bool(t.evidence))
            m = t.extraction_method
            if m in ("rule", "llm", "hybrid"):
                rv = rule_vs_llm.setdefault(m, {"source": m, "fields": 0, "with_evidence": 0,
                                                "mean_confidence": []})
                rv["fields"] += 1
                rv["with_evidence"] += int(bool(t.evidence))
                rv["mean_confidence"].append(t.confidence)

    rows = []
    for f, d in per_field.items():
        if "f1" in d:
            rows.append({"field": f, "n": d["n"],
                         "precision": round(statistics.mean(d["p"]), 3),
                         "recall": round(statistics.mean(d["r"]), 3),
                         "f1": round(statistics.mean(d["f1"]), 3),
                         "exact_match": "—"})
        else:
            acc = d["correct"] / max(1, d["n"])
            rows.append({"field": f, "n": d["n"], "precision": round(acc, 3),
                         "recall": round(acc, 3), "f1": round(acc, 3),
                         "exact_match": f"{d['correct']}/{d['n']} "
                                        f"({d['abstained']} abstained)"})
    rows.sort(key=lambda r: r["field"])

    for rv in rule_vs_llm.values():
        rv["mean_confidence"] = round(statistics.mean(rv["mean_confidence"]), 3)
        rv["evidence_coverage"] = round(rv["with_evidence"] / max(1, rv["fields"]), 3)

    return {
        "n_documents": len(by_file),
        "schema_validity": round(schema_ok / max(1, len(GOLD)), 3),
        "macro_f1": round(statistics.mean([r["f1"] for r in rows]), 4),
        "hallucination_rate": round(len(hallucinations) / max(1, tot_fields), 4),
        "hallucinations": hallucinations,
        "abstention_rate": round(abstentions / max(1, tot_fields), 4),
        "evidence_coverage": round(evid_ok / max(1, evid_tot), 4),
        "years_experience_mae": round(statistics.mean(years_err), 2) if years_err else None,
        "years_experience_n": len(years_err),
        "per_field": rows,
        "rule_vs_llm": list(rule_vs_llm.values()),
    }


def _tracked_for(p, f):
    # NOTE: uses p.current_role(), not employment[0]. A real hallucination on Viktor
    # Sharat's CV (every tenure stated as a bare duration, no absolute dates) traced
    # back to exactly this fallback picking an arbitrary role and presenting it as
    # "current" -- see the docstring on CandidateProfile.current_role for the story.
    cur = p.current_role()
    return {"full_name": p.sensitive.full_name, "email": p.sensitive.email,
            "phone": p.sensitive.phone,
            "current_employer": cur.employer_raw if cur else None,
            "current_title": cur.title_raw if cur else None,
            }.get(f)


def _pred_scalar(p, f):
    if f == "full_name":
        return p.sensitive.full_name.value
    if f == "email":
        return p.sensitive.email.value
    if f == "phone":
        return p.sensitive.phone.value
    if f == "country":
        return p.geography.label if p.geography else None
    if f == "region":
        return p.geo_region.label if p.geo_region else None
    if f == "current_employer":
        cur = p.current_role()
        return (cur.employer_canonical or cur.employer_raw.value) if cur else None
    if f == "current_title":
        cur = p.current_role()
        return cur.title_raw.value if cur else None
    return None


def _all_tracked(obj):
    from pydantic import BaseModel

    from millennium.schema import Tracked
    if isinstance(obj, Tracked):
        yield obj
        return
    if isinstance(obj, BaseModel):
        for v in obj.__dict__.values():
            yield from _all_tracked(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _all_tracked(v)


# ------------------------------------------------------------------- retrieval
def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def eval_retrieval(profiles) -> tuple[list[dict], list[dict]]:
    file_to_id = {p.provenance.source_file: p.candidate_id for p in profiles if p.provenance}
    real_embedder = build_embedder()
    configs = [
        ("lexical (BM25/FTS5)", real_embedder, "lexical"),
        ("dense (bge-small)", real_embedder, "dense"),
        ("hybrid RRF", real_embedder, "hybrid"),
        ("hybrid RRF · hashing embedder", HashingEmbedder(), "hybrid"),
    ]
    table, per_query = [], []
    for name, emb, mode in configs:
        idx = build_index(profiles, emb)
        metrics = {"ndcg@10": [], "recall@5": [], "recall@10": [], "precision@5": [],
                   "mrr": [], "latency_ms": []}
        for q in QUERIES:
            gold = {file_to_id[f]: g for f, g in q["relevance"].items() if f in file_to_id}
            if not gold:
                continue
            t0 = time.perf_counter()
            hits = retrieve(idx, q["text"], mode, top_k=10).output or []
            lat = (time.perf_counter() - t0) * 1000
            ids = [h.candidate_id for h in hits]
            gains = [gold.get(i, 0) for i in ids[:10]]
            ideal = sorted(gold.values(), reverse=True)[:10]
            nd = dcg(gains) / dcg(ideal) if ideal else 0.0
            rel = {i for i, g in gold.items() if g >= 2}
            r5 = len(set(ids[:5]) & rel) / max(1, len(rel))
            r10 = len(set(ids[:10]) & rel) / max(1, len(rel))
            p5 = len(set(ids[:5]) & rel) / 5
            rr = next((1 / (i + 1) for i, x in enumerate(ids) if x in rel), 0.0)
            for k, v in (("ndcg@10", nd), ("recall@5", r5), ("recall@10", r10),
                         ("precision@5", p5), ("mrr", rr), ("latency_ms", lat)):
                metrics[k].append(v)
            if mode == "hybrid" and "hashing" not in name:
                per_query.append({"id": q["id"], "kind": q["kind"], "query": q["text"],
                                  "ndcg@10": round(nd, 3), "recall@5": round(r5, 3),
                                  "mrr": round(rr, 3), "n_relevant": len(rel)})
        table.append({"mode": name,
                      **{k: round(statistics.mean(v), 4) for k, v in metrics.items()}})
    return table, per_query



# ----------------------------------------------------------------- calibration
def eval_calibration(profiles, n_bins: int = 5) -> dict:
    """Is the confidence number honest?

    A confidence score that does not track observed accuracy is worse than no score:
    it invites a recruiter to trust a field precisely when they should not. So we
    collect every field where gold gives a verdict, bucket by predicted confidence,
    and compare predicted against observed.

    Reported: a reliability curve, Expected Calibration Error (the bucket-size-weighted
    mean gap between confidence and accuracy) and the Brier score (mean squared error
    of the probability itself). Both are standard, and both are things almost no
    resume tool publishes.
    """
    by_file = {p.provenance.source_file: p for p in profiles if p.provenance}
    samples: list[tuple[float, int, str]] = []   # (confidence, correct, field)

    for fname, gold in GOLD.items():
        p = by_file.get(fname)
        if p is None:
            continue
        checks = [("full_name", p.sensitive.full_name, gold.get("full_name")),
                  ("email", p.sensitive.email, gold.get("email")),
                  ("phone", p.sensitive.phone, gold.get("phone"))]
        cur = p.current_role()
        if cur:
            checks += [("current_employer", cur.employer_raw, gold.get("current_employer")),
                       ("current_title", cur.title_raw, gold.get("current_title"))]
        gold_insts = {_norm(i) for i in (gold.get("institutions") or [])}
        for e in p.education[:6]:
            if e.institution.is_known:
                checks.append(("institution", e.institution,
                               _best(gold_insts, _norm(e.institution.value))))
        gold_emps = {_norm(x) for x in (gold.get("employers") or [])}
        for e in p.employment[:8]:
            if e.employer_raw.is_known:
                cand = _norm(e.employer_canonical or e.employer_raw.value)
                checks.append(("employer", e.employer_raw, _best(gold_emps, cand)))

        for field, t, g in checks:
            if not t.is_known:
                continue           # abstentions have no confidence to calibrate
            samples.append((t.confidence, int(_scalar_match(t.value, g)
                                              or _scalar_match(
                                                  getattr(t, "normalized_value", None), g)),
                            field))

    if not samples:
        return {"note": "no calibratable samples"}

    lo = min(c for c, _o, _f in samples)
    hi = max(c for c, _o, _f in samples)
    edges = [lo + (hi - lo) * i / n_bins for i in range(n_bins + 1)]
    edges[-1] += 1e-9
    curve, ece, n = [], 0.0, len(samples)
    for a, b in zip(edges, edges[1:]):
        bucket = [(c, o) for c, o, _f in samples if a <= c < b]
        if not bucket:
            continue
        mean_conf = statistics.mean(c for c, _o in bucket)
        acc = statistics.mean(o for _c, o in bucket)
        curve.append({"bucket": f"{a:.2f}–{b:.2f}", "n": len(bucket),
                      "mean_confidence": round(mean_conf, 4),
                      "observed_accuracy": round(acc, 4),
                      "gap": round(acc - mean_conf, 4)})
        ece += (len(bucket) / n) * abs(acc - mean_conf)

    brier = statistics.mean((c - o) ** 2 for c, o, _f in samples)
    per_field = {}
    for _c, _o, f in samples:
        per_field.setdefault(f, {"field": f, "n": 0, "correct": 0, "conf": []})
    for c, o, f in samples:
        per_field[f]["n"] += 1
        per_field[f]["correct"] += o
        per_field[f]["conf"].append(c)
    for d in per_field.values():
        d["accuracy"] = round(d["correct"] / d["n"], 3)
        d["mean_confidence"] = round(statistics.mean(d.pop("conf")), 3)

    over = [c for c in curve if c["gap"] < -0.05]
    verdict = ("well calibrated" if ece < 0.10 else
               "over-confident" if over else "mis-calibrated")
    return {"n_samples": n, "n_bins": len([c for c in curve]),
            "ece": round(ece, 4), "brier": round(brier, 4),
            "verdict": verdict, "reliability_curve": curve,
            "per_field": sorted(per_field.values(), key=lambda d: d["field"]),
            "interpretation": (
                f"Across {n} grounded field predictions, mean |confidence − accuracy| "
                f"is {ece:.3f} (ECE). A value below 0.10 means the confidence number "
                f"can be read as a probability rather than a vibe.")}


def _best(gold_set: set, candidate: str):
    """Return the gold string this prediction was plausibly aiming at, else None.

    Calibration asks 'was this particular value right', so a prediction needs the gold
    item it corresponds to, not the whole set.
    """
    for g in gold_set:
        if g and (g in candidate or candidate in g):
            return g
    # No gold item corresponds to this prediction, so it is scored as incorrect --
    # which is the right verdict for an employer the CV never listed.
    return None


# -------------------------------------------------------------------- fairness
SWAP_NAMES = ["Aisha Okonkwo", "Bjorn Lindqvist", "Wei Zhang", "Maria Gonzalez",
              "Rajesh Iyer", "Thomas O'Brien"]


def eval_fairness(profiles) -> dict:
    """Counterfactual name-swap audit.

    Mean rank change is expected to be exactly zero, and the reason is structural
    rather than statistical: the scorer's parameter type is `ScorableProfile`, which
    has no name field, so the swapped name is not merely ignored -- it is not present
    in the object the scorer receives. This test verifies that the architecture does
    what the architecture claims.
    """
    pq = parse_query_rules("healthcare equity long/short analyst with 5+ years").output
    weights = ScoreWeights()
    base = [r.candidate_id for r in rank(profiles, pq, weights).output["ranked"]]
    shifts: list[int] = []
    for swap in SWAP_NAMES:
        for p in profiles:
            p.sensitive.full_name.value = swap
        order = [r.candidate_id for r in rank(profiles, pq, weights).output["ranked"]]
        shifts += [abs(order.index(c) - i) for i, c in enumerate(base) if c in order]

    from millennium.schema import ScorableProfile, SensitiveAttributes
    leaked = set(SensitiveAttributes.model_fields) & set(ScorableProfile.model_fields)
    return {
        "names_tested": len(SWAP_NAMES),
        "mean_abs_rank_change": round(statistics.mean(shifts), 4) if shifts else 0.0,
        "max_abs_rank_change": max(shifts) if shifts else 0,
        "protected_fields_reachable_by_scorer": sorted(leaked),
        "mechanism": "structural, not statistical: the scoring function accepts only "
                     "ScorableProfile, which has no field capable of carrying a "
                     "protected attribute. The name never reaches the scorer.",
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default=None,
                    help="path to a candidates.json (default: the LLM-path export)")
    ap.add_argument("--label", default=None,
                    help="label written into the artefact, e.g. 'rule-baseline'")
    ap.add_argument("--out", default=None, help="output filename under data/artifacts/")
    args = ap.parse_args()

    src = Path(args.src) if args.src else None
    profiles, _ = app_data.load_profiles_from_artifact(src)
    app_data.load_raw_texts(profiles)
    if not profiles:
        print("No parsed profiles. Run scripts/run_pipeline.py first.", file=sys.stderr)
        return 2
    real = [p for p in profiles if not (p.provenance and p.provenance.is_synthetic)]
    paths_used = {pr.provenance.extractor for pr in real if pr.provenance}
    print(f"Evaluating {len(real)} real candidate(s)")
    print(f"extraction path(s): {sorted(paths_used)}\n")

    print("── extraction vs hand-labelled gold ──")
    ex = eval_extraction(real)
    print(f"  macro F1            {ex['macro_f1']:.3f}")
    print(f"  hallucination rate  {ex['hallucination_rate']:.3%}  ({len(ex['hallucinations'])} instance(s))")
    print(f"  abstention rate     {ex['abstention_rate']:.3%}")
    print(f"  evidence coverage   {ex['evidence_coverage']:.3%}")
    print(f"  years-exp MAE       {ex['years_experience_mae']} y  (n={ex['years_experience_n']})")
    for r in ex["per_field"]:
        print(f"    {r['field']:20s} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f}  {r['exact_match']}")
    for h in ex["hallucinations"]:
        print(f"    ✗ HALLUCINATION {h['file'][:28]:28s} {h['field']}: {h['predicted']!r}")

    print("\n── retrieval ablation ──")
    ablation, per_query = eval_retrieval(real)
    hdr = f"  {'mode':32s} {'nDCG@10':>8s} {'R@5':>7s} {'R@10':>7s} {'P@5':>7s} {'MRR':>7s} {'ms':>7s}"
    print(hdr)
    for r in ablation:
        print(f"  {r['mode']:32s} {r['ndcg@10']:>8.4f} {r['recall@5']:>7.3f} "
              f"{r['recall@10']:>7.3f} {r['precision@5']:>7.3f} {r['mrr']:>7.3f} "
              f"{r['latency_ms']:>7.2f}")

    print("\n── calibration ──")
    cal = eval_calibration(real)
    if cal.get("reliability_curve"):
        print(f"  ECE {cal['ece']:.4f} · Brier {cal['brier']:.4f} · {cal['verdict']} "
              f"(n={cal['n_samples']})")
        print(f"  {'bucket':14s} {'n':>4s} {'confidence':>11s} {'accuracy':>9s} {'gap':>7s}")
        for c in cal["reliability_curve"]:
            print(f"  {c['bucket']:14s} {c['n']:>4d} {c['mean_confidence']:>11.3f} "
                  f"{c['observed_accuracy']:>9.3f} {c['gap']:>+7.3f}")
    else:
        print("  " + cal.get("note", "unavailable"))

    print("\n── fairness ──")
    fa = eval_fairness(real)
    print(f"  mean |rank change| under name swap: {fa['mean_abs_rank_change']}")
    print(f"  protected fields reachable by scorer: {fa['protected_fields_reachable_by_scorer'] or 'none'}")

    label = args.label or ("llm" if src is None else Path(src).parent.name)
    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "extraction_path": label,
           "note": (None if label == "llm" else
                    f"'{label}' is NOT the case study's required LLM-via-API path. "
                    f"It is published as the rule-vs-LLM baseline."),
           "extraction": ex, "ablation": ablation, "per_query": per_query,
           "calibration": cal, "fairness": fa}
    p = SETTINGS.paths.artifacts / (args.out or "evaluation.json")
    p.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
