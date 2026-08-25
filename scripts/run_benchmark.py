#!/usr/bin/env python3
"""Scalability benchmark. Deliverable #5, answered with a curve instead of a paragraph.

Measures, at 10 / 50 / 100 / 250 / 500 indexed candidates:
  - index build time and memory footprint
  - p50 / p95 / p99 search latency across the three retrieval modes
  - throughput

The corpus is synthetic and labelled as such. That is fine here and only here: this
measures the *retrieval system's* behaviour as a function of corpus size, which does
not depend on whether the underlying CVs are real. Extraction accuracy is measured
separately, only on the ten real, hand-labelled resumes.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from millennium import app_data                       # noqa: E402
from millennium.config import SETTINGS                # noqa: E402
from millennium.index import build_embedder, build_index  # noqa: E402
from millennium.retrieval import retrieve             # noqa: E402

QUERIES = [
    "healthcare equity long/short analyst in Asia Pacific",
    "quantitative developer C++ derivatives pricing",
    "CFA charterholder credit research high yield",
    "sell-side TMT equity research moving buy-side",
    "systematic factor research python backtesting",
    "global macro rates and FX discretionary trader",
    "private equity growth healthcare due diligence",
    "kdb+ market data infrastructure",
    "consumer sector fundamental analyst Brazil",
    "event driven merger arbitrage special situations",
]

MIGRATION_TRIGGERS = [
    {"trigger": "> ~100k vectors", "symptom": "flat-index scan latency exceeds ~50 ms",
     "action": "move to an ANN index (FAISS IVF-PQ or HNSW), accepting ~1-2% recall loss",
     "why_not_now": "at 500 candidates the exhaustive scan is sub-millisecond; an "
                    "approximate index would trade accuracy for latency we do not need"},
    {"trigger": "concurrent writers", "symptom": "index rebuild blocks ingestion",
     "action": "move to Qdrant or pgvector for transactional upserts and metadata filters",
     "why_not_now": "single-process batch ingestion has no write contention"},
    {"trigger": "> ~1M documents", "symptom": "index no longer fits in a single dyno's RAM",
     "action": "shard by region or tenant; move to a managed store (Pinecone / OpenSearch)",
     "why_not_now": "500 candidates x ~6 chunks x 384 dims x 4 bytes is under 5 MB"},
    {"trigger": "multi-tenant / RBAC", "symptom": "per-desk data isolation required",
     "action": "Postgres + pgvector with row-level security; per-tenant index namespaces",
     "why_not_now": "single BD team, single trust boundary"},
    {"trigger": "> ~1k resumes/day ingest", "symptom": "synchronous parsing blocks the UI",
     "action": "async task queue (Celery/SQS), object storage for documents, idempotent "
               "jobs keyed on file SHA-256, incremental index updates",
     "why_not_now": "the pipeline already memoises on inputs_hash and threads across "
                    "documents; at 10-500 documents a batch run is seconds"},
    {"trigger": "SLA on freshness", "symptom": "nightly rebuild is too slow",
     "action": "incremental add/remove on the live index plus a background compaction",
     "why_not_now": "full rebuild at 500 candidates takes under two seconds"},
]


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = min(len(xs) - 1, int(round((len(xs) - 1) * q)))
    return xs[i]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 50, 100, 250, 500])
    ap.add_argument("--reps", type=int, default=12)
    args = ap.parse_args()

    real, _ = app_data.load_profiles_from_artifact()
    synth = app_data.load_synthetic()
    if not synth:
        print("No synthetic corpus. Run: python scripts/make_synthetic.py -n 500",
              file=sys.stderr)
        return 2
    pool = real + synth
    embedder = build_embedder()
    print(f"embedder: {embedder.name}  ·  pool available: {len(pool)}\n")

    points = []
    for n in args.sizes:
        if n > len(pool):
            print(f"  skipping n={n}: only {len(pool)} profiles available")
            continue
        subset = pool[:n]
        tracemalloc.start()
        t0 = time.perf_counter()
        idx = build_index(subset, embedder)
        build_ms = (time.perf_counter() - t0) * 1000
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        row = {"n_candidates": n, "n_chunks": idx.store.size(),
               "index_build_ms": round(build_ms, 1),
               "peak_mem_mb": round(peak / 1e6, 1)}

        for mode in ("hybrid", "dense", "lexical"):
            lat = []
            for _ in range(args.reps):
                for q in QUERIES:
                    t = time.perf_counter()
                    retrieve.__wrapped__(idx, q, mode) if hasattr(retrieve, "__wrapped__") \
                        else retrieve(idx, q, mode)
                    lat.append((time.perf_counter() - t) * 1000)
            row[f"{mode}_p50_ms"] = round(statistics.median(lat), 2)
            row[f"{mode}_p95_ms"] = round(percentile(lat, 0.95), 2)
            row[f"{mode}_p99_ms"] = round(percentile(lat, 0.99), 2)
            row[f"{mode}_qps"] = round(1000 / max(statistics.mean(lat), 1e-6), 1)

        row["p50_ms"] = row["hybrid_p50_ms"]
        row["p95_ms"] = row["hybrid_p95_ms"]
        points.append(row)
        print(f"  n={n:>4}  chunks={row['n_chunks']:>5}  build={row['index_build_ms']:>8.1f}ms  "
              f"mem={row['peak_mem_mb']:>5.1f}MB  hybrid p50={row['hybrid_p50_ms']:>6.2f}ms  "
              f"p95={row['hybrid_p95_ms']:>6.2f}ms  {row['hybrid_qps']:>6.1f} q/s")

    # Parsing throughput is measured on the REAL corpus only, replayed from cache so
    # the number reflects pipeline overhead rather than API round-trip time.
    parse_stats = {}
    manifest_files = sorted(SETTINGS.paths.artifacts.glob("manifest_*.json"))
    if manifest_files:
        m = json.loads(manifest_files[-1].read_text())
        parse_stats = {
            "documents": m.get("documents"), "elapsed_s": m.get("elapsed_s"),
            "throughput_docs_per_min": m.get("throughput_docs_per_min"),
            "cost_per_doc_usd": m.get("cost_per_doc_usd"),
            "llm_calls": m.get("llm_calls"),
            "note": "measured on the 10 real resumes; 4 worker threads",
        }

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "embedder": embedder.name, "vector_store": "faiss:IndexFlatIP",
        "reps_per_query": args.reps, "n_queries": len(QUERIES),
        "corpus_note": "SYNTHETIC corpus (scripts/make_synthetic.py, seeded) padded "
                       "with the real parsed candidates. Used only to characterise "
                       "retrieval behaviour vs corpus size — never for accuracy.",
        "points": points, "parsing": parse_stats,
        "migration_triggers": MIGRATION_TRIGGERS,
    }
    p = SETTINGS.paths.artifacts / "scalability_benchmark.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {p.relative_to(ROOT)}")
    if len(points) >= 2:
        a, b = points[0], points[-1]
        growth = b["hybrid_p50_ms"] / max(a["hybrid_p50_ms"], 1e-6)
        size = b["n_candidates"] / a["n_candidates"]
        print(f"\n{size:.0f}x the corpus -> {growth:.2f}x the p50 latency "
              f"({a['hybrid_p50_ms']:.2f}ms at {a['n_candidates']} -> "
              f"{b['hybrid_p50_ms']:.2f}ms at {b['n_candidates']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
