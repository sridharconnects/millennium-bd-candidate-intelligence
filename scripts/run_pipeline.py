#!/usr/bin/env python3
"""Parse every supplied resume with the LLM pipeline and write the export artefacts.

    python scripts/run_pipeline.py              # live parse (needs ANTHROPIC_API_KEY)
    python scripts/run_pipeline.py --demo       # replay the committed cache, no key

Live runs populate data/llm_cache/, which is what makes every later run -- the notebook,
the tests, the deployed app -- deterministic, offline and free.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from millennium.config import SETTINGS          # noqa: E402  (path set above)
from millennium.export import export_all         # noqa: E402
from millennium.llm import LLMClient, LLMUnavailable  # noqa: E402
from millennium.orchestrator import Pipeline     # noqa: E402
from millennium.store import Store               # noqa: E402


def resume_paths() -> list[Path]:
    return sorted(p for p in ROOT.iterdir()
                  if p.suffix.lower() in (".pdf", ".docx") and not p.name.startswith("~$"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="replay from data/llm_cache only; never call the API")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--extractor", choices=["llm", "rules"], default="llm",
                    help="'llm' is the required path. 'rules' runs the deterministic "
                         "baseline with no API key -- used for the rule-vs-LLM "
                         "comparison and for zero-cost CI.")
    args = ap.parse_args()

    demo = args.demo or SETTINGS.flags.demo_mode
    if args.extractor == "rules":
        demo = True                    # no API is involved on this path
    if args.extractor == "llm" and not demo and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.\n"
              "  cp .env.example .env   and put your key in it, then re-run.\n"
              "  Or run with --demo to replay the committed cache.", file=sys.stderr)
        return 2

    paths = resume_paths()
    if args.limit:
        paths = paths[:args.limit]
    if args.extractor == "rules":
        print("!" * 78)
        print("RULE BASELINE RUN — no LLM API will be called.")
        print("This is NOT the case study's required parsing path. It exists to publish")
        print("the rule-vs-LLM comparison and to exercise every downstream stage without")
        print("a key. Artefacts are stamped llm_model=null and parse:rules.")
        print("!" * 78 + "\n")
    print(f"Parsing {len(paths)} document(s)  ·  extractor={args.extractor}  "
          f"·  model={SETTINGS.llm.model if args.extractor == 'llm' else '—'}  "
          f"·  mode={'DEMO/replay' if demo else 'LIVE API'}\n")

    client = LLMClient(demo_mode=demo)
    pipe = Pipeline(client=client, max_workers=args.workers, extractor=args.extractor)

    t0 = time.perf_counter()
    try:
        profiles, results, manifest = pipe.run(
            paths, progress=lambda i, n, f: print(f"  [{i:>2}/{n}] {f}"))
    except LLMUnavailable as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 3

    print("\n" + "=" * 78)
    for line in pipe.log:
        print("  " + line)
    print("=" * 78)

    failed = [r for r in results if r.status == "failed"]
    if failed:
        print(f"\n{len(failed)} document(s) failed:")
        for r in failed:
            print(f"  ✗ {r.source_file}: {r.error}")

    out_dir = (SETTINGS.paths.exports if args.extractor == "llm"
               else SETTINGS.paths.artifacts / "baseline_rules")
    written = export_all(profiles, out_dir=out_dir, manifest=manifest)
    store = Store()
    store.upsert(profiles)

    print(f"\nParsed {len(profiles)}/{len(paths)} in {time.perf_counter() - t0:.1f}s")
    print(f"LLM: {manifest['llm_calls']} calls "
          f"({manifest['llm_cache_hits']} cache hits), "
          f"{manifest['tokens_in']:,} in / {manifest['tokens_out']:,} out, "
          f"${manifest['cost_usd']:.4f} total "
          f"(${manifest['cost_per_doc_usd']:.4f}/resume)")
    print(f"Review queue: {manifest['needs_review']}/{len(profiles)}")
    print("\nExports:")
    for k, v in written.items():
        print(f"  {k:22s} {v.stat().st_size:>8,} bytes")
    print(f"  SQLite                 {store.stats()}")
    print(f"\nManifest: data/artifacts/manifest_{manifest['run_id']}.json")
    print("\nNext:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
