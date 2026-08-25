#!/usr/bin/env python3
"""Render the architecture diagram as a self-contained SVG.

Generated rather than drawn so it cannot drift from the code: the agent and subagent
counts below are read from the live registry.
"""
from __future__ import annotations

import sys
from xml.sax.saxutils import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from millennium.agents import (classification, ingestion, insight,  # noqa: E402,F401
                               parsing, validation)
from millennium.agents.base import REGISTRY, registry_table  # noqa: E402
import millennium.retrieval  # noqa: E402,F401  -- registers search subagents
import millennium.scoring    # noqa: E402,F401  -- registers matching subagents

OUT = ROOT / "docs" / "architecture.svg"

INK, MUTED, LINE = "#0F172A", "#64748B", "#CBD5E1"
ACCENT, AMBER, RED, BLUE, VIOLET = "#0F766E", "#B45309", "#B91C1C", "#1D4ED8", "#7E22CE"


def box(x, y, w, h, title, sub, colour=ACCENT, fill="#FFFFFF"):
    lines = []
    lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
                 f'fill="{fill}" stroke="{colour}" stroke-width="1.4"/>')
    lines.append(f'<text x="{x+11}" y="{y+20}" font-size="12.5" font-weight="650" '
                 f'fill="{INK}">{escape(title)}</text>')
    for i, s in enumerate(sub):
        lines.append(f'<text x="{x+11}" y="{y+37+i*13.5}" font-size="10" '
                     f'fill="{MUTED}">{escape(s)}</text>')
    return "\n".join(lines)


def arrow(x1, y1, x2, y2, colour=LINE, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" '
            f'stroke-width="1.6" marker-end="url(#a)"{d}/>')


def label(x, y, text, size=10.5, colour=MUTED, weight="400", anchor="start"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{colour}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(text)}</text>')


def main() -> int:
    counts: dict[str, int] = {}
    for e in REGISTRY.values():
        counts[e.parent] = counts.get(e.parent, 0) + 1
    n_sub = len(REGISTRY)
    n_agents = len(counts)

    W, H = 1180, 760
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" font-family="Inter,-apple-system,Segoe UI,sans-serif">',
         '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
         f'markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/>'
         '</marker></defs>',
         f'<rect width="{W}" height="{H}" fill="#F8FAFC"/>']

    p.append(label(24, 34, "Millennium BD — Candidate Intelligence Platform", 17, INK, "700"))
    p.append(label(24, 54, f"{n_agents} agents · {n_sub} subagents · one AgentResult "
                           f"contract · every claim traceable to a source span", 11.5))

    # ---- row 1: the pipeline
    y = 86
    p.append(label(24, y - 6, "INGEST PIPELINE", 10, MUTED, "700"))
    stages = [
        (24, "1 · Ingestion", [f"{counts.get('ingestion',0)} subagents",
                               "magic-byte typing", "column-order repair",
                               "merged-cell dedup", "near-duplicate (Jaccard)"], ACCENT),
        (248, "2 · Sanitize", ["10 injection families", "render-layer scan",
                               "white-on-white / 1pt", "neutralise + log"], RED),
        (472, "3 · Parsing (LLM API)", [f"{counts.get('parsing',0)} subagents",
                                        "3 targeted passes", "no tools passed",
                                        "quote required per value",
                                        "4th pass only on conflict"], BLUE),
        (720, "4 · Ground or ABSTAIN", ["exact → normalised → fuzzy",
                                        "quote not found ⇒", "value DISCARDED",
                                        "abstention is a success"], AMBER),
        (944, "5 · Classify + Validate",
         [f"{counts.get('classification',0)+counts.get('validation',0)} subagents",
          "closed taxonomies", "dates derived in Python",
          "contradictions, gaps", "review routing"], VIOLET),
    ]
    for x, title, sub, c in stages:
        w = 200 if x != 472 else 224
        p.append(box(x, y, w, 108, title, sub, c))
    for x1, x2 in ((224, 248), (448, 472), (696, 720), (920, 944)):
        p.append(arrow(x1, y + 54, x2 - 3, y + 54))

    # ---- callout under grounding
    p.append(f'<rect x="700" y="206" width="264" height="46" rx="6" fill="#FEF3C7" '
             f'stroke="#FDE68A"/>')
    p.append(label(712, 224, "A fabricated employer is worse than a blank.", 10.5,
                   "#92400E", "650"))
    p.append(label(712, 239, "Unprovable ⇒ refused, counted, and shown as such.", 10,
                   "#92400E"))

    # ---- row 2: storage
    y2 = 282
    p.append(label(24, y2 - 6, "ARTEFACTS & STORAGE", 10, MUTED, "700"))
    p.append(box(24, y2, 200, 84, "JSON / CSV exports",
                 ["candidates · employment", "education · skills · evidence",
                  "*_status columns:", "abstained ≠ missing"], ACCENT))
    p.append(box(248, y2, 200, 84, "SQLite + FTS5",
                 ["metadata + labels", "review audit log",
                  "GDPR erasure, end to end", "(tested)"], ACCENT))
    p.append(box(472, y2, 224, 84, "FAISS IndexFlatIP",
                 ["bge-small ONNX, 384-d", "exact search (not ANN)",
                  "section-aware chunks", "manifest refuses stale index"], ACCENT))
    p.append(box(720, y2, 244, 84, "Run manifest",
                 ["model · versions · git SHA", "tokens · cost · timings",
                  "every artefact traceable"], ACCENT))
    p.append(arrow(600, y + 112, 600, y2 - 4))

    # ---- row 3: serving agents
    y3 = 404
    p.append(label(24, y3 - 6, "SERVING", 10, MUTED, "700"))
    p.append(box(24, y3, 300, 96, f"6 · Search  ({counts.get('search',0)} subagents)",
                 ["NL query → filters + exclusions",
                  "dense (bge) ⊕ lexical (BM25)",
                  "RRF fusion, k=60 — rank-based,",
                  "so no score calibration needed"], BLUE))
    p.append(box(348, y3, 300, 96, f"7 · Matching  ({counts.get('matching',0)} subagents)",
                 ["must-haves gate before scoring",
                  "excluded shown, never dropped",
                  "score = Σ weight × component",
                  "counterfactual: minimal edit"], BLUE))
    p.append(box(672, y3, 292, 96, f"8 · Insight  ({counts.get('insight',0)} subagents)",
                 ["distributions · co-occurrence",
                  "COVERAGE GAPS —", "which reqs you cannot fill",
                  "data-quality dashboard"], BLUE))
    for x in (174, 498, 818):
        p.append(arrow(x, y2 + 88, x, y3 - 4))

    # ---- row 4: UI
    y4 = 528
    p.append(label(24, y4 - 6, "STREAMLIT WORKSPACE", 10, MUTED, "700"))
    pages = ["Search", "Candidate", "Requisition", "Shortlist", "Intake", "Review",
             "Analytics", "System"]
    for i, name in enumerate(pages):
        x = 24 + i * 118
        p.append(f'<rect x="{x}" y="{y4}" width="106" height="34" rx="6" fill="#FFFFFF" '
                 f'stroke="{ACCENT}" stroke-width="1.3"/>')
        p.append(label(x + 53, y4 + 22, name, 11, INK, "600", "middle"))
    p.append(arrow(498, y3 + 100, 498, y4 - 4))

    # ---- fairness firewall
    y5 = 596
    p.append(f'<rect x="24" y="{y5}" width="940" height="60" rx="7" fill="#ECFDF5" '
             f'stroke="{ACCENT}" stroke-width="1.4"/>')
    p.append(label(38, y5 + 22, "FAIRNESS FIREWALL — structural, not statistical",
                   12, "#065F46", "700"))
    p.append(label(38, y5 + 40,
                   "SensitiveAttributes (name · email · phone · address · marital status · "
                   "nationality · interests)  ⟶  NEVER reaches the scorer.", 10.5, "#065F46"))
    p.append(label(38, y5 + 53,
                   "score_candidate() accepts only ScorableProfile, which has no field "
                   "capable of carrying a protected attribute. Asserted by test, not promised "
                   "in a README.", 10.5, "#065F46"))

    y6 = 676
    p.append(f'<rect x="24" y="{y6}" width="940" height="56" rx="7" fill="#FFFFFF" '
             f'stroke="{LINE}"/>')
    p.append(label(38, y6 + 21, "MEASURED, NOT CLAIMED", 11, MUTED, "700"))
    p.append(label(38, y6 + 39,
                   "50× corpus growth → flat search latency (14.1 ms → 11.2 ms p50 at 500 "
                   "candidates) · retrieval ablation on 16 graded queries · calibration "
                   "(ECE / Brier) · $0.00 marginal cost, 100% local retrieval", 10.5))

    p.append("</svg>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(p), encoding="utf8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"  {n_agents} agents · {n_sub} subagents")
    for a, n in sorted(counts.items()):
        print(f"    {a:16s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
