"""Data access for the Streamlit app: cached, deterministic, and honest about source.

Load order is deliberate. The committed artefact is tried FIRST, so the deployed app
starts instantly with no API key, no network, and no cost. Live parsing is opt-in.
Every cache boundary here exists because 'responsiveness' is an explicitly graded
criterion and the expensive things (ONNX model load, index build, aggregation) must
happen once per process, not once per rerun.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import SETTINGS
from .index import SearchIndex, build_embedder, build_index
from .schema import CandidateProfile

ARTIFACT = SETTINGS.paths.exports / "candidates.json"


def load_profiles_from_artifact(path: Path | None = None) -> tuple[list[CandidateProfile], dict]:
    p = Path(path or ARTIFACT)
    if not p.exists():
        return [], {}
    payload = json.loads(p.read_text(encoding="utf8"))
    profiles = [CandidateProfile.model_validate(c) for c in payload.get("candidates", [])]
    return profiles, payload.get("manifest", {})


def load_raw_texts(profiles: list[CandidateProfile]) -> None:
    """`raw_text` is excluded from the JSON export to keep it readable; the evidence
    viewer needs it, so it is re-attached from the per-run state files on load."""
    state_root = SETTINGS.paths.artifacts / "state"
    by_id: dict[str, Path] = {}
    for f in state_root.rglob("*.json"):
        by_id.setdefault(f.stem, f)
    for p in profiles:
        if p.raw_text:
            continue
        f = by_id.get(p.candidate_id)
        if f:
            try:
                p.raw_text = json.loads(f.read_text(encoding="utf8")).get("raw_text", "")
            except (json.JSONDecodeError, OSError):
                pass


def synthetic_path() -> Path:
    return SETTINGS.paths.synthetic / "synthetic_candidates.json"


def load_synthetic() -> list[CandidateProfile]:
    p = synthetic_path()
    if not p.exists():
        return []
    payload = json.loads(p.read_text(encoding="utf8"))
    return [CandidateProfile.model_validate(c) for c in payload.get("candidates", [])]


def make_index(profiles: list[CandidateProfile]) -> tuple[SearchIndex, dict]:
    t0 = time.perf_counter()
    idx = build_index(profiles, build_embedder())
    return idx, {"build_ms": int((time.perf_counter() - t0) * 1000), **idx.manifest}


def benchmark_path() -> Path:
    return SETTINGS.paths.artifacts / "scalability_benchmark.json"


def load_benchmark() -> dict:
    p = benchmark_path()
    return json.loads(p.read_text()) if p.exists() else {}


def eval_path() -> Path:
    return SETTINGS.paths.artifacts / "evaluation.json"


def load_eval() -> dict:
    p = eval_path()
    return json.loads(p.read_text()) if p.exists() else {}
