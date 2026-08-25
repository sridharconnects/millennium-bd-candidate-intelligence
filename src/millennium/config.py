"""Central configuration. Every tunable lives here, nothing is hardcoded downstream.

Feature flags exist so that the core demo can be run with every optional subsystem
disabled (see tests/test_flags.py). That is a hard requirement, not a nicety: an
optional feature that can break the core path is a liability during a live demo.
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]

# Load .env before any Settings object reads the environment. Secrets live in .env
# (gitignored) or Streamlit secrets -- never in code, never in the notebook.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:  # dotenv is a convenience, not a requirement
    pass


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Paths(BaseModel):
    root: Path = ROOT
    resumes: Path = ROOT
    raw_text: Path = ROOT / "data" / "raw_text"
    artifacts: Path = ROOT / "data" / "artifacts"
    exports: Path = ROOT / "data" / "exports"
    llm_cache: Path = ROOT / "data" / "llm_cache"
    gold: Path = ROOT / "data" / "gold"
    synthetic: Path = ROOT / "data" / "synthetic"
    index: Path = ROOT / "data" / "index"
    db: Path = ROOT / "data" / "candidates.sqlite"

    def ensure(self) -> "Paths":
        for p in (self.raw_text, self.artifacts, self.exports, self.llm_cache,
                  self.gold, self.synthetic, self.index):
            p.mkdir(parents=True, exist_ok=True)
        return self


class LLMConfig(BaseModel):
    provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic"))
    model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-sonnet-4-5-20250929"))
    max_tokens: int = 8000
    temperature: float = 0.0
    timeout_s: int = 120
    max_retries: int = 4
    # Published per-MTok pricing; used for the cost dashboard.
    price_in_per_mtok: float = 3.00
    price_out_per_mtok: float = 15.00


class RetrievalConfig(BaseModel):
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    rrf_k: int = 60
    top_k_dense: int = 50
    top_k_lexical: int = 50
    top_k_final: int = 25
    chunk_max_chars: int = 900


class ScoreWeights(BaseModel):
    """Requisition match weights. Sum is normalised at scoring time, so a user can
    edit any single slider in the UI without having to rebalance the rest."""
    skills: float = 0.30
    strategy: float = 0.20
    sector: float = 0.15
    semantic: float = 0.15
    geography: float = 0.10
    experience: float = 0.05
    data_quality: float = 0.05

    def normalised(self) -> dict[str, float]:
        d = self.model_dump()
        total = sum(d.values()) or 1.0
        return {k: v / total for k, v in d.items()}


class Flags(BaseModel):
    demo_mode: bool = Field(default_factory=lambda: _flag("DEMO_MODE", True))
    enable_semantic: bool = Field(default_factory=lambda: _flag("ENABLE_SEMANTIC", True))
    enable_llm_query_parse: bool = Field(default_factory=lambda: _flag("ENABLE_LLM_QUERY", True))
    enable_counterfactuals: bool = Field(default_factory=lambda: _flag("ENABLE_COUNTERFACTUALS", True))
    enable_injection_scan: bool = Field(default_factory=lambda: _flag("ENABLE_INJECTION_SCAN", True))
    enable_synthetic: bool = Field(default_factory=lambda: _flag("ENABLE_SYNTHETIC", True))
    blind_review: bool = Field(default_factory=lambda: _flag("BLIND_REVIEW", False))


class Settings(BaseModel):
    paths: Paths = Field(default_factory=lambda: Paths().ensure())
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    weights: ScoreWeights = Field(default_factory=ScoreWeights)
    flags: Flags = Field(default_factory=Flags)

    schema_version: str = "1.3.0"
    taxonomy_version: str = "1.2.0"
    # Span verification threshold. Below this the value is discarded, not downgraded.
    span_fuzzy_threshold: float = 0.92


SETTINGS = Settings()
