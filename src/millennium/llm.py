"""LLM access layer: one call site, deterministic replay, honest cost accounting.

Two properties matter more than provider choice:

1. **Determinism.** Every response is written to `data/llm_cache/` keyed by a hash of
   (provider, model, system, messages, schema). With DEMO_MODE=1 the cache is the only
   source and a miss is a hard error. Conference wifi fails and APIs rate-limit; a demo
   that depends on neither is worth more than one that is 5% smarter.

2. **Structural isolation.** `complete_json` takes the instruction and the untrusted
   document as *separate* blocks and issues the call with no tools. The model cannot
   act on anything it reads in a resume, because it has nothing to act with.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import SETTINGS


class LLMUnavailable(RuntimeError):
    """Raised on a cache miss in DEMO_MODE, or when no key is configured."""


@dataclass
class LLMResponse:
    data: dict | list
    raw_text: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    cached: bool = False
    model: str = ""
    attempts: int = 1
    stop_reason: str = ""


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    per_stage: dict[str, dict] = field(default_factory=dict)

    def add(self, r: LLMResponse, stage: str = "misc") -> None:
        self.calls += 1
        self.cache_hits += int(r.cached)
        self.tokens_in += r.tokens_in
        self.tokens_out += r.tokens_out
        self.cost_usd += r.cost_usd
        self.latency_ms += r.latency_ms
        s = self.per_stage.setdefault(stage, {"calls": 0, "tokens_in": 0, "tokens_out": 0,
                                              "cost_usd": 0.0, "latency_ms": 0, "cache_hits": 0})
        s["calls"] += 1
        s["cache_hits"] += int(r.cached)
        s["tokens_in"] += r.tokens_in
        s["tokens_out"] += r.tokens_out
        s["cost_usd"] += r.cost_usd
        s["latency_ms"] += r.latency_ms


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _extract_json(text: str) -> dict | list:
    """Models occasionally wrap JSON in prose or a fence despite prefill. Be tolerant
    of the wrapper, strict about the payload."""
    t = text.strip()
    m = _JSON_FENCE.search(t)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = t.find(opener), t.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no parseable JSON in model output: {text[:300]!r}")


class LLMClient:
    """Anthropic-first, with an OpenAI-compatible fallback and a disk replay cache."""

    def __init__(self, cfg=None, cache_dir: Path | None = None, demo_mode: bool | None = None):
        self.cfg = cfg or SETTINGS.llm
        self.cache_dir = Path(cache_dir or SETTINGS.paths.llm_cache)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.demo_mode = SETTINGS.flags.demo_mode if demo_mode is None else demo_mode
        self.usage = Usage()
        self._client = None

    # ---------------------------------------------------------------- plumbing
    def _key(self, system: str, blocks: list[dict], schema_hint: str) -> str:
        payload = json.dumps({"p": self.cfg.provider, "m": self.cfg.model, "s": system,
                              "b": blocks, "h": schema_hint, "t": self.cfg.temperature},
                             sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _connect(self):
        if self._client is not None:
            return self._client
        if self.cfg.provider == "anthropic":
            import anthropic
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise LLMUnavailable(
                    "ANTHROPIC_API_KEY is not set. Either export it to run live parsing, "
                    "or keep DEMO_MODE=1 to replay the committed cache.")
            self._client = anthropic.Anthropic(api_key=key, timeout=self.cfg.timeout_s)
        else:
            from openai import OpenAI
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise LLMUnavailable("OPENAI_API_KEY is not set.")
            self._client = OpenAI(api_key=key, timeout=self.cfg.timeout_s)
        return self._client

    def _price(self, tin: int, tout: int) -> float:
        return (tin / 1e6) * self.cfg.price_in_per_mtok + (tout / 1e6) * self.cfg.price_out_per_mtok

    # ------------------------------------------------------------------- public
    def complete_json(self, system: str, blocks: list[dict], schema_hint: str = "",
                      stage: str = "misc", max_tokens: int | None = None) -> LLMResponse:
        """`blocks` is a list of {"role","content"} messages already segregated by
        trust level (see prompts.build_extraction_messages). No tools are passed."""
        key = self._key(system, blocks, schema_hint)
        cpath = self._cache_path(key)

        if cpath.exists():
            payload = json.loads(cpath.read_text())
            r = LLMResponse(data=payload["data"], raw_text=payload["raw_text"],
                            tokens_in=payload.get("tokens_in", 0),
                            tokens_out=payload.get("tokens_out", 0),
                            cost_usd=0.0, latency_ms=payload.get("latency_ms", 0),
                            cached=True, model=payload.get("model", self.cfg.model))
            self.usage.add(r, stage)
            return r

        if self.demo_mode:
            raise LLMUnavailable(
                f"DEMO_MODE=1 and no cached response for stage '{stage}' (key {key[:12]}). "
                f"Run the pipeline once with DEMO_MODE=0 and an API key to populate the cache.")

        client = self._connect()
        last_err: Exception | None = None
        for attempt in range(1, self.cfg.max_retries + 1):
            t0 = time.perf_counter()
            try:
                if self.cfg.provider == "anthropic":
                    msgs = list(blocks)
                    # Prefill the opening brace: forces JSON, removes preamble entirely.
                    msgs.append({"role": "assistant", "content": "{"})
                    resp = client.messages.create(
                        model=self.cfg.model,
                        max_tokens=max_tokens or self.cfg.max_tokens,
                        temperature=self.cfg.temperature,
                        system=system,
                        messages=msgs,
                    )
                    raw = "{" + "".join(b.text for b in resp.content if b.type == "text")
                    tin, tout = resp.usage.input_tokens, resp.usage.output_tokens
                    stop = resp.stop_reason or ""
                else:
                    resp = client.chat.completions.create(
                        model=self.cfg.model,
                        max_tokens=max_tokens or self.cfg.max_tokens,
                        temperature=self.cfg.temperature,
                        response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": system}] + blocks,
                    )
                    raw = resp.choices[0].message.content or ""
                    tin, tout = resp.usage.prompt_tokens, resp.usage.completion_tokens
                    stop = resp.choices[0].finish_reason or ""

                latency = int((time.perf_counter() - t0) * 1000)
                if stop in ("max_tokens", "length"):
                    raise ValueError("response truncated at max_tokens; raise the budget")
                data = _extract_json(raw)
                r = LLMResponse(data=data, raw_text=raw, tokens_in=tin, tokens_out=tout,
                                cost_usd=self._price(tin, tout), latency_ms=latency,
                                cached=False, model=self.cfg.model, attempts=attempt,
                                stop_reason=stop)
                cpath.write_text(json.dumps({
                    "data": data, "raw_text": raw, "tokens_in": tin, "tokens_out": tout,
                    "latency_ms": latency, "model": self.cfg.model, "stage": stage,
                }, ensure_ascii=False, indent=1))
                self.usage.add(r, stage)
                return r
            except Exception as e:  # noqa: BLE001 -- retried with backoff, then surfaced
                last_err = e
                if attempt < self.cfg.max_retries:
                    time.sleep(min(2 ** attempt, 12))
        raise RuntimeError(f"LLM call failed after {self.cfg.max_retries} attempts: {last_err}")

    # ------------------------------------------------------------- diagnostics
    def cache_stats(self) -> dict:
        files = list(self.cache_dir.glob("*.json"))
        return {"entries": len(files),
                "bytes": sum(f.stat().st_size for f in files),
                "demo_mode": self.demo_mode,
                "model": self.cfg.model,
                "provider": self.cfg.provider}


def default_client() -> LLMClient:
    return LLMClient()
