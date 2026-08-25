"""Agent contract.

Seven agents, each with real subagents. The contract is uniform so that the
orchestrator, the memoisation cache, the UI pipeline trace, and the tests all speak
one language.

Two properties are non-negotiable:

* **Graceful degradation.** A subagent that fails returns `status='failed'` with an
  empty output. It never raises into the orchestrator. The consequence downstream is
  abstained fields, not a crashed batch -- one malformed resume out of five hundred
  must not take the run down.
* **Determinism.** Every subagent is memoised on `inputs_hash`, so re-running a
  pipeline over unchanged inputs is free and produces byte-identical results. That is
  what makes the offline demo and the replay tests possible.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

from ..schema import Evidence

T = TypeVar("T")

Status = Literal["ok", "partial", "failed", "skipped"]


class AgentResult(BaseModel, Generic[T]):
    name: str
    version: str = "1.0"
    status: Status = "ok"
    output: T | None = None
    confidence: float = 1.0
    evidence: list[Evidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    latency_ms: int = 0
    retries: int = 0
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    inputs_hash: str = ""
    cached: bool = False
    children: list["AgentResult"] = Field(default_factory=list)

    def flatten(self) -> list["AgentResult"]:
        out = [self]
        for c in self.children:
            out.extend(c.flatten())
        return out

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "partial")


AgentResult.model_rebuild()


def hash_inputs(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class _Entry:
    fn: Callable
    version: str
    parent: str
    description: str


REGISTRY: dict[str, _Entry] = {}


def subagent(name: str, parent: str, version: str = "1.0", description: str = ""):
    """Register a subagent. The registry is what the System page renders, so a
    subagent that is not registered is invisible -- and one that does nothing is
    deleted rather than left as a decorative node."""
    def deco(fn: Callable) -> Callable:
        REGISTRY[name] = _Entry(fn=fn, version=version, parent=parent,
                                description=description or (fn.__doc__ or "").strip().split("\n")[0])
        fn._agent_name = name  # type: ignore[attr-defined]
        return fn
    return deco


_MEMO: dict[str, AgentResult] = {}


def run_subagent(name: str, *args, timeout_ms: int | None = None,
                 memo: bool = True, **kwargs) -> AgentResult:
    """Invoke a registered subagent with timing, memoisation and error containment."""
    entry = REGISTRY.get(name)
    if entry is None:
        return AgentResult(name=name, status="failed",
                           errors=[f"subagent '{name}' is not registered"])

    ih = hash_inputs(name, entry.version, args, sorted(kwargs.items()))
    if memo and ih in _MEMO:
        cached = _MEMO[ih].model_copy(deep=True)
        cached.cached = True
        return cached

    t0 = time.perf_counter()
    try:
        result = entry.fn(*args, **kwargs)
        if not isinstance(result, AgentResult):
            result = AgentResult(name=name, output=result)
        result.name = name
        result.version = entry.version
    except Exception as exc:  # noqa: BLE001 -- containment is the point
        result = AgentResult(
            name=name, version=entry.version, status="failed", confidence=0.0,
            errors=[f"{type(exc).__name__}: {exc}"],
            warnings=[f"degraded: downstream fields from '{name}' will be abstained"],
        )
        result.errors.append(traceback.format_exc(limit=3).strip().splitlines()[-1])

    result.latency_ms = int((time.perf_counter() - t0) * 1000)
    result.inputs_hash = ih
    if timeout_ms and result.latency_ms > timeout_ms:
        result.warnings.append(f"exceeded soft timeout ({result.latency_ms}ms > {timeout_ms}ms)")
        if result.status == "ok":
            result.status = "partial"
    if memo:
        _MEMO[ih] = result.model_copy(deep=True)
    return result


def clear_memo() -> None:
    _MEMO.clear()


def registry_table() -> list[dict]:
    return sorted(
        ({"subagent": k, "agent": v.parent, "version": v.version, "description": v.description}
         for k, v in REGISTRY.items()),
        key=lambda r: (r["agent"], r["subagent"]))
