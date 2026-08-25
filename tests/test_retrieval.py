"""Retrieval behaviour: fusion, filter semantics, and the unknown/missing distinction."""
from millennium.config import ScoreWeights
from millennium.index import HashingEmbedder, build_index
from millennium.retrieval import (apply_filters, parse_query_rules, retrieve, rrf_fuse)
from millennium.schema import CandidateProfile, SkillEntry, Tracked
from millennium.scoring import rank


def _p(cid, headline, **kw):
    p = CandidateProfile(candidate_id=cid, doc_id=f"d_{cid}")
    p.headline = Tracked(value=headline, validation_status="verified", confidence=0.9)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def test_rrf_is_rank_based_not_score_based():
    """RRF must not need score calibration between BM25 and cosine."""
    r = rrf_fuse({"dense": [("a", 0.91), ("b", 0.90)],
                  "lexical": [("b", 88.0), ("a", 2.0)]}, k=60).output
    ids = [x[0] for x in r]
    assert set(ids) == {"a", "b"}
    # Both appear at rank 1 and 2 in one list each, so scores must be equal --
    # the wildly different magnitudes (0.9 vs 88) must not matter.
    assert abs(r[0][1] - r[1][1]) < 1e-9


def test_preferences_never_eliminate_candidates():
    pool = [_p("a", "healthcare analyst"), _p("b", "energy analyst")]
    pq = parse_query_rules("healthcare analyst").output          # soft, no 'must'
    kept, excluded, _cav = apply_filters(pool, pq)
    assert len(kept) == 2 and not excluded, "a preference eliminated a candidate"


def test_must_have_gates_and_records_a_reason():
    from millennium.schema import Classification
    pool = [_p("a", "x", sectors=[Classification(label="healthcare", confidence=0.9)]),
            _p("b", "y", sectors=[Classification(label="energy", confidence=0.9)])]
    pq = parse_query_rules("must have healthcare sector coverage").output
    kept, excluded, _cav = apply_filters(pool, pq)
    assert [p.candidate_id for p in kept] == ["a"]
    assert excluded and "must-have" in excluded[0]["reasons"][0]


def test_unknown_never_excludes_it_becomes_a_caveat():
    """A CV that never mentions sectors is UNVERIFIED, not unqualified. Excluding on
    unknown is how a strong candidate vanishes because of a formatting quirk."""
    pool = [_p("a", "x")]                     # no sectors extracted at all
    pq = parse_query_rules("must have healthcare sector coverage").output
    kept, excluded, caveats = apply_filters(pool, pq)
    assert [p.candidate_id for p in kept] == ["a"], "unknown wrongly excluded a candidate"
    assert not excluded
    assert "a" in caveats and "not stated" in caveats["a"][0]


def test_known_mismatch_does_exclude():
    """The other half of the rule: a candidate we KNOW does not match is gated."""
    from millennium.schema import Classification
    pool = [_p("a", "x", sectors=[Classification(label="energy", confidence=0.9)])]
    pq = parse_query_rules("must have healthcare sector coverage").output
    kept, excluded, _cav = apply_filters(pool, pq)
    assert not kept and excluded
    assert "has energy" in " ".join(excluded[0]["reasons"])


def test_missing_experience_is_not_scored_as_zero():
    """Unknown experience must neither exclude the candidate nor be treated as 0y."""
    pool = [_p("a", "analyst")]               # years_experience unknown
    pq = parse_query_rules("5+ years required").output
    out = rank(pool, pq, ScoreWeights()).output
    assert not out["excluded"], "unknown experience excluded the candidate"
    assert out["ranked"], "candidate disappeared entirely"
    r = out["ranked"][0]
    exp = next(c for c in r.components if c.name == "experience")
    assert exp.score > 0.0, "unknown experience was scored as zero"
    assert "unknown" in exp.note
    assert any("could not be derived" in c for c in out["caveats"].get("a", []))


def test_lexical_retrieval_finds_exact_terms_dense_would_blur():
    p = _p("a", "CFA Level II candidate with kdb+ experience")
    idx = build_index([p], HashingEmbedder())
    assert idx.lexical_search("kdb+", 5), "FTS5 missed an exact rare token"


def test_chunk_aggregation_takes_best_chunk_not_sum():
    """A long CV with many mediocre chunks must not outrank a short perfect match."""
    long_p = _p("long", "generalist analyst")
    for i in range(8):
        long_p.headline = long_p.headline
    idx = build_index([long_p, _p("short", "healthcare equity long short analyst")],
                      HashingEmbedder())
    hits = retrieve(idx, "healthcare equity long short analyst", "hybrid").output
    assert hits[0].candidate_id == "short"


def test_must_prefer_is_decided_per_clause_not_per_query():
    """Regression: a single 'must have' anywhere used to make EVERY matched term
    mandatory — including terms the requisition explicitly marked 'preferred' — which
    gated out the entire pool on a perfectly ordinary JD."""
    req = ("Investment Analyst — Healthcare Long/Short (New York). "
           "Must have 3-7 years in healthcare equity research or healthcare investment "
           "banking, with demonstrated financial modelling. CFA preferred. "
           "Prior buy-side experience at a multi-manager platform preferred.")
    pq = parse_query_rules(req).output
    assert "cfa" not in (pq.must_have.get("certifications") or []), \
        "'CFA preferred' was promoted to a hard requirement"
    assert "cfa" in (pq.preferences.get("certifications") or [])
    assert "healthcare" in (pq.must_have.get("sectors") or [])
    assert pq.must_have.get("min_years") == 3.0
    # "investment banking" is a feeder path, not sector coverage.
    assert "financials" not in (pq.must_have.get("sectors") or []), \
        "'investment banking' wrongly demanded financials sector coverage"


def test_a_realistic_requisition_does_not_empty_the_pool():
    from millennium.schema import Classification
    pool = []
    for i, (sector, skill) in enumerate([("healthcare", "financial_modelling"),
                                         ("healthcare", "equity_research"),
                                         ("technology", "financial_modelling")]):
        p = _p(f"c{i}", "analyst",
               sectors=[Classification(label=sector, confidence=0.9)],
               skills=[SkillEntry(canonical=skill)])
        p.years_experience = Tracked(value=5.0, validation_status="derived")
        pool.append(p)
    req = ("Must have 3-7 years in healthcare equity research or healthcare investment "
           "banking, with demonstrated financial modelling. CFA preferred.")
    pq = parse_query_rules(req).output
    kept, excluded, _cav = apply_filters(pool, pq)
    assert kept, f"a realistic requisition gated out every candidate: " \
                 f"{[e['reasons'] for e in excluded]}"



def test_length_sorted_batching_returns_vectors_in_the_callers_order():
    """The embedder encodes shortest-first internally (2.6x faster -- ONNX pads every
    text in a batch to the longest one in it). That reordering must be invisible.

    This is the failure mode worth a test: `build_index` zips the returned rows against
    chunk ids POSITIONALLY, so an off-by-one in the scatter-back would not raise -- it
    would attach each candidate's vector to somebody else's chunk id and quietly poison
    every search result.

    Asserted as identity-under-nearest-neighbour rather than float equality, because
    padding genuinely does perturb the arithmetic: the same text encoded alone versus
    inside a padded batch agrees only to ~1e-4, so an exact-equality assertion here
    would be testing ONNX's float determinism, not our ordering. What must hold is that
    row i still belongs to text i -- by a margin far wider than that noise.
    """
    import numpy as np
    from millennium.index import build_embedder

    emb = build_embedder()
    texts = ["kdb+ and q for tick data",                                  # short
             "CFA charterholder covering EMEA industrials " * 12,         # long
             "M&A",                                                       # shortest
             "healthcare long/short equity at a multi-manager " * 6]      # medium

    batched = emb.encode(texts)                                  # sorted internally
    alone = np.vstack([emb.encode([t]) for t in texts])          # cannot be reordered
    assert batched.shape == (len(texts), emb.dim)

    sim = batched @ alone.T          # both L2-normalised, so this is cosine
    assert sim.argmax(axis=1).tolist() == list(range(len(texts))), (
        "a row came back paired with the wrong text -- the scatter-back is broken")

    # And the match is unambiguous, not a coin flip between two near-identical rows.
    for i in range(len(texts)):
        runner_up = max(sim[i, j] for j in range(len(texts)) if j != i)
        assert sim[i, i] > 0.999 and sim[i, i] - runner_up > 0.1
