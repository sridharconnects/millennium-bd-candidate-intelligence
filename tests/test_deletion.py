"""GDPR Art. 17 erasure, end to end. A delete that leaves the person in the search
index is not a delete."""
from millennium.index import FaissStore, HashingEmbedder, NumpyStore, build_index
from millennium.schema import CandidateProfile, Tracked
from millennium.store import Store


def _profile(cid: str, text: str) -> CandidateProfile:
    p = CandidateProfile(candidate_id=cid, doc_id=f"doc_{cid}")
    p.headline = Tracked(value=text, validation_status="verified", confidence=0.9)
    return p


def test_delete_removes_from_sql_fts_and_vector_index(tmp_path):
    a = _profile("aaa", "healthcare long short analyst in New York")
    b = _profile("bbb", "quantitative developer c++ derivatives in Paris")
    idx = build_index([a, b], HashingEmbedder())
    store = Store(tmp_path / "t.sqlite")
    store.upsert([a, b])

    assert store.stats()["candidates"] == 2
    before = idx.store.size()
    assert any(c.candidate_id == "aaa" for c in idx.chunks.values())

    res = store.delete_candidate("aaa", idx)

    assert res["sql_rows_removed"] == 1
    assert res["vectors_removed"] >= 1
    assert store.stats()["candidates"] == 1
    assert not any(c.candidate_id == "aaa" for c in idx.chunks.values())
    assert idx.store.size() == before - res["chunks_removed"]
    assert not idx.db.execute(
        "SELECT 1 FROM chunks WHERE candidate_id='aaa'").fetchall()
    assert not idx.db.execute(
        "SELECT 1 FROM chunks WHERE chunks MATCH 'healthcare'"
        " AND candidate_id='aaa'").fetchall()
    # The surviving candidate must be untouched.
    assert idx.lexical_search("quantitative", 5)


def test_erasure_is_audited_without_retaining_the_erased_data(tmp_path):
    a = _profile("aaa", "healthcare analyst")
    store = Store(tmp_path / "t.sqlite")
    store.upsert([a])
    store.delete_candidate("aaa")
    trail = store.audit_trail("aaa")
    assert trail and trail[0]["action"] == "gdpr_delete"
    assert "healthcare" not in json_dumps_all(trail)


def json_dumps_all(rows) -> str:
    import json
    return json.dumps(rows, default=str)


def test_both_vector_stores_support_removal():
    """The VectorStore ABC is a real seam, so both implementations must honour it."""
    import numpy as np
    for cls in (FaissStore, NumpyStore):
        s = cls(8)
        v = np.eye(8, dtype="float32")[:3]
        s.add(v, ["a", "b", "c"])
        assert s.size() == 3
        assert s.remove(["b"]) == 1
        assert s.size() == 2
        assert [i for i, _ in s.search(v[0], 3)] == ["a"] or "b" not in \
            [i for i, _ in s.search(v[0], 3)]
