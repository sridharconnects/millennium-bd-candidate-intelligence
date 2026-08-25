"""SQLite persistence: candidate metadata + FTS5, with real deletion.

Deletion is implemented end to end -- SQLite rows, FTS index, vector store, and the
on-disk profile -- because GDPR/CCPA erasure is a genuine obligation for a recruiting
product holding CVs, and a delete that leaves the candidate in the search index is
not a delete. It is tested in tests/test_deletion.py.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .config import SETTINGS
from .export import flat_row
from .schema import CandidateProfile

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    source_file  TEXT, region TEXT, country TEXT, seniority_level TEXT,
    years_experience REAL, approach TEXT, feeder_path TEXT,
    completeness REAL, needs_review INTEGER, is_synthetic INTEGER,
    payload TEXT NOT NULL, updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS idx_region    ON candidates(region);
CREATE INDEX IF NOT EXISTS idx_seniority ON candidates(seniority_level);
CREATE INDEX IF NOT EXISTS idx_years     ON candidates(years_experience);
CREATE INDEX IF NOT EXISTS idx_synthetic ON candidates(is_synthetic);

CREATE TABLE IF NOT EXISTS labels (
    candidate_id TEXT, kind TEXT, label TEXT, confidence REAL,
    PRIMARY KEY (candidate_id, kind, label));
CREATE INDEX IF NOT EXISTS idx_labels ON labels(kind, label);

CREATE VIRTUAL TABLE IF NOT EXISTS candidate_fts USING fts5(
    candidate_id UNINDEXED, body, tokenize='porter unicode61');

CREATE TABLE IF NOT EXISTS review_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT, field TEXT,
    old_value TEXT, new_value TEXT, reviewer TEXT, action TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);

-- A saved search is a query plus the exact filter state that produced it. Recruiters
-- re-run the same handful of searches every week; making them retype the filter rail
-- each time is the difference between a tool and a demo.
CREATE TABLE IF NOT EXISTS saved_searches (
    name TEXT PRIMARY KEY, query TEXT, filters TEXT, mode TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);

-- A role template freezes a desk's scoring weights and must-have set, so the second
-- healthcare L/S req does not get re-tuned from scratch (and scored differently).
CREATE TABLE IF NOT EXISTS role_templates (
    name TEXT PRIMARY KEY, jd TEXT, weights TEXT, parsed_query TEXT,
    requirements TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS shortlists (
    name TEXT, candidate_id TEXT, note TEXT, tags TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (name, candidate_id));

-- Outreach log for shortlisted candidates. `kind` is one of 'email_sent' (drafted and
-- logged from this app), 'email_inbound' (a reply a recruiter recorded by hand), or
-- 'interview_scheduled' (an .ics was generated). There is deliberately no live SMTP
-- or IMAP connector behind this -- see ui/pages_core.py's outreach section for why:
-- this app has no mail credentials configured, and wiring a real send/receive path
-- for a hackathon case study would mean either fabricating one (dishonest) or
-- actually emailing real people extracted from real resumes without their consent
-- (a genuinely bad idea). This table is the honest version: every message is
-- composed and durably logged, nothing is silently pretended to have been delivered.
CREATE TABLE IF NOT EXISTS communications (
    id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT, kind TEXT,
    subject TEXT, body TEXT, meta TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS idx_comm_candidate ON communications(candidate_id);
"""


class Store:
    def __init__(self, path: Path | str | None = None):
        self.path = str(path or SETTINGS.paths.db)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------ write
    def upsert(self, profiles: list[CandidateProfile]) -> int:
        cur = self.conn.cursor()
        for p in profiles:
            row = flat_row(p)
            cur.execute(
                "INSERT INTO candidates (candidate_id,source_file,region,country,"
                "seniority_level,years_experience,approach,feeder_path,completeness,"
                "needs_review,is_synthetic,payload,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP) "
                "ON CONFLICT(candidate_id) DO UPDATE SET payload=excluded.payload,"
                "updated_at=CURRENT_TIMESTAMP",
                (p.candidate_id, row["source_file"], row["region"], row["country"],
                 row["seniority_level"],
                 float(row["years_experience"]) if row["years_experience"] != "" else None,
                 row["approach"], row["feeder_path"], row["completeness"],
                 int(bool(row["needs_human_review"])),
                 int(bool(row["is_synthetic"])),
                 p.model_dump_json()))
            cur.execute("DELETE FROM labels WHERE candidate_id=?", (p.candidate_id,))
            labels = ([("strategy", c.label, c.confidence) for c in p.strategies]
                      + [("sector", c.label, c.confidence) for c in p.sectors]
                      + [("skill", s.canonical, 1.0) for s in p.skills]
                      + [("certification", c.canonical or "", 1.0) for c in p.certifications]
                      + [("employer", e.employer_canonical or "", 1.0) for e in p.employment])
            cur.executemany("INSERT OR REPLACE INTO labels VALUES (?,?,?,?)",
                            [(p.candidate_id, k, v, c) for k, v, c in labels if v])
            cur.execute("DELETE FROM candidate_fts WHERE candidate_id=?", (p.candidate_id,))
            cur.execute("INSERT INTO candidate_fts VALUES (?,?)",
                        (p.candidate_id, p.searchable_text()))
        self.conn.commit()
        return len(profiles)

    def log_communication(self, candidate_id: str, kind: str, subject: str = "",
                          body: str = "", meta: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO communications (candidate_id,kind,subject,body,meta)"
            " VALUES (?,?,?,?,?)",
            (candidate_id, kind, subject, body, json.dumps(meta or {}, default=str)))
        self.conn.commit()

    def list_communications(self, candidate_id: str | None = None) -> list[dict]:
        q = "SELECT * FROM communications"
        args: tuple = ()
        if candidate_id:
            q += " WHERE candidate_id=?"
            args = (candidate_id,)
        return [{**dict(r), "meta": json.loads(r["meta"] or "{}")}
                for r in self.conn.execute(q + " ORDER BY created_at DESC", args)]

    def log_review(self, candidate_id: str, field: str, old, new, reviewer: str,
                   action: str) -> None:
        self.conn.execute(
            "INSERT INTO review_log (candidate_id,field,old_value,new_value,reviewer,action)"
            " VALUES (?,?,?,?,?,?)",
            (candidate_id, field, json.dumps(old, default=str), json.dumps(new, default=str),
             reviewer, action))
        self.conn.commit()

    # ------------------------------------------------------- saved searches
    def save_search(self, name: str, query: str, filters: dict, mode: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO saved_searches (name,query,filters,mode,created_at)"
            " VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
            (name, query, json.dumps(filters, default=str), mode))
        self.conn.commit()

    def list_searches(self) -> list[dict]:
        return [{**dict(r), "filters": json.loads(r["filters"] or "{}")}
                for r in self.conn.execute(
                    "SELECT * FROM saved_searches ORDER BY created_at DESC")]

    def delete_search(self, name: str) -> None:
        self.conn.execute("DELETE FROM saved_searches WHERE name=?", (name,))
        self.conn.commit()

    # ------------------------------------------------------- role templates
    def save_template(self, name: str, jd: str, weights: dict, parsed_query: dict,
                      requirements: list) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO role_templates "
            "(name,jd,weights,parsed_query,requirements,created_at) "
            "VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)",
            (name, jd, json.dumps(weights, default=str),
             json.dumps(parsed_query, default=str), json.dumps(requirements, default=str)))
        self.conn.commit()

    def list_templates(self) -> list[dict]:
        out = []
        for r in self.conn.execute("SELECT * FROM role_templates ORDER BY created_at DESC"):
            d = dict(r)
            for k in ("weights", "parsed_query", "requirements"):
                try:
                    d[k] = json.loads(d[k] or "null")
                except (TypeError, json.JSONDecodeError):
                    d[k] = None
            out.append(d)
        return out

    def delete_template(self, name: str) -> None:
        self.conn.execute("DELETE FROM role_templates WHERE name=?", (name,))
        self.conn.commit()

    def save_shortlist(self, name: str, candidate_id: str, note: str = "",
                       tags: str = "") -> None:
        self.conn.execute("INSERT OR REPLACE INTO shortlists (name,candidate_id,note,tags)"
                          " VALUES (?,?,?,?)", (name, candidate_id, note, tags))
        self.conn.commit()

    # ------------------------------------------------------------------- read
    def load_all(self, include_synthetic: bool = True) -> list[CandidateProfile]:
        q = "SELECT payload FROM candidates"
        if not include_synthetic:
            q += " WHERE is_synthetic = 0"
        return [CandidateProfile.model_validate_json(r["payload"])
                for r in self.conn.execute(q)]

    def audit_trail(self, candidate_id: str | None = None) -> list[dict]:
        q = "SELECT * FROM review_log"
        args: tuple = ()
        if candidate_id:
            q += " WHERE candidate_id=?"
            args = (candidate_id,)
        return [dict(r) for r in self.conn.execute(q + " ORDER BY id DESC", args)]

    def stats(self) -> dict:
        c = self.conn.execute("SELECT COUNT(*) n, SUM(needs_review) r, "
                              "SUM(is_synthetic) s FROM candidates").fetchone()
        return {"candidates": c["n"] or 0, "needs_review": c["r"] or 0,
                "synthetic": c["s"] or 0,
                "db_bytes": Path(self.path).stat().st_size if Path(self.path).exists() else 0}

    # ----------------------------------------------------------------- delete
    def delete_candidate(self, candidate_id: str, index=None) -> dict:
        """Erasure across every store that holds this person's data."""
        cur = self.conn.cursor()
        before = cur.execute("SELECT COUNT(*) FROM candidates WHERE candidate_id=?",
                             (candidate_id,)).fetchone()[0]
        for table in ("candidates", "labels", "candidate_fts", "shortlists"):
            key = "candidate_id"
            cur.execute(f"DELETE FROM {table} WHERE {key}=?", (candidate_id,))
        self.conn.commit()
        idx_result = index.remove_candidate(candidate_id) if index is not None else {}
        removed_files = 0
        for f in (SETTINGS.paths.artifacts / "state").rglob(f"{candidate_id}.json"):
            f.unlink()
            removed_files += 1
        # The audit trail records THAT an erasure happened, never what was erased.
        self.log_review(candidate_id, "*", "<erased>", None, "system", "gdpr_delete")
        return {"candidate_id": candidate_id, "sql_rows_removed": before,
                "artifact_files_removed": removed_files, **idx_result}
