"""Workflow — every real pipeline in this system, as its own page.

Previously this lived as one tab inside System, buried alongside evaluation and
fairness metrics that have nothing to do with "how does this thing actually work."
It is its own top-level page for the same reason Overview is: how the platform works
is a first-class thing a user comes here to answer, not a detail to dig for.

Every diagram below is read directly off the real code (`orchestrator.py`,
`retrieval.py`, `scoring.py`, `assistant.py`) -- not a summary of it. Every subagent
name, every model name, and every "does this call an API" claim is checked against
the live agent registry at render time (`_render_registry_section`), so this page
cannot silently drift out of sync with the code the way a written diagram could.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from millennium.agents.base import registry_table
from millennium.config import SETTINGS
from . import components as C
from . import theme

# The only subagents that ever call the Claude API. Checked against the live registry
# in `_render_registry_section` below (this list is intentionally the single source of
# truth `_mechanism()` reads from), and it must be short: an entry that inflates this
# set inflates the platform's own honesty about what is a model call versus what is
# regex and taxonomy lookups pretending to be one.
_LLM_SUBAGENTS = {"parse.llm_identity", "parse.llm_employment", "parse.llm_profile",
                  "parse.llm_adjudicate", "search.parse_query_llm"}
_LOCAL_MODEL_SUBAGENTS = {"search.retrieve", "search.similar"}


def _mechanism(name: str) -> str:
    if name in _LLM_SUBAGENTS:
        return "🤖 LLM API call"
    if name in _LOCAL_MODEL_SUBAGENTS:
        return "⚙ local embedding model + BM25"
    return "𝑓 rule-based / deterministic"


# ============================================================== 1. document parsing
_PARSE_MERMAID = r"""
flowchart TD
    classDef llm fill:#EEF2FF,color:#3730A3,stroke:#4F46E5,stroke-width:1.5px;
    classDef decision fill:#FFFBEB,color:#92400E,stroke:#D97706,stroke-width:1px;
    classDef io fill:#18181B,color:#ffffff,stroke:#09090B;
    classDef stage fill:#F4F4F5,color:#09090B,stroke:#D4D4D8,stroke-width:1px;

    DOC["Resume — PDF or DOCX"]:::io --> ST1

    subgraph ST1 ["① INGEST — ingestion agent · Python, no API"]
        direction LR
        n1["detect_type\nfile signature check"] --> n2["extract text\npdfplumber / python-docx"]
        n2 --> n3["language\nheuristic langdetect"]
        n2 --> n4["quality score\nheuristic"]
        n2 --> n5["near_duplicate\ntext-hash comparison"]
    end

    ST1 --> ST2

    subgraph ST2 ["② SANITIZE — ingestion agent · Python, no API"]
        n6["injection_scan\npattern rules — hidden text, prompt injection\n(on by default, ENABLE_INJECTION_SCAN)"]
    end

    ST2 --> ST3

    subgraph ST3 ["③ PARSE — parsing agent · reads the SANITIZED text"]
        direction LR
        n7["segment_sections\nregex/heading detector — NOT an LLM chunker"]
        n8["rule_contacts\nregex: email / phone"]
        n9["llm_identity\nClaude Sonnet 4.5 · API call"]:::llm
        n10["llm_employment\nClaude Sonnet 4.5 · API call"]:::llm
        n11["llm_profile\nClaude Sonnet 4.5 · API call"]:::llm
    end

    ST3 --> ST4

    subgraph ST4 ["④ MERGE and GROUND — parsing agent · checks against the ORIGINAL text"]
        direction TB
        n12["merge_identity\nrule vs LLM, span-verify each field"] --> n13{"rule and LLM\ndisagree?"}:::decision
        n14["merge_employment\nspan-verify each role"]
        n13 -- "yes" --> n15["llm_adjudicate\nClaude Sonnet 4.5 · API call\na SECOND, separate call"]:::llm
        n13 -- "no" --> n16["keep merged value"]
        n15 -. "feedback: overwrites the\nconflicted field with the winner" .-> n12
    end

    ST4 --> ST5

    subgraph ST5 ["⑤ CLASSIFY — classification agent · taxonomy + rules, no LLM"]
        direction LR
        n17["skills\nkeyword/taxonomy match"]
        n18["strategy\ntaxonomy match"]
        n19["sector\ntaxonomy match"]
        n20["geography\nemployer-location lookup"]
        n21["quant_profile\nrule scoring"]
        n22["feeder_path\nrule scoring"]
    end

    ST5 --> ST6

    subgraph ST6 ["⑥ VALIDATE — validation agent · Python, no API"]
        direction LR
        n23["dates\ndate math"] --> n24["seniority\nrule scoring"]
        n25["spans\nevidence-span check\nrefuses unverifiable claims"]
        n26["consistency\ncross-field rule checks"]
        n27["completeness\nrule scoring"]
        n28["route_review\ndecides human_review flag"]
    end

    ST6 --> ST7["⑦ FINALIZE\nassemble profile + provenance + cost"]:::stage
    ST7 --> OUT[["candidates.json/csv, employment.csv,\neducation.csv, skills.csv, evidence.csv"]]:::io
    ST7 -. "feedback: a human corrects a\nflagged field — see below" .-> REV["Review queue\n(Review page)"]:::io
"""


def _render_parse_pipeline() -> None:
    st.caption("ingestion → parsing → classification → validation agents · "
              "29 subagents · runs once per document, on Intake or the initial load")
    st.mermaid_chart(_PARSE_MERMAID)
    st.markdown(
        '<div class="mm-banner"><b>Feedback loop 1 — same run:</b> when the '
        'rule-based extractor and the LLM disagree on a field, '
        '<code>llm_adjudicate</code> makes a <b>second, separate</b> Claude call with '
        'both candidate values and the source text, and its winner is written back '
        'into the field it came from — a real loop, bounded to one extra round per '
        'conflicting field, per document.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mm-banner"><b>Feedback loop 2 — human-in-the-loop, persists '
        'across sessions:</b> <code>validate.route_review</code> flags a record → it '
        'appears in the Review queue → a recruiter corrects or approves a field → the '
        'correction is written both to the live profile (<code>validation_status = '
        '"human_corrected"</code>) and permanently to a SQLite audit log '
        '(<code>Store.log_review</code>) — every other page sees the corrected value '
        'from that point on. This is the only loop with a person inside it.</div>',
        unsafe_allow_html=True)


# ============================================================== 2. search / retrieval
_SEARCH_MERMAID = r"""
flowchart TD
    classDef llm fill:#EEF2FF,color:#3730A3,stroke:#4F46E5,stroke-width:1.5px;
    classDef localmodel fill:#EFF6FF,color:#1D4ED8,stroke:#3B82F6,stroke-width:1.5px;
    classDef rule fill:#FAFAFA,color:#52525B,stroke:#E4E4E7,stroke-width:1px;
    classDef decision fill:#FFFBEB,color:#92400E,stroke:#D97706,stroke-width:1px;
    classDef io fill:#18181B,color:#ffffff,stroke:#09090B;

    Q["User query — free text"]:::io --> UQ{"ENABLE_LLM_QUERY flag on?"}:::decision
    UQ -- "yes" --> PL["parse_query_llm\nClaude Sonnet 4.5 · API call"]:::llm
    UQ -- "no, or the call fails" --> PR["parse_query_rules\nregex + taxonomy keywords"]:::rule
    PL -. "falls back on failure" .-> PR
    PL --> PQ["ParsedQuery\nmust_have / preferences / exclusions"]:::rule
    PR --> PQ

    PQ --> DS["dense_search\nlocal embedding model\nno API call, no network"]:::localmodel
    PQ --> LS["lexical_search\nBM25 over SQLite FTS5"]:::rule
    DS --> FUSE["rrf_fuse\nreciprocal-rank fusion"]:::rule
    LS --> FUSE
    FUSE --> AGG["aggregate to candidate\nbest chunk wins, not summed"]:::rule
    AGG --> OUT["Ranked results + evidence chunks"]:::io

    SIM["'More like this' on a\ncandidate's own profile"]:::io --> SS["search.similar\nsame local embedding model"]:::localmodel
    SS --> OUT2["Similar candidates"]:::io
"""


def _render_search_pipeline(index) -> None:
    st.caption(f"search agent · 5 subagents · runs on every query, live · "
              f"embedding model in this session: {index.embedder.name}")
    st.mermaid_chart(_SEARCH_MERMAID)
    st.caption("Query understanding defaults to rules (fast, free, deterministic); "
              "the LLM path is opt-in and only ever replaces the *query parse* step — "
              "the embedding model that actually finds candidates never leaves this "
              "machine, on every query, regardless of that flag.")


# ============================================================ 3. requisition matching
_MATCH_MERMAID = r"""
flowchart LR
    classDef rule fill:#FAFAFA,color:#52525B,stroke:#E4E4E7,stroke-width:1px;
    classDef io fill:#18181B,color:#ffffff,stroke:#09090B;

    JD["Job description text\n+ must-have toggles + weights"]:::io --> SC["score_candidate\nweighted rubric — deterministic math\nover ScorableProfile only"]:::rule
    SC --> RK["rank\nsort + top-N"]:::rule
    RK --> GA["gap_analysis\nwhich must-haves each candidate is missing"]:::rule
    RK --> WS["weight_sensitivity\nhow much the ranking would move\nif weights changed"]:::rule
    RK --> ME["minimal_edit\nsmallest requirement relaxation\nthat would include a near-miss candidate"]:::rule
    GA --> OUT["Ranked shortlist + explanation"]:::io
    WS --> OUT
    ME --> OUT
"""


def _render_match_pipeline() -> None:
    st.caption("matching agent · 5 subagents · runs when a requisition is scored, live")
    st.mermaid_chart(_MATCH_MERMAID)
    st.markdown(
        '<div class="mm-banner">No LLM anywhere in this pipeline, and none of it '
        'could reach one even by mistake: <code>score_candidate</code>\'s type '
        'signature only accepts a <code>ScorableProfile</code>, which structurally '
        'has no field for name, contact details, or any other sensitive attribute — '
        'see System → Fairness for the enforced type boundary.</div>',
        unsafe_allow_html=True)


# ================================================================ 4. analytics/insight
_INSIGHT_MERMAID = r"""
flowchart LR
    classDef rule fill:#FAFAFA,color:#52525B,stroke:#E4E4E7,stroke-width:1px;
    classDef io fill:#18181B,color:#ffffff,stroke:#09090B;

    POOL["Candidate pool\nevery parsed profile"]:::io --> D1["distributions\naggregation"]:::rule
    POOL --> D2["skill_cooccurrence\naggregation"]:::rule
    POOL --> D3["coverage_gaps\naggregation"]:::rule
    POOL --> D4["data_quality\naggregation"]:::rule
    D1 --> OUT["Analytics page charts"]:::io
    D2 --> OUT
    D3 --> OUT
    D4 --> OUT
"""


def _render_insight_pipeline() -> None:
    st.caption("insight agent · 4 subagents · on-demand — runs when the Analytics "
              "page renders, not part of per-document ingest")
    st.mermaid_chart(_INSIGHT_MERMAID)


# =================================================================== 5. chat assistant
_CHAT_MERMAID = r"""
flowchart TD
    classDef llm fill:#EEF2FF,color:#3730A3,stroke:#4F46E5,stroke-width:1.5px;
    classDef rule fill:#FAFAFA,color:#52525B,stroke:#E4E4E7,stroke-width:1px;
    classDef decision fill:#FFFBEB,color:#92400E,stroke:#D97706,stroke-width:1px;
    classDef io fill:#18181B,color:#ffffff,stroke:#09090B;

    U["User message"]:::io --> C1["Claude Sonnet 4.5\nlive API call — NOT the disk-cached\nLLMClient the parsing pipeline uses"]:::llm
    C1 --> D1{"response contains\na tool_use block?"}:::decision
    D1 -- "no" --> F["Final answer\nshown in the chat dock"]:::io
    D1 -- "yes" --> T["execute_tool()\nlocal Python — navigate, search,\nshortlist, filter, summarize, …\nNO further API call"]:::rule
    T --> R{"round < 8?\n(MAX_TOOL_ROUNDS)"}:::decision
    R -- "yes, loop back" --> C1
    R -- "no" --> STOP["Stop — hard cap,\nprevents a runaway loop from\nrunning up API cost"]:::io
"""


def _render_chat_pipeline() -> None:
    st.caption("not in the subagent registry — a separate tool-calling architecture "
              "in assistant.py · runs per chat turn, live, uncached")
    st.mermaid_chart(_CHAT_MERMAID)
    st.markdown(
        '<div class="mm-banner"><b>Feedback loop 3 — within one turn:</b> Claude and '
        'the local tool executor hand control back and forth up to 8 times before the '
        'loop is forcibly stopped. 10 tools are registered — navigate, '
        'search_candidates, set_filters, open_candidate, add/remove/list shortlist, '
        'get_pool_summary, get_candidate_summary, list_candidates. None of them can '
        'delete a record or approve a review — those stay behind your own click on '
        'the Review page.</div>', unsafe_allow_html=True)


# ==================================================================== registry table
def _render_registry_section() -> None:
    rows = registry_table()
    for r in rows:
        r["mechanism"] = _mechanism(r["subagent"])
    df = pd.DataFrame(rows)[["agent", "subagent", "mechanism", "version", "description"]]
    n_llm = sum(1 for r in rows if r["subagent"] in _LLM_SUBAGENTS)
    n_local = sum(1 for r in rows if r["subagent"] in _LOCAL_MODEL_SUBAGENTS)
    n_rule = len(rows) - n_llm - n_local
    k = st.columns(4)
    C.kpi(k[0], len(rows), "subagents registered", f"{df['agent'].nunique()} agent modules")
    C.kpi(k[1], n_llm, "call the LLM API", "Claude Sonnet 4.5", theme.ACCENT)
    C.kpi(k[2], n_local, "call a local model", "embeddings + BM25, no network", "#60A5FA")
    C.kpi(k[3], n_rule, "pure rule-based Python", "regex, taxonomy, date math, span checks")
    st.caption("Every subagent here does real work and is independently testable — "
              "the same registry the System page's own Agents tab reads from, so the "
              "two can never disagree.")
    st.dataframe(df, width="stretch", hide_index=True,
                column_config={"description": st.column_config.TextColumn(width="large")})


def render_workflow(profiles, synth, pool, index, index_manifest, manifest, store,
                    client, bench, evals) -> None:
    st.caption(
        "Five real pipelines run this platform. Each diagram is read off the code "
        "that runs it — every node labelled by what produces it: an LLM API call, "
        "a local model, or deterministic Python.")
    legend = st.columns(4)
    legend[0].markdown('<span class="mm-chip" style="background:#EEF2FF;'
                       'color:#3730A3;border-color:#C7D2FE">LLM API call</span>',
                       unsafe_allow_html=True)
    legend[1].markdown('<span class="mm-chip" style="background:#EFF6FF;'
                       'color:#1D4ED8;border-color:#BFDBFE">local model, no API</span>',
                       unsafe_allow_html=True)
    legend[2].markdown('<span class="mm-chip mm-chip-plain">rule-based / deterministic</span>',
                       unsafe_allow_html=True)
    legend[3].markdown('<span class="mm-chip" style="background:#FFFBEB;'
                       'color:#92400E;border-color:#FDE68A">decision / branch</span>',
                       unsafe_allow_html=True)

    C.section_break("① Document parsing", 0)
    _render_parse_pipeline()
    C.section_break("② Search & retrieval", 1)
    _render_search_pipeline(index)
    C.section_break("③ Requisition matching", 2)
    _render_match_pipeline()
    C.section_break("④ Analytics / insight", 3)
    _render_insight_pipeline()
    C.section_break("⑤ Chat assistant loop", 4)
    _render_chat_pipeline()
    C.section_break("Full registry", 5)
    _render_registry_section()
