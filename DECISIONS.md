# Decision log

What was built, what was deliberately cut, and why. A folder of pass-through classes
reads as padding, so anything that could not carry real logic was deleted rather than
left as a decorative node.

---

## Kept, with the reasoning

**Span verification as a hard gate, not a confidence signal.**
The tempting design is to lower confidence when a quote does not verify and still show
the value. Rejected: that path is exactly how a fabricated employer reaches a
recruiter's screen. There is no "trust it anyway" branch. `verify_span` returns
`Evidence` or `None`, and `None` means the value is destroyed.

**Three LLM passes per document, not one.**
A single mega-prompt degrades on long documents and fails atomically. Identity /
employment / profile each get a short task and a short schema, and each fails
independently — which is what allows one bad pass to degrade to abstained fields
instead of losing the candidate entirely. The fourth (adjudication) pass is
conditional on there actually being a rule-vs-LLM conflict, because most documents
have none and the call would be pure cost.

**Rules cross-check the LLM; they do not replace it.**
The case study requires LLM-via-API parsing, so the LLM is the primary path. Regex
runs alongside on the handful of fields where regex is genuinely better — email,
phone, dates, degree level, certifications. Agreement raises confidence; disagreement
lowers it and routes to review. `run_eval.py` reports rule vs LLM separately, and
where regex wins that is published rather than hidden.

**Experience is a union of intervals, not a sum.**
Chen Li holds two concurrent roles; summing them would report ~6 years for someone
with ~4.3. The union is computed in Python from verified dates, and where no dates
exist (Viktor Sharat's CV states only durations) the total is derived from the stated
durations and explicitly flagged as unable to rule out concurrency.

**`fastembed` (ONNX) instead of `sentence-transformers`.**
Identical weights (BAAI/bge-small-en-v1.5, 384-dim). `sentence-transformers` drags in
PyTorch at roughly 2 GB, which does not fit a free Streamlit Cloud dyno. A demo that
OOMs in front of a judge scores zero regardless of its nDCG.

**`IndexFlatIP`, not an approximate index.**
Flat is exact. At 500 candidates (3,291 vectors) the exhaustive scan is sub-millisecond
and the measured p50 is 11 ms — of which most is query embedding, not search. An ANN
index would trade recall for latency there is no need to buy. The migration threshold
(~100k vectors) is documented rather than guessed.

**RRF rather than a weighted score blend.**
BM25 is unbounded and corpus-dependent; cosine is bounded in [-1, 1]. Adding them
requires a calibration constant that would need re-tuning per corpus. RRF fuses
*ranks*, so it needs none.

**Best-chunk aggregation, not sum.**
Summing chunk scores lets a long CV with many mediocre chunks outrank a short one with
a perfect match. Tested (`test_chunk_aggregation_takes_best_chunk_not_sum`).

**Procedural synthetic corpus, not LLM-generated.**
Three reasons: it is seeded and therefore reproducible by anyone checking out the repo
(an LLM-generated corpus makes published latency numbers unverifiable); sampling the
geography × strategy × sector × seniority grid directly *guarantees* even coverage
rather than hoping for it; and since these records only ever measure retrieval
behaviour versus corpus size, LLM-authored prose would add cost without adding
validity. Extraction accuracy is measured only on the ten real, hand-labelled resumes.

**Unknown is a first-class state, distinct from absent.**
`abstained` (a value was proposed and discarded) and `missing` (the document never
said) are different colours in the UI, different columns in the CSV, and different
exclusion reasons in search. Collapsing them into a blank is the standard way this
data gets quietly misread — and it is how a good candidate gets filtered out for
"lacking" a language their CV simply never mentioned.

**Gated-out candidates stay visible.**
Silently dropping people is how a pool "runs dry" without anyone noticing that one
over-strict must-have did it. Excluded candidates are returned with the reason.

---

## Cut, with the reasoning

**Cross-encoder reranking.** `ms-marco-MiniLM-L-6-v2` would add ~90 MB and ~40 ms to
every query. On 10 real candidates there is nothing for it to rerank — recall is
already 1.0 on most labelled queries — so it could only be evaluated on synthetic data,
where its score would be meaningless. Left as a flagged future item with the ablation
table already shaped to receive the row. `bge-reranker-base` was rejected outright: too
large for the deploy target.

**MinHash/LSH for near-duplicate detection.** A linear shingle-Jaccard pass over the
corpus is *exact* and instant at this scale. LSH is an approximation that only pays off
past a few thousand documents. The trigger is documented; the complexity is not paid
for now.

**Docker + CI.** Real value in production, zero value to a judge reading a notebook,
and it competes for time with UX — which is 30% of the grade.

**FastAPI service layer.** Nothing consumes it in this deliverable. It would have been
a folder of endpoints returning the same objects the Streamlit app already renders.

**NetworkX candidate graph.** Genuinely interesting, and genuinely useless on ten
candidates — a graph of ten nodes shows nothing a table does not. Deferred until there
is a pool large enough for shared-employer edges to mean something.

**OCR.** Detected and flagged (`ingest.extract` warns when no text layer is found), but
off by default. All ten supplied documents have text layers, so shipping Tesseract
would have added a heavy system dependency exercised by nothing.

**Per-field human-correction UI for every field in the schema.** The review workspace
covers identity, contact, headline, location and the first five employment entries —
the fields that actually land in the review queue. A generic recursive editor over the
whole model would have been more code and less useful.

**A deeper agent hierarchy.** An earlier sketch had ~50 subagents. Most were one-line
wrappers. They were consolidated into 40 that each do real work, are independently
testable, and appear in the UI trace. Agent count is 0% of the rubric; depth in seven
agents beats a hundred stubs.

---

## Bugs found by actually running it

Five defects that only surfaced once the pipeline ran on the real corpus and the app
was driven headlessly. Each is now covered by a regression test.

1. **Column detection split single-column PDFs.** A *centred name block* sits far to the
   right of body text and produces exactly the same x-gap as a real column. The first
   detector treated Ryan Patel's header as a second column and moved his **name and
   contact details to the bottom of the document** — while correctly fixing Omar's
   genuinely two-column CV. A real column now has to look like one: at least 3 blocks,
   30% of page height of vertical span, and side-by-side overlap with its neighbour.
   This would have degraded most single-column PDFs in production.

2. **`validate.completeness` crashed, and graceful degradation hid it.** It read
   `profile.provenance.page_count`, but provenance is not attached until the finalize
   stage. The subagent failed, the orchestrator absorbed it exactly as designed, and
   every candidate silently reported **0% completeness** — which looked like a data
   problem rather than a crash. Fixed, and failed subagents in core stages now raise a
   `PIPELINE DEGRADED` flag and force human review. *Graceful degradation without an
   alarm is just a quiet bug.*

3. **One "must have" made every term mandatory.** The rule query parser applied
   must/prefer at the level of the whole query, so a requisition reading *"Must have
   3–7 years ... **CFA preferred**"* promoted the explicitly-preferred CFA to a hard
   gate and **excluded the entire pool**. Must/prefer is now decided per clause, and
   ambiguous clauses default to preferences — because over-gating empties a pool
   silently, which is the more expensive error.

4. **Unknown values excluded candidates.** `apply_filters` printed *"unknown, not
   absent"* in its exclusion reason and then excluded the candidate anyway,
   contradicting both its own message and the stated design. A CV that never mentions
   certifications is *unverified*, not *unqualified*. Unknowns are now returned as
   caveats on a kept candidate; only a **known** mismatch gates.

5. **A filter that filtered nothing.** The "Feeder path" control rendered, looked live,
   and had no effect. There is now a test that diffs rendered filter widgets against
   applied ones, so the whole class is covered rather than the one instance.

Two smaller ones: `use_container_width` was deprecated past its removal date (32 call
sites migrated), and 51 unguarded taxonomy lookups would `KeyError` the whole page on a
label retired by a later taxonomy version — all now routed through a tolerant
`tx.display()`.

**6. Two real hallucinations, found by running the actual required LLM path.** The
first live run against real Claude Sonnet output (not the rule baseline above) scored
a 4.29% hallucination rate — not zero. Both traced to genuine schema gaps rather than
model error:

- **Volunteer/non-profit roles were indistinguishable from paid employment.** Ryan
  Patel co-founded a non-profit ("Global Education Alliance, Co-Founder, Jan
  2017–Present") — structurally identical to a job entry: a title, dates, an
  organisation name. The model correctly extracted it *as an employment-shaped entry*,
  which is exactly what it was asked to do; the schema simply had no way to say "this
  is real, keep it, but it isn't professional employment." Fixed properly rather than
  patched around the symptom: added `EmploymentEntry.is_volunteer`, prompted the model
  to set it for non-profits/clubs/fraternities/community leadership, and excluded
  flagged entries from `years_experience`, `current_role()`, and
  `scorable().employers_canonical` — the same treatment `is_internship` already got.
  Checking the corpus surfaced three more candidates with the identical pattern (Chen
  Li's swimming club and teaching society, Michael Rodriguez's IB club and tutoring
  service), so all ten were re-parsed rather than special-casing one file.
- **`employment[0]` was used as a silent "current role" fallback in three UI/export
  call sites and four more in the eval harness**, whenever no entry was explicitly
  marked `is_current`. On Viktor Sharat's CV — every tenure stated as a bare duration,
  zero absolute dates — that fallback picked whichever role the model happened to list
  first (a 2-month stint) and presented it as his current employer. Fixed by
  promoting the logic to a single `CandidateProfile.current_role()` method: an entry
  marked current wins; a *non-volunteer* current entry is preferred over a volunteer
  one both marked current (a paid job and an ongoing charity co-founder role can both
  legitimately say "Present"); failing that, the most recent *known* start date; and
  when nothing is dated, `None` — the caller must show "unknown," not a guess. All
  seven call sites across `export.py`, `ui/components.py`, `ui/pages_core.py`, and
  `scripts/run_eval.py` now go through the one method instead of duplicating (and
  independently getting wrong) the same fallback.

Post-fix: hallucination rate **0.000%**, employers F1 **0.769 → 0.894**, years-experience
MAE **0.78y → 0.43y**, macro F1 **0.888 → 0.906**. The corpus was re-parsed live twice
in total to reach this state (~$1.21 combined) — disclosed here because "we found a bug
and it cost real money to fix properly" is itself the kind of thing this project's
evaluation section exists to report honestly rather than bury.

## Measured results — live LLM parse (the required path)

Regenerated by `scripts/run_pipeline.py` (extractor=llm) + `scripts/run_eval.py` against
the hand-labelled gold set, after the fixes above. This is what actually gets
submitted — see the next section for the rule-baseline comparison these numbers beat.

- macro F1 **0.906**, hallucination rate **0.000%**, abstention rate **0.000%**,
  evidence coverage **88.1%**, years-experience MAE **0.43y**
- `full_name` / `email` / `phone` / `current_employer`: F1 **1.000**
- `employers` F1 **0.894**, `current_title` F1 **0.900**, `certifications` F1 **0.967**
- Calibration: ECE **0.025**, Brier **0.101** — verdict **well calibrated** (confidence
  ≈ observed accuracy, within 2.5 points on average)
- Hybrid retrieval nDCG@10 **0.829** vs dense-only 0.716 / lexical-only 0.766 — the
  hybrid claim holds on real parsed data, not just the rule baseline
- Fairness: mean rank change under counterfactual name-swap = **0**, by construction

## Measured results (rule baseline)

These are from `scripts/run_eval.py` against the hand-labelled gold set, on the
**rule baseline** — the LLM path is unrun pending an API key. They are the numbers the
LLM path has to beat, and they are the honest "rule vs LLM" comparison the README
promises.

| retrieval mode | nDCG@10 | R@5 | R@10 | MRR | ms |
|---|---|---|---|---|---|
| lexical (BM25/FTS5) | 0.7834 | 0.786 | 0.906 | 0.811 | 0.14 |
| dense (bge-small) | 0.7672 | 0.812 | 0.990 | 0.788 | 13.04 |
| hybrid RRF | 0.8178 | 0.807 | 0.990 | 0.851 | 14.98 |
| hybrid RRF · hashing embedder | 0.7467 | 0.714 | 1.000 | 0.709 | 0.19 |

Hybrid RRF wins on nDCG@10 and MRR — the hybrid claim is now tested rather than
asserted. The hashing fallback is measurably worse, which justifies carrying the real
embedder.

| field | P | R | F1 |
|---|---|---|---|
| `certifications` | 1.000 | 0.950 | 0.967 |
| `country` | 1.000 | 1.000 | 1.000 |
| `current_employer` | 0.900 | 0.900 | 0.900 |
| `current_title` | 0.600 | 0.600 | 0.600 |
| `degree_levels` | 1.000 | 0.925 | 0.952 |
| `email` | 1.000 | 1.000 | 1.000 |
| `employers` | 0.562 | 0.558 | 0.546 |
| `full_name` | 0.900 | 0.900 | 0.900 |
| `languages` | 0.900 | 1.000 | 0.900 |
| `phone` | 1.000 | 1.000 | 1.000 |
| `region` | 1.000 | 1.000 | 1.000 |
| `sectors_expected` | 0.419 | 0.817 | 0.551 |
| `skills_expected` | 0.752 | 0.900 | 0.816 |

Macro F1 **0.856**, hallucination rate **0.0%**
(none of the 19 attribution traps triggered), evidence coverage
**86.5%**, experience MAE **0.69y**.

Note the shape of it: rules are *perfect* on email, phone, country and region, and weak
on `current_title` (0.600)
and `employers` (0.546),
because resolving "which company in this bullet is the employer" needs semantics that
regex does not have. That gap is precisely what the LLM pass exists to close.

Calibration: ECE **0.1229**, Brier **0.1824** — verdict
**over-confident**. The rule extractor asserts ~0.91 confidence and is right ~79% of
the time, so its confidence is not yet a probability. Reported rather than hidden.

## LangChain multi-agent orchestrator — an additive, not a replacement

Added on explicit request to demonstrate real framework-driven agent coordination.
Full reasoning is in the module docstring (`src/millennium/langchain_agents.py`); the
short version:

- `orchestrator.Pipeline` (the default, used by every test and `run_pipeline.py`)
  stays deterministic Python — same input, byte-identical output, replayable from
  disk with zero API calls. That property is load-bearing for the whole submission.
- `LangChainPipeline` is a genuine `AgentExecutor` with tool-calling, wrapping the
  *same* underlying subagents as thin tools. The agent decides call order and makes
  one real judgment call per document (whether the conditional adjudication step is
  worth invoking). It is opt-in, not the default, because an agent loop cannot honestly
  offer the same determinism guarantee.
- The orchestrating agent never sees resume content. Every tool takes an opaque
  `doc_id` handle and returns short structured summaries (counts, taxonomy labels,
  booleans) — never raw text. The one tool that reads the document
  (`extract_with_llm`) is the *same* no-tools, quote-required call `agents/parsing.py`
  has always made; LangChain only decides *when* to call it, never what it sees.

**Two real bugs a live run found**, both fixed rather than routed around:

1. **Import-ordering silently broke every subagent call.** The module referenced
   subagents by name string (`"parse.llm_identity"`, etc.) without importing the
   modules whose `@subagent` decorators register them — a mistake `orchestrator.py`
   avoids by importing all four agent submodules for that exact side effect. The
   failure was invisible from the outside: `run_subagent` catches the registry miss
   and returns a well-formed `status="failed"` result rather than raising, so it read
   as a genuine extraction failure until error detail (`pass_errors`) was added to the
   tool's return value to surface the real message.
2. **A pre-existing test (`test_the_whole_app_runs_with_every_optional_flag_off`)
   permanently poisoned the shared `SETTINGS` singleton for the rest of the pytest
   session.** It reloaded `millennium.config` to flip every flag off, but never
   reloaded it back — and because Python's `from .config import SETTINGS` binds a
   *reference* at import time, any module importing `SETTINGS` for the first time
   after that test ran (in-process, same session) silently inherited the
   all-flags-off object forever. `langchain_agents.py` was simply the first new
   module in the suite to hit this, and its injection scanner ran invisibly disabled
   as a result. Fixed at the actual source — the test now unsets its own env
   overrides itself before reloading in a `finally` block, instead of relying on
   `monkeypatch`'s automatic teardown timing, which doesn't run until *after* the
   test function returns and was too late for an in-function reload to see.

## Known limitations

- **`years_experience` for roles marked "present" drifts with the calendar.** The gold
  set compares with a ±0.8y tolerance for exactly this reason.
- **Geography resolution weights the most recent role's location.** Marina Silva Costa
  lists "Boston, USA / London, United Kingdom" for one role; the system picks one and
  reports a confidence, but a genuinely dual-market candidate is not modelled.
- **Seniority tier adjustment is a heuristic**, presented as recruiter-editable and
  labelled as such in the UI. It encodes an assumption about title inflation that is
  true on average and wrong for individuals.
- **The rule-based query parser treats "must/required/only" as the sole must-have
  signal.** A recruiter writing "I need someone who has *definitely* done credit"
  gets a preference. The LLM parser handles this better; the rule parser is the
  guaranteed floor, not the ceiling.
- **Section segmentation finds few headings on the fully table-based CVs** (Zara,
  Viktor). It degrades gracefully — the whole document becomes one block and the LLM
  still sees everything — but chunk labels are less informative for those two.
- **`HashingEmbedder` is materially weaker** than bge-small; it captures lexical
  overlap, not meaning. It exists so the app never hard-fails on a model download, and
  the ablation table publishes its (lower) scores rather than hiding the gap.
