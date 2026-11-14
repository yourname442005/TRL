# Milestone 2 Evidence Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:test-driven-development (red-green per step)
> and superpowers:verification-before-completion (gate per task). Steps use checkbox (`- [ ]`) syntax.
> **No git commits** (user rule). **Milestone 1 is frozen:** its 75 tests and all M1 files must stay
> unchanged and green — run the full suite after every task.

**Goal:** Offline evidence-driven TRL assessment pipeline: Document → EvidenceSet → TRLAssessment →
GapAnalysis, explainable from evidence records, with abstention when evidence is insufficient.

**Architecture:** New packages `document/`, `assessment/`, `pipeline.py` plus filling the existing empty
`evidence/`, `extraction/`, `gap_analysis/` skeletons. stdlib dataclasses for schemas, JSON for criteria
and pipeline config, PyMuPDF (already declared) for PDF. M1 modules imported read-only.

**Tech Stack:** Python 3.11, PyMuPDF 2.0 (`import pymupdf`), stdlib (dataclasses/json/hashlib/re/pathlib),
pytest, ruff. **No new dependencies.**

**Design spec:** `docs/superpowers/specs/2026-11-12-milestone2-design.md` (field-level authority).

---

## File map (create unless marked Modify)

```
configs/trl_criteria_v1.json                 # versioned criteria content (Task 4)
configs/pipeline_v1.json                     # extraction+aggregation knobs (Task 6, keys for 8)
src/trl_intelligence/document/__init__.py
src/trl_intelligence/document/models.py      # Document/Page/TextBlock/Section, errors (Task 1)
src/trl_intelligence/document/loaders.py     # pdf/txt/md loaders (Task 2)
src/trl_intelligence/document/normalize.py   # headers/footers, page numbers, sections (Task 3)
src/trl_intelligence/assessment/__init__.py
src/trl_intelligence/assessment/criteria.py  # CriteriaRegistry loader/validator (Task 4)
src/trl_intelligence/assessment/models.py    # TRLAssessment/CriterionStatus/GapItem dataclasses (Tasks 5,8)
src/trl_intelligence/evidence/models.py      # EvidenceItem/EvidenceSet/enums (Task 5)
src/trl_intelligence/extraction/base.py      # EvidenceExtractor protocol (Task 6)
src/trl_intelligence/extraction/candidates.py
src/trl_intelligence/extraction/detect.py
src/trl_intelligence/extraction/match.py     # (Task 7)
src/trl_intelligence/extraction/score.py
src/trl_intelligence/extraction/records.py
src/trl_intelligence/extraction/pipeline.py  # DeterministicExtractor (Tasks 6–7)
src/trl_intelligence/assessment/aggregate.py # policy v1 (Task 8)
src/trl_intelligence/gap_analysis/analyze.py # GapAnalysis (Task 9)
src/trl_intelligence/pipeline.py             # assess_document core API (Task 10)
src/trl_intelligence/prediction/signals.py   # ModelSignal adapter (Task 10) — new file, M1 untouched
scripts/assess.py                            # CLI (Task 10)
tests/test_document_models.py                # Task 1
tests/test_document_loaders.py               # Task 2
tests/test_document_normalize.py             # Task 3
tests/test_criteria.py                       # Task 4
tests/test_evidence_schema.py                # Task 5
tests/test_extraction_candidates.py          # Task 6
tests/test_extraction_match_score.py         # Task 7
tests/test_aggregation.py                    # Task 8
tests/test_gap_analysis.py                   # Task 9
tests/test_pipeline.py                       # Task 10
tests/test_baseline_signal.py                # Task 10
docs/milestone2_assessment.md                # final report (written during Task 10 verification)
```

Frozen (never modified): all M1 files listed in spec §1. M1 suite command stays:
`.venv/bin/python -m pytest -q` → 75 passed; `.venv/bin/ruff check src tests scripts` → exit 0.

---

### Task 1: Document schema

**Files:** Create `src/trl_intelligence/document/{__init__,models}.py`; Test `tests/test_document_models.py`

- [ ] **Step 1 (RED):** write tests:

```python
from trl_intelligence.document.models import Document, Page, TextBlock, Section, DocumentLoadError

def test_textblock_fields_and_offsets():
    b = TextBlock(block_id="p1:b00", kind="paragraph", text="hello world", start_char=0, end_char=11)
    assert (b.start_char, b.end_char) == (0, 11)

def test_document_requires_matching_pages_and_sections():
    doc = Document(document_id="d1", source_path="x.txt", source_type="txt", title=None,
                   metadata={}, pages=[Page(page_number=1, text="abc", blocks=[])],
                   sections=[Section(section_id="s0", heading="body", level=0, path=["body"],
                                     page_start=1, page_end=1, block_ids=[])],
                   warnings=[])
    assert doc.page_count == 1 and doc.text == "abc"

def test_page_numbers_are_one_based():
    with pytest.raises(ValueError):
        Page(page_number=0, text="x", blocks=[])

def test_load_error_carries_reason():
    err = DocumentLoadError("bad pdf", source_path="a.pdf")
    assert "bad pdf" in str(err) and err.source_path == "a.pdf"
```

- [ ] **Step 2:** `pytest tests/test_document_models.py -q` → 5+ errors (ImportError) = correct RED.
- [ ] **Step 3 (GREEN):** implement dataclasses with `page_count`, `text` (joined page texts),
  validation (`page_number >= 1`, `source_type in {"pdf","txt","md"}`), `DocumentLoadError(Exception)`
  with `source_path`. `to_dict()` with stable key order (used later by pipeline).
- [ ] **Step 4:** `pytest tests/test_document_models.py -q` → PASS; full suite → 75+new still green.

### Task 2: Document loaders (txt, md, pdf)

**Files:** Create `src/trl_intelligence/document/loaders.py`; Test `tests/test_document_loaders.py`

- [ ] **RED tests:**
  - `load_document` dispatches: `.txt`→txt, `.md`/`.markdown`→md, `.pdf`→pdf, `.csv`→`UnsupportedFormatError`
  - empty txt (0 bytes) → `warnings == ["empty_document"]`, `pages` has 1 page, `text == ""`
  - txt with blank-line paragraphs → blocks split, `block_id`s `p1:b00`, `p1:b01`, offsets contiguous
  - md headings: `# A` / `## B` → heading blocks with `kind="heading"` (levels parsed in Task 3 sections;
    loader only marks kind + keeps text)
  - pdf: generate fixture at runtime (no binary committed):
    `doc = pymupdf.open(); [doc.new_page().insert_text((72, 72+20*i), line) for …]; doc.save(p)`
    → `load_document(p)` returns `source_type=="pdf"`, 2 pages, page boundaries preserved
    (text of page 1 not concatenated into page 2)
  - image-only pdf (page with no text): `warnings` contains `"no_extractable_text"`
  - malformed pdf (file containing `b"not a pdf"`): raises `DocumentLoadError`
- [ ] **GREEN:** implement `load_document(path, document_id=None)` + `_load_pdf/_load_text/_load_markdown`.
  `document_id = sha1(file_bytes).hexdigest()[:16]` when not given. PDF via `import pymupdf`
  (never `import fitz` — deprecated warning). Blocks from `page.get_text("blocks")`, cleaned with M1
  `clean_text` (read-only import).
- [ ] **Verify:** `pytest tests/test_document_loaders.py -q` PASS; full suite green; ruff exit 0.

### Task 3: Normalization — headers/footers, page numbers, sections

**Files:** Create `src/trl_intelligence/document/normalize.py`; Test `tests/test_document_normalize.py`

- [ ] **RED tests:**
  - repeated header: 5-page synthetic pdf (or Document built directly) with identical top line
    `ACME CORP PROPRIETARY` on every page → after `normalize(doc)`: line gone from every page,
    `metadata["removed_repeated_lines"] == 5`
  - a line appearing on only 1 of 5 pages is NOT stripped
  - page-number line `3` on page 3 → removed, `metadata["removed_page_number_lines"] == 1`
  - section detection from md headings: `# Top`, `## Sub`, body → sections `["Top"]`, `["Top","Sub"]`
    with `page_start/page_end` and `block_ids` populated
  - pdf ALL-CAPS heading line followed by blank line → heading section; fallback single `"body"` section
    for plain text with no headings
  - `normalize` is idempotent-ish: running twice does not double-count removals
- [ ] **GREEN:** `normalize(document: Document) -> Document` (returns new/updated Document):
  strip repeated lines (normalized-line frequency across pages ≥ max(3, 60% of pages) within top/bottom
  10% for pdf; first/last line for multi-page text), strip page-number lines, rebuild blocks with
  adjusted offsets, detect sections, record metadata counters.
- [ ] **Verify:** `pytest tests/test_document_normalize.py -q` PASS; full suite; ruff.

### Task 4: TRL criteria configuration

**Files:** Create `configs/trl_criteria_v1.json`, `src/trl_intelligence/assessment/{__init__,criteria}.py`;
Test `tests/test_criteria.py`

- [ ] **RED tests:**
  - loader returns `CriteriaRegistry` with `criteria_set_id == "hackathon_trl_v1"`,
    `levels` covering trl 1..9 exactly, every level has ≥1 criterion with non-empty
    `match_hints`, `guidance`, `evidence_types`, `min_strength in {weak,moderate,strong}`
  - all `criterion_id` unique; pattern `T{1-9}.C{n}`
  - `strength_map` covers all 8 evidence types; `mention → weak`
  - corrupted file (duplicate criterion_id) → `CriteriaValidationError`
  - `registry.level(5)` / `registry.criterion("T5.C1")` lookups work; unknown id → KeyError
  - canonical sha1: `registry.content_hash()` stable across loads of same file
- [ ] **GREEN:** author full criteria JSON: **2–4 criteria per level, levels 1–9** (content below),
  then `criteria.py` with `CriteriaRegistry` (frozen dataclasses), `from_file(path)`, validation.

Criteria content (author exactly; adjust wording only if a test requires):

| Lvl | Criteria (id: description · hints · guidance) |
|-----|-----------------------------------------------|
| 1 | T1.C1 principles observed/reported · "basic principles","proof of concept","fundamental research","scientific basis" · report of observed principles |
| 2 | T2.C1 application concept formulated · "application concept","concept formulated","trade study" · written concept; T2.C2 analytical/experimental proof of concept · "proof of concept","analytical","experimentally verified" |
| 3 | T3.C1 analytical critical function/characteristic proof · "analytical","critical function","characteristic" · analysis report; T3.C2 laboratory demonstration · "laboratory demonstration","bench test","component level" · lab test data |
| 4 | T4.C1 component/breadboard validation in lab · "component validation","breadboard","laboratory test","verified in lab" · lab test report w/ acceptance criteria; T4.C2 integration risk identified · "integration risk","risk assessment" |
| 5 | T5.C1 component/breadboard validation in relevant environment · "relevant environment","component test","simulated environment" · test report in relevant environment; T5.C2 fidelity of environment justified · "representative environment","fidelity","justified" · environment rationale; T5.C3 (optional) supporting analysis |
| 6 | T6.C1 system/subsystem prototype demonstration in relevant environment · "system prototype","prototype demonstration","relevant environment" · prototype demo report w/ results; T6.C2 critical operations documented · "operational","mission profile" |
| 7 | T7.C1 system prototype demonstration in flight environment · "flight demonstration","flight test","airborne" · flight test report; T7.C2 flight readiness/review documented · "flight readiness","readiness review" |
| 8 | T8.C1 actual system completed and flight qualified · "flight qualified","qualification test","airworthiness" · qualification report; T8.C2 first unit produced/tested · "first unit","production unit" |
| 9 | T9.C1 actual system proven successful in operation · "operational success","mission success","in service","deployed operationally" · operational record; T9.C2 sustained operations documented · "sustained","ongoing operations" |

(Each criterion also gets `min_strength` — levels 1–4 `moderate`, levels 5–9 `strong` for demonstration
classes — `weight` 1.0, `evidence_types` from the type set, `guidance` + `example_artifacts` concrete:
e.g. T5.C1 guidance "Provide component test report executed in a relevant environment, including
acceptance criteria, conditions, and results."; example_artifacts ["test plan","test report","test data"].)

- [ ] **Verify:** `pytest tests/test_criteria.py -q` PASS; full suite; ruff.

### Task 5: Evidence schema

**Files:** Create `src/trl_intelligence/evidence/models.py` (+ `__init__` re-exports);
`src/trl_intelligence/assessment/models.py` (CriterionStatus only this task);
Test `tests/test_evidence_schema.py`

- [ ] **RED tests:** enum value sets (`POLARITIES = supports|contradicts|insufficient`,
  `EVIDENCE_TYPES` = 8 types, `STRENGTHS = weak|moderate|strong`); `EvidenceItem` rejects
  `trl_level` outside 1..9 and `confidence` outside [0,1] (ValueError); `evidence_id` determinism —
  two items with identical location+criteria fields → equal ids; different criterion → different id;
  `EvidenceSet.items` sorted by `(trl_level, criterion_id, page_number, start_char)`;
  `quote` clipped at 300 chars with trailing "…"; `EvidenceSet.to_dict()` JSON-serializable.
- [ ] **GREEN:** implement dataclasses + `make_evidence_id()` (sha1[:16]) + `EvidenceSet.add/sort/to_dict`.
- [ ] **Verify:** `pytest tests/test_evidence_schema.py -q` PASS; full suite; ruff.

### Task 6: Candidate segmentation + technical detection (+ config file)

**Files:** Create `configs/pipeline_v1.json`, `extraction/{__init__,base,candidates,detect}.py`;
Test `tests/test_extraction_candidates.py`

- [ ] **RED tests:**
  - `configs/pipeline_v1.json` exists with `schema_version`, `extraction.*`,
    `aggregation.{level_threshold,criterion_min_confidence,min_evidence_to_assess}`,
    `strength_map`, `signal.{enabled,max_chars,variant}`
  - sentence split: paragraph with 3 sentences → 3 candidates with correct in-block offsets;
    "e.g. Dr. Smith" not split after abbreviation (pinned cases)
  - candidates below `min_candidate_chars` dropped; per-section cap `max_candidates_per_section`
    respected with stable ordering
  - `detect` keeps "The prototype was demonstrated in a relevant environment." and drops
    "The report is organized as follows." (cue list-driven)
  - `base.EvidenceExtractor` protocol: `DeterministicExtractor` satisfies
    `isinstance(x, EvidenceExtractor)` via `@runtime_checkable` Protocol
- [ ] **GREEN:** implement with cue/substring lists from `pipeline_v1.json` (`extraction.technical_cues`,
  `extraction.prospective_cues`, `extraction.negation_cues`, …). `load_pipeline_config(path)`.
- [ ] **Verify:** `pytest tests/test_extraction_candidates.py -q` PASS; full suite; ruff.

### Task 7: Criterion matching, scoring, evidence records

**Files:** Create `extraction/{match,score,records,pipeline}.py`; Test `tests/test_extraction_match_score.py`

- [ ] **RED tests** (build tiny Document + real criteria registry from Task 4):
  - match: sentence containing "demonstrated in a relevant environment" matches T5.C1 (hint hit
    recorded); sentence with no hints matches nothing
  - score → supports: "The system was demonstrated in a relevant environment in 2024 and the
    results show full functionality." → `polarity=="supports"`, type `demonstration` (or `test`),
    `strength=="strong"` (via strength_map), `confidence ≥ 0.5`
  - score → insufficient: "TRL 5 will be demonstrated next year." / bare "relevant environment"
    mention without assertion → `polarity=="insufficient"`
  - score → contradicts: "The component failed the relevant environment test." /
    "has not yet been demonstrated" → `polarity=="contradicts"`
  - record: evidence_id/quote/normalized fields populated; extraction_method
    `"deterministic_v1"`; EvidenceSet carries `criteria_set_id`
  - `DeterministicExtractor().extract(document, registry, config)` returns EvidenceSet end-to-end
    on a 2-section synthetic document (integration within the package)
- [ ] **GREEN:** implement match (substring, case-insensitive, all hints for a criterion grouped),
  score (polarity precedence: contradicts cues > prospective/insufficient > supports assertion cues;
  confidence formula pinned: `base{supports:0.7, insufficient:0.4, contradicts:0.6} +
  0.05·(extra_hint_hits)`, clamp [0,1]), records, pipeline composition.
- [ ] **Verify:** `pytest tests/test_extraction_match_score.py -q` PASS; full suite; ruff.

### Task 8: Aggregation policy

**Files:** Create `assessment/{aggregate.py}` + extend `assessment/models.py` (TRLAssessment);
Test `tests/test_aggregation.py`

- [ ] **RED tests** (hand-built EvidenceSets, no files needed):
  - criterion gate: supporting item strength=strong, conf=0.8 → `satisfied`; weak only → `weak`;
    contradicting only → `contradicted`; none → `missing`
  - level score: 2/3 satisfied, weights 1.0 → `support == approx(2/3)`
  - monotone walk: supports engineered so levels 1–5 pass, 6 fails → `predicted_trl == 5`,
    `next_trl == 6`, `support_scores` has all 9 keys as strings
  - abstention: zero evidence → `status=="insufficient_evidence"`, `predicted_trl is None`,
    confidence is not None (computed from support(1))
  - confidence formula pinned with hand-computed expected value (exact float approx)
  - signal agreement: ModelSignal with `predicted_trl == 5` vs evidence 5 → confidence higher than
    same fixture with signal 9 (monotonicity of agreement term)
  - explainability: every `supported_criteria` id has ≥1 backing evidence id in `evidence`
  - contradiction surfaces: contradicted criterion id appears in `contradictory_evidence` (as evidence ids)
- [ ] **GREEN:** `aggregate(registry, evidence_set, config, signals) -> TRLAssessment` implementing
  policy v1 exactly as spec §7 (rule list, formula, bands, monotone walk, abstention).
- [ ] **Verify:** `pytest tests/test_aggregation.py -q` PASS; full suite; ruff.

### Task 9: Gap analysis

**Files:** Create `gap_analysis/analyze.py` (+ `__init__` re-export); Test `tests/test_gap_analysis.py`

- [ ] **RED tests:**
  - from an assessment with `predicted_trl=5`, `next_trl=6`: `gaps` contain exactly the unsatisfied
    criteria of level 6; each has `requirement` from criteria description, `blocker is True`,
    `recommendation.specifics ==` that criterion's `guidance` (non-generic assertion: no gap without
    a criterion-specific `specifics`), `evidence_types_acceptable` non-empty
  - `weak_evidence` gap carries `partial_evidence_ids` non-empty when weak hits exist
  - lower-level `contradicted` gap ranks before higher-level `missing` in `priority_order`
    (trl asc, then weight desc)
  - abstain case (`predicted_trl=None`): `next_trl is None`, gaps = level-1 criteria, blockers true
  - `demonstrated` lists satisfied criteria with backing evidence ids
- [ ] **GREEN:** `analyze(assessment, registry, evidence_set) -> GapAnalysis` (pure function).
- [ ] **Verify:** `pytest tests/test_gap_analysis.py -q` PASS; full suite; ruff.

### Task 10: End-to-end pipeline, model-signal adapter, CLI, final gate

**Files:** Create `pipeline.py`, `prediction/signals.py`, `scripts/assess.py`,
`tests/test_pipeline.py`, `tests/test_baseline_signal.py`, `docs/milestone2_assessment.md`

- [ ] **RED tests `test_pipeline.py`:**
  - `assess_document(AssessmentRequest(text=..., source_type="txt"))` returns `AssessmentResult`
    with all four components populated and `status=="assessed"` on a fixture rich in demo language
  - `status=="insufficient_evidence"` for a trivial empty/one-liner document (abstention e2e)
  - `status=="failed"` + `error` for a malformed pdf path (no exception escapes)
  - `to_json()` round-trips: `json.loads` works; stable across two calls (determinism)
  - result JSON contains keys `predicted_trl`, `confidence`, `supported_criteria`,
    `missing_criteria`, `contradictory_evidence`, `next_trl`, `next_trl_gaps`, `evidence`
  - PDF e2e: runtime-generated 2-page pdf with section heading + demo sentences →
    `page_number` on evidence items reflects true page
- [ ] **RED tests `test_baseline_signal.py`:**
  - `baseline_signal(model_path, document, variant="full")` with a synthetic B1 fitted in-test
    (build_b1 + tiny corpus, saved via `save_model` to tmp_path — read-only M1 imports) returns
    `ModelSignal(source=..., predicted_trl in 1..9)`
  - variant `"masked"` applies `mask_trl` to signal text only (spy via a text containing "TRL 5")
  - pipeline with `signals=[sig]` records `signals` on assessment and shifts confidence vs no signal
    (agreement term) — real-joblib variant of this test uses `pytest.mark.skipif(not checkpoint.exists())`
- [ ] **GREEN:** `pipeline.py` (AssessmentRequest/Result, `assess_document`, exception mapping),
  `prediction/signals.py` (`ModelSignal`, `baseline_signal`), `scripts/assess.py`
  (argparse: `--doc` | `--text`, `--config`, `--criteria`, `--signal`; exit 0/2).
  Write `docs/milestone2_assessment.md` report (actual run output, limitations per spec §10).
- [ ] **Verify (final gate — all must pass before claiming completion):**

```bash
.venv/bin/python -m pytest -q          # M1's 75 + all new tests, 0 failures
.venv/bin/ruff check src tests scripts # exit 0
.venv/bin/python scripts/assess.py --text "<rich demo document>"   # status assessed, exit 0
.venv/bin/python scripts/infer.py --model models/checkpoints/baseline_v1/b2_masked.joblib --text "x"  # M1 untouched
/Library/Developer/CommandLineTools/usr/bin/git status --short     # no M1 file modified (only ?? new files + docs)
/Library/Developer/CommandLineTools/usr/bin/git diff --stat        # empty except pre-existing .gitignore M
```

---

## Task ordering rationale

1→2→3 (schema before loaders before normalization) · 4 (criteria independent, needed by 7–9) ·
5 (evidence schema needed by 7) · 6→7 (extraction stages) · 8 (aggregation needs 4+5+7) ·
9 (gaps need 8) · 10 (pipeline wires everything; signal adapter last because optional).

After each task: full suite + ruff (M1 green preserved). No commits at any point.
