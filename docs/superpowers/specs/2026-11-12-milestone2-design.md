# Milestone 2: Evidence-Driven TRL Assessment Pipeline — Design Spec

**Status:** DRAFT for approval (Stage 10 gate — no implementation until approved)
**Date:** 2026-11-12
**Frozen constraint:** Milestone 1 stays byte-identical and green (75 tests, ruff clean).

---

## 1. Context and constraints

- Hackathon scope: unstructured technical documents (PDF/TXT/MD), not short project abstracts.
- Fully offline: no network at runtime; deterministic, seed-free core (stable ordering, stable IDs).
- No new dependencies. No frontend/API this milestone. No OCR this milestone (non-OCR path correct first).
- Milestone 1 files are frozen: `models/baselines.py`, `preprocessing/text.py`, `evaluation/*`,
  `ingestion/techport.py`, `prediction/artifacts.py`, `scripts/run_baseline.py`, `scripts/infer.py`,
  `configs/baseline_v1.json`, all M1 tests. M2 code **imports** from them read-only; never edits them.
- No commits. Every behavior change lands test-first (TDD).

## 2. Stage 1 — Frozen baseline findings

### 2.1 Current architecture (M1)

```
data/raw/techport/pages/offset_*.json (43 pages, 21,161 records, provenance: SOURCE.md)
  → preprocessing/text.py   clean_text / mask_trl
  → ingestion/techport.py   load_usable_dataset → DataFrame (project_id, title, text, trl, group_id, …)
  → evaluation/splits.py    group-aware 70/15/15 (sha1-of-text groups), class dist + missing classes
  → models/baselines.py     B1 TF-IDF+balanced multinomial LR · B2 8-threshold cumulative ordinal LR
                            · B3 TF-IDF→TruncatedSVD→GradientBoosting  (+ build_from_config)
  → prediction/artifacts.py joblib save/load/predict_texts (offline)
  → scripts/run_baseline.py train/eval (val-only selection, test once) → metrics.json + report
  → scripts/infer.py        single-text offline inference
Configs: configs/baseline_v1.json   Results: experiments/results/baseline_v1/*.json
Models:  models/checkpoints/baseline_v1/{b1,b2,b3}_{full,masked}.joblib (gitignored)
Tests:   75 (8 files) — FROZEN, must remain green and unchanged.
Empty skeleton packages awaiting M2: evidence/, extraction/, gap_analysis/, features/ (0-byte __init__).
```

Note: `configs/config.yaml` does not exist; M1 uses JSON (stdlib) — M2 continues with JSON.

### 2.2 Reusable components (import read-only)

| Component | Reuse in M2 |
|---|---|
| `preprocessing.text.clean_text` | normalize text blocks (tags/entities/whitespace) |
| `preprocessing.text.mask_trl` | optional masking **for the frozen-model signal only** (never for evidence extraction) |
| `prediction.artifacts.load_model` | load frozen joblibs for the model-signal adapter |
| `models.baselines.build_b1` etc. | test-time synthetic models for adapter tests (no dependency on gitignored joblibs) |
| `evaluation.metrics.classification_metrics` | later: evaluate M2 against labeled sets (not in core path) |
| Config pattern (JSON, stdlib), script pattern (`sys.path` insert + argparse), TDD/ruff/pytest conventions | house style for all M2 files |

### 2.3 Data structures to reuse / mirror

- TRL label space {1..9} and its public semantics (NASA/DoD definitions, documented in
  `data/raw/techport/population_report.json`).
- Result-as-JSON-dict conventions (`metrics.json` style: one auditable artifact per run).
- Stable-hash idea from `group_id` (sha1) → reused for `document_id` and `evidence_id` determinism.
- Frozen metrics/split reports remain the M1 evidence of record; M2 never rewrites them.

### 2.4 M1 assumptions that must NOT leak into the document pipeline

1. **Input is one flat cleaned string** (`text` column). M2 input = multi-page, multi-section documents.
2. **Dataset filters** (`trlCurrent` 1–9, `min_chars=200`) are TechPort dataset rules, not document validity.
3. **Group splitting / sha1-duplicate groups** — irrelevant to single-document assessment.
4. **Labels exist at inference** — M2 is label-free at prediction time.
5. **`mask_trl` by default** — M1 masks to measure leakage; M2 evidence extraction must see real TRL
   mentions (they are evidence). Masking applies only inside the optional model-signal adapter.
6. **TF-IDF abstract language** — frozen models were trained on ~1-paragraph project descriptions;
   long-document domain shift is expected. The frozen classifier may be *one weak signal*, never the assessor.
7. **TechPort fields** (`projectId`, `program`, `status`) — Document schema must be source-agnostic.

### 2.5 Frozen classifier as one signal (design now, wire as optional)

```
Document → (truncate to signal.max_chars, apply mask_trl iff variant=="masked")
        → prediction.artifacts.load_model(...).predict_texts(...)   [read-only import]
        → ModelSignal{source: "b1_masked", predicted_trl, details}
        → passed into AssessmentRequest.signals
        → aggregation records signals; agreement adjusts confidence only;
          evidence-derived TRL is authoritative; signal NEVER overrides it (policy v1).
```

Default: **signal disabled** (`pipeline_v1.json → signal.enabled=false`). Tests cover adapter with a
synthetic in-test model; the real-joblib integration test skips when checkpoints are absent (gitignored).

## 3. Stage 2 — Evidence schema

Package: `src/trl_intelligence/evidence/models.py` (stdlib dataclasses, frozen where practical).

```
EvidenceItem
  document_id: str            # stable sha1 of content (or caller-provided)
  evidence_id: str            # sha1(f"{document_id}:{page}:{block}:{start}:{criterion_id}")[:16]
  page_number: int | None     # 1-based; None for single-page text
  section_heading: str | None
  block_id: str               # e.g. "p3:b07"
  start_char: int             # offsets within block text
  end_char: int
  quote: str                  # exact source span, clipped to 300 chars with "…"
  normalized: str             # whitespace-collapsed, html-unescaped span text
  criterion_id: str           # e.g. "T5.C2" — joins to criteria registry
  trl_level: int              # 1..9 the criterion belongs to
  evidence_type: str          # report|analysis|simulation|design|test|demonstration|operation|mention
  polarity: str               # supports | contradicts | insufficient
  strength: str               # weak | moderate | strong  (derived from evidence_type via strength_map)
  confidence: float           # 0..1, rule-scored
  extraction_method: str      # e.g. "deterministic_v1"
  extractor_version: str      # e.g. "1.0.0"

EvidenceSet
  document_id: str
  items: list[EvidenceItem]   # deterministically sorted: (trl_level, criterion_id, page, start_char)
  extraction_config: dict     # snapshot of knobs used
  criteria_set_id: str        # which criteria version matched against
  warnings: list[str]
```

Answers the two required questions:

- **"What evidence caused this TRL?"** → `TRLAssessment.supported_criteria` → filter
  `EvidenceSet.items` by those `criterion_id`s and `polarity == "supports"`; each item carries
  page/section/span/quote → full traceability.
- **"What evidence is missing for the next TRL?"** → `GapAnalysis.gaps` → per missing criterion of
  `next_trl`: requirement text, status (`missing`/`weak_evidence`/`contradicted`), any partial hits,
  and a criterion-specific recommendation.

No ontology beyond type/polarity/strength enums above.

## 4. Stage 3 — TRL 1–9 criteria (machine-readable, versioned)

File: `configs/trl_criteria_v1.json` (JSON — matches M1 no-YAML rule). Loaded by
`src/trl_intelligence/assessment/criteria.py` into an immutable `CriteriaRegistry`
(dataclass, `from_dict` + validation: unique ids, trl 1..9, non-empty criteria per level).
Prediction/aggregation code reads the registry; **no criterion text lives in Python code**.

```json
{
  "schema_version": "1.0",
  "criteria_set_id": "hackathon_trl_v1",
  "source_note": "Adapted from public NASA/DoD TRL definitions for hackathon use; not a certification standard.",
  "strength_map": {"mention": "weak", "design": "weak", "report": "moderate", "analysis": "moderate",
                   "simulation": "moderate", "test": "strong", "demonstration": "strong", "operation": "strong"},
  "levels": [
    {
      "trl": 1,
      "name": "Basic principles observed and reported",
      "criteria": [
        {
          "criterion_id": "T1.C1",
          "description": "Basic scientific principles of the technology are observed and reported.",
          "evidence_types": ["report", "analysis"],
          "min_strength": "moderate",
          "weight": 1.0,
          "match_hints": ["basic principles", "proof of concept", "fundamental research",
                          "laboratory observation", "scientific basis"],
          "guidance": "Provide a report or paper stating the observed principles and initial results."
        }
      ]
    },
    { "trl": 2, "…": "2–4 criteria, same fields" },
    "… levels 3–9 …"
  ]
}
```

Design requirements covered:

| Requirement | Mechanism |
|---|---|
| positive evidence | `polarity == supports` items whose strength ≥ `min_strength`, confidence ≥ threshold |
| missing evidence | no such item for a criterion → `missing` |
| contradictory evidence | `polarity == contradicts` items for that criterion |
| next-level requirements | criteria of level `current+1` (gap analysis input) |
| evidence strength | `min_strength` per criterion vs item `strength` from `strength_map` |
| mention ≠ proof | `mention` maps to `weak`; weak evidence cannot satisfy a `min_strength ≥ moderate` criterion |
| versioning | `schema_version` + `criteria_set_id` + sha1 of canonical JSON, recorded on every assessment |

Hints are **case-insensitive substrings** (not regex) for v1: deterministic, safe, easy to test.
Content: 2–4 criteria per level (~25 total), authored in Task 4 of the plan (full content in plan doc).

## 5. Stage 4 — Document ingestion layer

Package: `src/trl_intelligence/document/`

```
Document
  document_id: str                 # caller-provided or sha1(file bytes)[:16]
  source_path: str
  source_type: "pdf" | "txt" | "md"
  title: str | None                # filename stem or first H1/# title
  metadata: dict                   # loader name+version, byte_size, page_count, removed_header_footer_lines, …
  pages: list[Page]                # txt/md → single page (page_number=1)
  sections: list[Section]          # built in normalize stage
  warnings: list[str]              # empty_document | no_extractable_text | possibly_scanned | …

Page
  page_number: int                 # 1-based, preserved for PDFs
  text: str                        # after header/footer/page-number strip (raw kept in blocks pre-strip audit)
  blocks: list[TextBlock]

TextBlock
  block_id: str                    # f"p{page}:b{idx:02d}"
  kind: "heading" | "paragraph" | "list_item" | "code" | "table_caption"
  text: str                        # clean_text()-normalized
  start_char: int; end_char: int   # within page.text

Section
  section_id: str; heading: str; level: int
  path: list[str]                  # e.g. ["5. Test Results", "5.2 Flight Test"]
  page_start: int; page_end: int
  block_ids: list[str]
```

Loaders (`loaders.py`): `load_document(path, document_id=None) -> Document` dispatches on suffix
`.pdf` / `.txt` / `.text` / `.md` / `.markdown`; anything else → `UnsupportedFormatError`.

- **PDF** via `import pymupdf` (modern API; declared dep, installed 2.0). `page.get_text("blocks")`
  preserves page boundaries and block boxes → TextBlocks. Font-size info is available but v1 section
  detection uses text heuristics only (deterministic, simpler tests).
- **TXT**: one page, blocks split on blank lines.
- **MD**: one page, `#`–`######` headings → heading blocks with levels; paragraphs/lists/code fences
  split by a small line scanner (no markdown library).
- **Empty** (0 bytes / no blocks): `warnings=["empty_document"]`, valid Document, zero blocks.
- **Malformed PDF**: pymupdf raises → `DocumentLoadError` (custom exception). Pipeline catches it →
  `status="failed"` result with reason. Never crashes the caller.
- **Scanned/image-only PDF**: all pages yield no text → `warnings=["no_extractable_text",
  "possibly_scanned_image_only"]`; pipeline abstains (`insufficient_evidence`). **No OCR in M2** —
  justification: no OCR engine is installed/declared; instructions require the non-OCR path correct first;
  the warning makes the gap explicit for a later milestone.
- **Repeated headers/footers** (`normalize.py`): a line (normalized: lowercase, digits/roman stripped)
  appearing in the top/bottom 10% region of ≥ max(3, 60%) of pages is stripped from blocks; count
  recorded in `metadata["removed_repeated_lines"]`.
- **Page-number lines**: standalone lines matching `^[\divxlcIVXLC]+$` equal/close to the page index
  are dropped and counted in `metadata["removed_page_number_lines"]`.
- **Very long documents**: loader builds pages incrementally (no whole-file join required); extraction
  (later) caps candidates per section (`max_candidates_per_section`). Tested with a generated 10k-line
  document (completion + counts, not a benchmark).

Section detection (after strip): MD uses `#` levels directly; PDF/TXT use heuristics — line ≤ 80 chars,
no terminal `.`/`:`, (ALL-CAPS or Title Case), followed by a blank line → heading; nesting by ALL-CAPS
(level 1) vs Title Case (level 2). Single-section fallback: `heading="body"`.

## 6. Stage 5 — Evidence extraction architecture (offline, replaceable)

Package: `src/trl_intelligence/extraction/` — five small modules, one composition class. No giant function.

```
base.py        EvidenceExtractor Protocol: extract(document, registry, config) -> EvidenceSet
               → future ML/LLM extractor implements the same protocol; assessment engine
                 consumes only EvidenceSet → swap without rewriting the engine.
candidates.py  segment: per section, paragraphs → sentences (deterministic regex splitter,
               abbreviation-safe list), drop < min_candidate_chars, cap max_candidates_per_section
               (stable order: page, block, offset) → CandidateSpan(text, page, block_id, section, start, end)
detect.py      technical-evidence detection: lexical cues (test, demonstrat, prototype, validat,
               flight, model/simulat, qualif, deploy, operational, …) + negation scan
               → keeps/discards candidates (classification stage, config-driven cue lists)
match.py       criterion matching: for each candidate × registry criterion: case-insensitive
               substring match of any match_hints → Match(criterion_id, trl, hit_hints, span)
score.py       scoring: polarity + confidence + type inference
               - supports: past/complete assertion cues ("demonstrated", "was tested", "validated in",
                 "results show") co-occurring with the hint
               - insufficient: hint present but no assertion (bare mention), or prospective cues
                 ("planned", "future work", "will be tested", "not yet")
               - contradicts: negation/failed cues near the hint ("not demonstrated", "failed",
                 "has not yet", "no test data")
               - evidence_type inferred from type cues near span (test report, simulation, …),
                 strength from registry strength_map, confidence = base by polarity/type + hint
                 density, clamped 0..1 (exact formula pinned by tests)
records.py     EvidenceItem construction (stable evidence_id hash, quote clipping, sorting),
               EvidenceSet assembly with config snapshot + criteria_set_id
pipeline.py    DeterministicExtractor(base.EvidenceExtractor): composes the five stages
```

Separation of concerns: **extraction** (spans) → **classification** (technical? type/polarity) →
(criterion **matching**) → **scoring** → **records**. Aggregation and final decision live outside
this package (`assessment/`). All cue lists and knobs live in `configs/pipeline_v1.json`
(`extraction.*` section) — not hardcoded in prediction functions.

## 7. Stage 6 — TRL aggregation logic

Package: `src/trl_intelligence/assessment/` (`criteria.py`, `aggregate.py`, `models.py`).

Policy v1 (explicit, versioned via `policy_version: "1.0"`, thresholds in `pipeline_v1.json`):

1. **Criterion status** — for each criterion of each level:
   - `satisfied` ⇔ ∃ supporting item with `strength ≥ min_strength` and `confidence ≥ criterion_min_confidence`
   - `weak` ⇔ supporting items exist but none pass the strength/confidence gate
   - `contradicted` ⇔ contradicting items exist and no satisfying support (contradictions never
     silently vanish: they surface in `contradictory_evidence`)
   - `missing` ⇔ no supporting items at all
2. **Level support score** — `support(L) = Σ weight(satisfied criteria of L) / Σ weight(all criteria of L)`
   (weak/contradicted/missing contribute 0).
3. **Monotone prediction** — `predicted = max L ∈ 1..9 such that ∀ j ≤ L: support(j) ≥ level_threshold`
   (default τ=0.6). If level 1 fails → **abstain**: `predicted_trl: null, status: "insufficient_evidence"`
   with `min_evidence_to_assess` (default 3) as an additional gate. Never forces 1–9.
4. **Confidence** (deterministic formula, pinned by tests):
   ```
   volume = min(1, log1p(n_supporting) / log1p(10))
   agreement = mean over signals of (1.0 if |signal_trl − predicted| ≤ 1 else 0.0);
               0.5 when no signals
   raw = 0.60·support(predicted) + 0.25·volume + 0.15·agreement
   penalty = 1 / (1 + 0.5·n_contradicted_criteria / max(1, n_criteria))
   confidence = clamp01(raw · penalty)     # abstain case: clamp01(support(1) · penalty)
   band: <0.4 low · <0.7 medium · else high
   ```
5. **Signals** — recorded on the assessment (`signals: [{source, predicted_trl, agreement}]`);
   they influence confidence only. Evidence-derived TRL always wins (policy v1; future: configurable weights).
6. **Explainability** — every entry in `supported_criteria` has ≥1 backing evidence id; `support_scores`
   exposes the level walk; `criteria_set_id`, `policy_version`, config snapshot recorded on the result.

Output shape (exactly the requested structure):

```json
{
  "predicted_trl": 5,
  "confidence": 0.62, "confidence_band": "medium", "status": "assessed",
  "support_scores": {"1": 1.0, "2": 1.0, "3": 0.75, "4": 0.67, "5": 0.6, "6": 0.33},
  "supported_criteria": ["T5.C1", "…"], "missing_criteria": ["T5.C3", "…"],
  "contradictory_evidence": ["ab12cd…"],
  "next_trl": 6, "next_trl_gaps": [GapItem…],
  "evidence": [EvidenceItem…],
  "signals": [], "criteria_set_id": "hackathon_trl_v1", "policy_version": "1.0",
  "document_id": "…"
}
```

## 8. Stage 7 — Gap analysis

Package: `src/trl_intelligence/gap_analysis/analyze.py` — pure function
`analyze(assessment, registry, evidence_set) -> GapAnalysis`. No re-extraction, no generic advice.

```
GapAnalysis
  document_id, current_trl (or None), next_trl (or None)
  demonstrated: [{criterion_id, evidence_ids}]          # answers Q1
  gaps: [GapItem]                                       # answers Q2–Q4
  priority_order: [criterion_id…]                        # trl asc, weight desc

GapItem
  criterion_id, trl, requirement        # criterion description
  status: "missing" | "weak_evidence" | "contradicted"
  partial_evidence_ids: [ids]           # weak/insufficient hits (shows what exists)
  blocker: bool                         # true for unsatisfied criteria of next_trl that cap progress
  recommendation:
    action: "document" | "demonstrate" | "test" | "analyze"
    specifics: str                      # from the criterion's `guidance` (never generic)
    evidence_types_acceptable: [types]  # criterion.evidence_types
    example_artifacts: [str]            # criterion.guidance-derived list in criteria JSON
```

- Q1 demonstrated → `demonstrated`; Q2 missing → gaps with `status=missing`;
  Q3 blocker → first `blocker=true` in `priority_order` (criteria of `next_trl`, plus any
  lower-level `weak/contradicted` gaps, which rank earlier by trl asc); Q4 next → recommendations.
- Every recommendation is templated from that criterion's `guidance` + `evidence_types` in the
  versioned criteria file — "conduct more testing" can never appear unless a criterion literally
  requires it, and then it names the artifact (e.g. "test report with acceptance criteria and results").
- If `predicted_trl` is None (abstain): `next_trl = None`, gaps = criteria of level 1 (what to
  produce first), blocker=true on unsatisfied level-1 criteria.

## 9. Stage 8 — End-to-end core API (HTTP/UI-independent)

`src/trl_intelligence/pipeline.py`:

```
AssessmentRequest
  path: Path | None            # or text: str + source_type
  text: str | None
  source_type: "pdf" | "txt" | "md" | None   # inferred from path suffix when path given
  document_id: str | None      # default sha1(bytes)[:16]
  config_path: Path | None     # default configs/pipeline_v1.json
  criteria_path: Path | None   # default configs/trl_criteria_v1.json
  signals: list[ModelSignal]   # default []

AssessmentResult                # one auditable object
  document: Document
  evidence_set: EvidenceSet
  assessment: TRLAssessment
  gap_analysis: GapAnalysis
  status: "assessed" | "insufficient_evidence" | "failed"
  error: str | None             # load/validation failure reason
  to_dict() / to_json()         # stable key order, JSON-safe (floats rounded to 4 dp)

assess_document(request: AssessmentRequest) -> AssessmentResult
  load → normalize → extract → aggregate → gaps   (each step isolated, exceptions → status="failed")
```

- Pure Python; imports nothing from `api/`, FastAPI, or UI.
- `scripts/assess.py` CLI mirrors `infer.py` style:
  `python scripts/assess.py --doc report.pdf [--config …] [--criteria …] [--signal model.joblib]`
  → prints `result.to_json()`; exit 0 on assessed/insufficient, 2 on failed.
- Determinism: no wall-clock timestamps in output; stable sorts everywhere; stable id hashes.

## 10. Risks and limitations (honest scope)

1. Substring/lexicon matching is heuristic — recall/precision bounded; LLM extractor later via
   `EvidenceExtractor` protocol without engine rewrite.
2. English-only cues; sentence splitter imperfect on abbreviations.
3. Criteria are an *adaptation* of public TRL definitions for the hackathon — not a certification
   standard; recorded in `source_note`.
4. Header/footer heuristic may strip legitimately repeated content (count exposed in metadata).
5. Confidence is rule-calibrated, not statistically calibrated — bands are indicative.
6. Frozen-classifier signal has abstract→long-doc domain shift; default OFF; masked variant preferred.
7. Scanned PDFs → abstain (no OCR in M2, by design).
8. Monotone level-walk policy (τ=0.6) is deliberately conservative and configurable, not ground truth.

## 11. Non-goals (M2)

Frontend, FastAPI service, PDF ingestion beyond text-layer PDFs, OCR, ML/LLM extractors, changes to
any M1 file, new dependencies, commits.
