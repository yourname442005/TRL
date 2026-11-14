# Milestone 2 — Evidence-Driven TRL Assessment

Offline, deterministic pipeline: **Document → EvidenceSet → TRLAssessment → GapAnalysis**.
No commits, no new dependencies, no OCR, no API/UI. M1 remains frozen (75 tests green).

## What was built

| Stage | Modules | Tests |
|---|---|---|
| Document schema | `document/models.py` | `test_document_models.py` |
| Loaders (pdf/txt/md) | `document/loaders.py` | `test_document_loaders.py` |
| Normalization (headers/page #s/sections) | `document/normalize.py` | `test_document_normalize.py` |
| Versioned TRL criteria (NASA/DoD/TechPort) | `configs/trl_criteria_v1.json`, `assessment/criteria.py` | `test_criteria.py` |
| Evidence schema (traceable, deterministic ids) | `evidence/models.py`, `assessment/models.py` | `test_evidence_schema.py` |
| Candidate segmentation + detection | `configs/pipeline_v1.json`, `extraction/{base,candidates,detect}.py` | `test_extraction_candidates.py` |
| Match → score → records → pipeline | `extraction/{match,score,records,pipeline}.py` | `test_extraction_match_score.py` |
| Aggregation policy v1 (monotone, abstaining) | `assessment/aggregate.py` | `test_aggregation.py` |
| Gap analysis (blockers, recommendations) | `gap_analysis/analyze.py` | `test_gap_analysis.py` |
| End-to-end API + CLI + signal adapter | `pipeline.py`, `prediction/signals.py`, `scripts/assess.py` | `test_pipeline.py`, `test_baseline_signal.py` |

## How to run

```bash
.venv/bin/python scripts/assess.py --text "<document text>"          # inline text
.venv/bin/python scripts/assess.py --doc report.pdf                  # text-layer PDF
.venv/bin/python scripts/assess.py --doc report.pdf --signal models/checkpoints/baseline_v1/b1_masked.joblib --signal-variant masked
```

Exit codes: `0` = assessed or insufficient_evidence (abstention), `2` = failed.
All knobs live in `configs/pipeline_v1.json`; criteria live in `configs/trl_criteria_v1.json`.

## Actual run output

Rich demonstration document (10 sentences covering TRL 1–5 criteria):

```bash
.venv/bin/python scripts/assess.py --text "BASIC PRINCIPLES … RELEVANT ENVIRONMENT …"
# exit 0
```

```json
{
  "status": "assessed",
  "predicted_trl": 5,
  "confidence": 0.925,
  "confidence_band": "high",
  "support_scores": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0,
                     "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0},
  "supported_criteria": ["T1.C1", "T1.C2", "T2.C1", "T2.C2", "T3.C1",
                         "T3.C2", "T4.C1", "T4.C2", "T5.C1", "T5.C2"],
  "missing_criteria": ["T6.C1", "T6.C2", "T7.C1", "T7.C2", "…"],
  "next_trl": 6,
  "next_trl_gaps": [
    {
      "criterion_id": "T6.C1",
      "requirement": "System/subsystem model or prototype demonstration in a relevant environment",
      "status": "missing",
      "blocker": true,
      "recommendation": {
        "action": "test",
        "specifics": "Provide a prototype demonstration report with results showing the system or subsystem operating under relevant conditions.",
        "evidence_types_acceptable": ["demonstration", "test"],
        "example_artifacts": ["prototype demo report", "test results"]
      }
    },
    { "criterion_id": "T6.C2", "…": "…" }
  ],
  "evidence": [
    {
      "evidence_id": "db972b70163fd305",
      "criterion_id": "T1.C1",
      "trl_level": 1,
      "page_number": 1,
      "section_heading": "BASIC PRINCIPLES",
      "quote": "The basic principles were observed and recorded in the scientific basis experiment report.",
      "polarity": "supports",
      "strength": "moderate",
      "confidence": 0.75,
      "extraction_method": "deterministic_v1"
    }
  ],
  "criteria_set_id": "hackathon_trl_v1",
  "policy_version": "1.0"
}
```

Trivial one-liner → `status: "insufficient_evidence"`, `predicted_trl: null` (abstention — never a forced guess).
Malformed PDF → `status: "failed"` with `error`, no exception escapes the caller.

## Verification (final gate)

```text
.venv/bin/python -m pytest -q                     → 204 passed  (M1's 75 + M2's 129)
.venv/bin/ruff check src tests scripts            → exit 0
.venv/bin/python scripts/assess.py --text "…"     → status assessed, exit 0
.venv/bin/python scripts/infer.py --model …/b2_masked.joblib --text "x"
                                                  → {"n":1,"predictions":[5]}, exit 0 (M1 untouched)
git status --short / git diff --stat              → no M1 file modified
                                                    (diff = pre-existing .gitignore + 3 M2 package __init__ files)
```

## Policy v1 (summary)

1. **Criterion status** — `satisfied` ⇔ supporting item with `strength ≥ min_strength` and
   `confidence ≥ 0.5`; else `contradicted` if contradicting evidence exists; else `weak`; else `missing`.
2. **Level support** = Σ weight(satisfied) / Σ weight(all criteria of level).
3. **Monotone walk** — predicted = highest L where every level ≤ L has support ≥ 0.6.
   Level 1 failing, or fewer than 3 evidence items → **abstain** (`predicted_trl: null`).
4. **Confidence** = clamp₀₁((0.60·support + 0.25·volume + 0.15·agreement) × penalty);
   signals shift `agreement` only — evidence-derived TRL always wins.
5. **Gaps** — unsatisfied criteria of `next_trl` are blockers; lower-level weak/contradicted gaps
   rank earlier (TRL asc, weight desc); every recommendation quotes the criterion's own guidance.

## Limitations (honest scope)

1. Substring/lexicon matching is heuristic — recall/precision bounded; an LLM extractor can later
   implement `EvidenceExtractor` without engine changes.
2. English-only cues; sentence splitter is abbreviation-aware but imperfect.
3. Criteria are an *adaptation* of public NASA/DoD TRL definitions for the hackathon —
   **not a certification standard** (recorded in `source_note`).
4. Header/footer heuristic may strip legitimately repeated content (count exposed in metadata).
5. Confidence is rule-calibrated, not statistically calibrated — bands are indicative.
6. Frozen-classifier signal has abstract→long-doc domain shift; default OFF; masked variant preferred.
7. Scanned/image-only PDFs → abstain (no OCR in M2, by design; warning `no_extractable_text`).
8. Monotone level-walk (τ=0.6) is deliberately conservative and configurable, not ground truth.
