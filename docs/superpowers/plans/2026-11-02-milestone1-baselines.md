# Milestone 1 Baselines Implementation Plan

> **For agentic workers:** Use inline execution with TDD (superpowers:test-driven-development). No git commits (user rule 12).

**Goal:** Reproducible offline training of three TRL baselines (B1 LogReg, B2 cumulative ordinal LR, B3 TF-IDF+SVD+GB) with leakage-aware group splits and full/masked evaluation.

**Architecture:** sklearn-only pipeline: TechPort JSON → clean/mask → group split → TF-IDF (+SVD for B3) → model → metrics JSON + joblib + offline infer.

**Tech Stack:** Python 3.11, scikit-learn 1.9.1, numpy, pandas, pytest, ruff. No new deps.

---

## Approved decisions
- B1: TF-IDF + balanced multinomial LogisticRegression
- B2: cumulative ordinal LR — 8 binary thresholds (P(y>t) for t=1..8), sum of probs → expected value → clip 1..9
- B3: TF-IDF + TruncatedSVD + GradientBoostingClassifier (dense only; never sparse→HGB)
- Splits: group-aware 70/15/15; report post-split class distributions + missing classes
- Eval: full + TRL-masked; masked primary; MAE abs + relative delta (no "percentage points" for MAE)
- Storage: Option A — data/raw/techport/ (gitignored) + SOURCE.md
- Test set untouched during model selection (select on validation only)
- TDD for all production code; full suite + ruff at end
- No commits/pushes; no API/PDF/evidence/gap-analysis

## File map
- Create: `configs/baseline_v1.yaml` (or .json — use JSON to avoid PyYAML dep!)
- Create: `src/trl_intelligence/preprocessing/text.py`
- Create: `src/trl_intelligence/ingestion/techport.py`
- Create: `src/trl_intelligence/evaluation/splits.py`
- Create: `src/trl_intelligence/evaluation/metrics.py`
- Create: `src/trl_intelligence/models/baselines.py`
- Create: `src/trl_intelligence/prediction/artifacts.py` (serialize/load)
- Create: `scripts/run_baseline.py`
- Create: `scripts/infer.py`
- Create: `tests/test_text.py`, `tests/test_ingestion.py`, `tests/test_splits.py`, `tests/test_metrics.py`, `tests/test_baselines.py`, `tests/test_artifacts.py`
- Create: `data/raw/techport/SOURCE.md` + pages copy (gitignored)
- Create: `docs/baseline_v1.md` (final report after run)

Config format: **JSON** (stdlib) — avoids undeclared PyYAML dependency.

## Tasks (TDD order)

### Task 0: Stage 0 data copy (not unit-testable; manual verify)
- cp research pages → `data/raw/techport/pages/`
- Write SOURCE.md with provenance
- Verify: 43 files, 21161 records, state matches

### Task 1: text clean + mask
- RED tests: clean_text strips HTML/unescape/ws; mask_trl replaces TRL patterns with [MASKED]
- GREEN minimal impl

### Task 2: TechPort ingestion
- RED: load pages dir → DataFrame with project_id, title, text, trl, group_id; usable filter; invalid label drop
- GREEN

### Task 3: group splits 70/15/15
- RED: ratios by group; no group_id overlap; deterministic seed; class distribution report function returns missing classes
- GREEN

### Task 4: metrics
- RED: mae, within_1_accuracy, macro_f1, qwk, per_class, confusion; relative_mae_delta
- GREEN

### Task 5: B1, B2, B3
- RED: fit/predict shape, labels in 1..9; B2 uses 8 thresholds; B3 pipeline has SVD then GBC
- GREEN

### Task 6: artifacts save/load + infer
- RED: roundtrip predict same; offline
- GREEN

### Task 7: run script + real training run
- Wire config JSON; train on train; select hyperparams/epochs on validation only; evaluate test once at end for report
- Produce metrics comparison full vs masked

### Task 8: ruff + full pytest + verification checklist

## Hyperparameters (fixed first pass — no search needed beyond optional small grid on val)
- TF-IDF: min_df=2, max_df=0.95, ngram_range=(1,2), sublinear_tf=True, max_features=50000
- B1: LogisticRegression(max_iter=1000, class_weight=balanced, multi_class=multinomial, C=1.0)
- B2: 8 × LogisticRegression(class_weight=balanced) on y>t; predict = sum P(y>t) then round+clip — actually expected value: sum_{t=1..8} P(y>t) gives E[y] if y in 1..9? E[y] = sum_{k=1}^9 P(y>=k) = sum_{t=0}^8 P(y>t). For y in 1..9: E[y]=P(y>0)+... no. Standard: E[y] = sum_{k=1}^{8} P(y > k) + 1. Because y = 1 + sum_{k=1}^{8} 1[y>k].
- B3: TruncatedSVD(n_components=100 or min(100, n_features-1)) + GradientBoostingClassifier(n_estimators=100, random_state=seed)
- SEED=42
- Ratios 0.70/0.15/0.15 by groups

## Reporting
- experiments/results/baseline_v1/{metrics_full.json, metrics_masked.json, comparison.md, split_indices.json, class_dists.json}
- models under models/checkpoints/ (gitignored)
