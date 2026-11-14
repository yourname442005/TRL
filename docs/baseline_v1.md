# Milestone 1 Baseline Comparison Report

**Date:** 2026-11-02  
**Config:** `configs/baseline_v1.json` (seed=42)  
**Data:** NASA TechPort pages under `data/raw/techport/pages/` (see `SOURCE.md`)  
**Primary text variant:** TRL-masked (full reported for comparison)

## Dataset and splits

- Usable rows: **14,679** (valid `trlCurrent` 1–9 + cleaned description ≥ 200 chars)
  - Note: prior feasibility count was 14,678; cleaning order (tags stripped before unescape) changed one borderline length filter.
- Group key: SHA-1 of cleaned full text (exact duplicates share a group).
- Split: group-aware **70 / 15 / 15** train / val / test (seed 42).

| Split | Rows | Groups | Missing classes (1–9) |
|-------|------|--------|------------------------|
| train | 10,266 | 10,167 | none |
| val   | 2,210  | 2,179  | none |
| test  | 2,203  | 2,178  | none |

Post-split class distributions: see `experiments/results/baseline_v1/split_report.json`.  
All nine classes appear in every split. Test indices frozen in `split_indices.json`.  
**Model selection used validation metrics only; test evaluated once at the end.**

## Models

| ID | Description |
|----|-------------|
| B1 | TF-IDF + balanced multinomial LogisticRegression |
| B2 | Cumulative ordinal LR: 8 binary thresholds \(P(y>t)\), \(t=1..8\); \(\hat y = \mathrm{clip}(\mathrm{round}(1+\sum_t P(y>t)),1,9)\) |
| B3 | TF-IDF + TruncatedSVD (dense) + GradientBoostingClassifier (no sparse→HGB) |

## Test-set results (n=2,203)

### Primary: TRL-masked text

| Model | MAE ↓ | ±1 acc ↑ | Macro-F1 ↑ | QWK ↑ | Accuracy ↑ |
|-------|------:|---------:|-----------:|------:|-----------:|
| **B2** | **1.0268** | **0.7494** | 0.1419 | **0.3765** | 0.2914 |
| B3 | 1.1189 | 0.7099 | 0.1298 | 0.2005 | 0.3355 |
| B1 | 1.2410 | 0.6518 | **0.2031** | 0.3733 | 0.2878 |

**Ranking by masked test MAE:** B2 → B3 → B1

### Full text (comparison)

| Model | MAE | ±1 acc | Macro-F1 | QWK | Accuracy |
|-------|----:|-------:|---------:|----:|---------:|
| B1 | 1.2397 | 0.6500 | 0.2041 | 0.3712 | 0.2905 |
| B2 | 1.0309 | 0.7476 | 0.1411 | 0.3719 | 0.2896 |
| B3 | 1.1285 | 0.7099 | 0.1258 | 0.1969 | 0.3314 |

### Full vs masked MAE (absolute and relative)

Relative delta defined as \((\mathrm{MAE}_{full} - \mathrm{MAE}_{masked}) / \mathrm{MAE}_{full}\)  
(positive ⇒ masked MAE is lower). **MAE differences are not percentage points.**

| Model | MAE full | MAE masked | Abs delta | Rel delta |
|-------|--------:|-----------:|----------:|----------:|
| B1 | 1.23967 | 1.24103 | −0.00136 | −0.11% |
| B2 | 1.03087 | 1.02678 | +0.00409 | +0.40% |
| B3 | 1.12846 | 1.11893 | +0.00953 | +0.84% |

Masking TRL keywords has **negligible** effect on MAE (|rel| < 1% for all three). Leakage from explicit TRL mentions is not driving headline MAE.

## Interpretation

- **B2** wins on ordinal distance (MAE, ±1) as designed for cumulative thresholds; it collapses extremes (per-class F1 for 1/8/9 ≈ 0) — expected for a mean-like ordinal decoder under imbalance.
- **B1** has the best macro-F1 and QWK (tied-ish with B2) — better rare-class balance, worse central MAE.
- **B3** sits in the middle on MAE with the highest raw accuracy but low QWK/macro-F1 (predicts mid classes often).
- No single model dominates all metrics; masked MAE ranks B2 first.

## Artifacts

| Path | Contents |
|------|----------|
| `experiments/results/baseline_v1/metrics.json` | Full metrics + ranking |
| `experiments/results/baseline_v1/split_report.json` | Per-split class distributions / missing classes |
| `experiments/results/baseline_v1/split_indices.json` | Frozen project_id lists |
| `models/checkpoints/baseline_v1/{b1,b2,b3}_{full,masked}.joblib` | Serialized models (gitignored) |
| `configs/baseline_v1.json` | Seed, ratios, paths |

## Offline inference

```bash
python scripts/infer.py --model models/checkpoints/baseline_v1/b2_masked.joblib --text "..."
```

Verified: loads joblib without network and returns integer TRL 1–9.

## Reproduce

```bash
# data already at data/raw/techport/pages/ (gitignored)
.venv/bin/python scripts/run_baseline.py
.venv/bin/python -m pytest
.venv/bin/ruff check src tests scripts
```

## Out of scope (per approval)

API, frontend, PDF ingestion, evidence extraction, gap-analysis — not built.  
No new dependencies introduced. No commits/pushes. Raw data never committed.
