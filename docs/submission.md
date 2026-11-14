# Submission Guide — Evaluator Quick Reference

One-page guide to install, verify, and run TRL Intelligence end to end.
Details live in `README.md` (run commands) and `docs/milestone4_application.md`
(API contract); this page is the judge-facing path only.

## Prerequisites

- Python 3.11 (only)
- Node.js 18+ with npm
- No internet access is required for assessment — the engine is fully offline

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api]"
cd frontend && npm install && cd ..
```

## Verify

```bash
pytest -q                          # expect: 471 passed
ruff check src tests scripts       # expect: All checks passed
npm run build --prefix frontend    # expect: build succeeds
```

## Run

```bash
# terminal 1 — backend on :8000
.venv/bin/uvicorn trl_intelligence.api.app:app --port 8000

# terminal 2 — frontend on :5173
cd frontend && npm run dev
```

Open **http://localhost:5173**, choose a technical PDF (or
`data/validation/09_multipage_report.pdf`), click **Assess document**.

CLI equivalent (no browser):

```bash
.venv/bin/python scripts/assess.py --doc data/validation/03_relevant_environment.md
```

## Sample API request

```bash
curl -X POST http://localhost:8000/api/v1/assess \
  -F "file=@data/validation/09_multipage_report.pdf;type=application/pdf"
```

## Expected output structure

Success is HTTP 200 with a JSON body (document-level failures also return 200
with `status: "failed"`; request-level problems return 4xx with a controlled
`code` — never a traceback):

```json
{
  "status": "assessed",
  "predicted_trl": 5,
  "confidence": 0.925,
  "confidence_band": "high",
  "supported_criteria": ["T1.C1", "..."],
  "missing_criteria": ["T6.C1", "..."],
  "contradictory_evidence": [],
  "next_trl": 6,
  "next_trl_gaps": [{"criterion_id": "T6.C1", "requirement": "...", "status": "missing", "blocker": true, "recommendation": {"action": "...", "specifics": "..."}}],
  "evidence": [{"criterion_id": "T1.C1", "page_number": 1, "section_heading": "...", "quote": "...", "polarity": "supports", "confidence": 0.75}],
  "document_id": "…sha1 of file bytes…",
  "policy_version": "1.0",
  "criteria_set_id": "hackathon_trl_v1",
  "document": {"source_name": "...", "source_type": "pdf", "page_count": 5},
  "processing": {"processing_time_ms": 12.0, "extractor_version": "1.0.0"}
}
```

Every evidence item traces to a page, section, and block in the uploaded
document. Reference expectations for the bundled corpus are pinned in
`docs/milestone3_validation.md`.

## Getting help

- Architecture and API contract: `docs/milestone4_application.md`
- Assessment methodology and evidence model: `docs/milestone2_assessment.md`
- Validation, adversarial, and failure handling: `docs/milestone3_validation.md`
- Baseline models and reproducibility: `docs/baseline_v1.md`
