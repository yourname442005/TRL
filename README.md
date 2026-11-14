# TRL Intelligence

AI-powered Technology Readiness Level assessment system.

## Objective

Analyze unstructured technical documentation and predict
Technology Readiness Levels (TRL 1-9).

## Core Capabilities

- Technical document ingestion
- PDF text extraction and preprocessing
- Technology evidence extraction
- Ordinal TRL prediction
- Evidence-backed assessment
- Technology maturity gap analysis

## Technology Stack

- Python 3.11
- Scikit-learn
- PyMuPDF
- FastAPI
- NumPy and Pandas
- React 19 + Vite (frontend)

## Project Structure

- configs/: Model and assessment configurations
- data/: Dataset storage (validation corpus in data/validation/)
- models/: Model checkpoints and final weights
- experiments/: Experiment results and logs
- src/trl_intelligence/: Core assessment pipeline
- src/trl_intelligence/api/: FastAPI backend
- frontend/: React + Vite frontend
- tests/: Automated tests
- scripts/: Training, evaluation and prediction scripts
- docs/: Technical documentation

## Development

Create and activate the virtual environment:

    python3.11 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -e ".[dev,api]"

Run tests:

    pytest

Lint:

    ruff check src tests scripts

## Judge quickstart

```bash
# 1. Install (once)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api]"
cd frontend && npm install && cd ..

# 2. Verify
pytest -q                          # expect: 471 passed
ruff check src tests scripts       # expect: All checks passed

# 3. Start the backend (terminal 1)
.venv/bin/uvicorn trl_intelligence.api.app:app --port 8000

# 4. Start the frontend (terminal 2)
cd frontend && npm run dev

# 5. Open http://localhost:5173 and upload a technical PDF
```

Expected flow: the page shows a processing spinner, then a result card with the
predicted TRL (1–9 scale), confidence and status, supported/missing criteria
chips, contradictions, the gap table for the next TRL, evidence quotes with
page/section/source, and run metadata (document ID, policy/criteria versions,
processing time). A known reference: `data/validation/09_multipage_report.pdf`
assesses to TRL 5 with high confidence. Errors (wrong file type, oversized
upload) appear as controlled messages — never tracebacks.

The Vite dev server proxies `/api` and `/health` to `localhost:8000` for local
development only; a production build expects same-origin API routing (e.g. a
reverse proxy) rather than the dev proxy.

## Running locally

Assess a document from the CLI:

    python scripts/assess.py --doc data/validation/03_relevant_environment.md

Start the API server (port 8000):

    .venv/bin/uvicorn trl_intelligence.api.app:app --reload --port 8000

Check it is up:

    curl http://localhost:8000/health

Start the frontend dev server (port 5173, first run installs npm deps):

    cd frontend
    npm install
    npm run dev

Open http://localhost:5173 — the Vite dev server proxies `/api` and `/health`
to the backend on port 8000. Keep the API server running in a second terminal.

Smoke test the API directly:

    curl -X POST http://localhost:8000/api/v1/assess \
      -F "file=@data/validation/03_relevant_environment.md;type=text/markdown"

Stop servers with Ctrl+C in each terminal.
