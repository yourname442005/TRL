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
- PyTorch (planned)
- Hugging Face Transformers (planned)
- PyMuPDF
- FastAPI
- NumPy and Pandas

## Project Structure

- configs/: Model and assessment configurations
- data/: Dataset storage
- models/: Model checkpoints and final weights
- experiments/: Experiment results and logs
- src/: Core ML pipeline
- api/: Backend API
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
