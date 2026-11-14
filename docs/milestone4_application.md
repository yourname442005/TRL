# Milestone 4 — End-to-End Application Layer

Milestone 4 wraps the frozen M1–M3 assessment engine in a thin application
layer: a FastAPI backend and a minimal React frontend. The engine itself was
not modified. All assessment logic stays in `src/trl_intelligence/pipeline.py`
and its collaborators; HTTP routes and React components only parse, transport,
and display results.

**Final gate:** 471 passed · `ruff check src tests scripts` exit 0 · CLI/API
equivalence verified · 5/5 real-socket E2E tests pass · frontend builds and
proxies correctly · zero commits made · zero M1–M3 assessment files modified.

## 1. Architecture

    Frontend (Vite+React :5173)
        │  fetch FormData
        ▼  proxy /api, /health → :8000
    FastAPI (src/trl_intelligence/api/app.py)
        │  parse upload, map errors to controlled status codes
        ▼
    AssessmentService (src/trl_intelligence/api/service.py)
        │  validate bytes, stage to safe temp file, cleanup
        ▼
    Existing pipeline: assess_document(AssessmentRequest)   ← UNCHANGED
        │
        ▼
    AssessmentResult.to_dict() + {document, processing} wrappers
        │
        ▼  JSON
    React renders the payload (no scoring logic in JS)

## 2. What was built

### Backend

- `src/trl_intelligence/api/service.py` — `AssessmentService.assess_upload`:
  filename sanitization, extension/content-type allowlists, empty/oversize/magic
  checks, staging to `tempfile.mkdtemp` as `upload{ext}` (so `document_id` is
  byte-identical to the CLI), guaranteed `shutil.rmtree` cleanup, and
  `ServiceValidationError` with machine-readable codes.
- `src/trl_intelligence/api/app.py` — `create_app()` factory plus module-level
  `app` for uvicorn. Two routes, four exception handlers.

### Frontend (`frontend/`, Vite + React, port 5173)

- Upload form (PDF/MD/TXT), processing spinner, controlled error box.
- Result views: TRL 1–9 scale with predicted level highlighted, confidence +
  band + status, supported / missing / contradictory criteria chips, next-TRL
  gap table with blocker flags and recommendations, evidence list with quote +
  page + section + source block, run metadata (document ID, policy version,
  criteria set, extractor version, processing time, source stats).
- Zero assessment logic in JavaScript — every value is rendered directly from
  the API JSON.

## 3. API contract

| Route | Method | Success | Failure modes (controlled JSON: `status`, `code`, `error`) |
|---|---|---|---|
| `/health` | GET | 200 `{status:"ok", service, policy_version, criteria_set_id, criteria_content_hash}` | — |
| `/api/v1/assess` | POST multipart, field `file` | 200 `AssessmentResult.to_dict()` + `document` + `processing` | 400 `empty_file` · 413 `file_too_large` (20 MB) · 415 `unsupported_type` / `unsupported_content_type` / `invalid_content` · 422 `invalid_request` · 500 `internal_error` (logged server-side only) |
| any other | — | — | 404 `not_found`, 405 `method_not_allowed` |

Document-level failures (e.g. unparseable PDF) return **200** with
`status:"failed"` and an `error` string — only request-level problems get 4xx.
No traceback, filesystem path, or library detail ever appears in a response.

## 4. How to run

```bash
# CLI (unchanged)
.venv/bin/python scripts/assess.py --doc data/validation/03_relevant_environment.md

# API
.venv/bin/uvicorn trl_intelligence.api.app:app --reload --port 8000
curl http://localhost:8000/health

# Frontend (second terminal)
cd frontend && npm install && npm run dev
# open http://localhost:5173  (proxies /api and /health to :8000)
```

Verified live during this milestone: `--reload` server healthy, Vite index
served (`<title>TRL Intelligence</title>`), `POST` through the proxy returned
TRL 5 / 11 evidence items / 2.4 ms for `03_relevant_environment.md`.

## 5. CLI/API equivalence (Stage 5)

`tests/test_api_equivalence.py` runs `scripts/assess.py` and
`POST /api/v1/assess` on the same bytes and asserts exact equality of
`status`, `predicted_trl`, `confidence`, `confidence_band`, `support_scores`,
`supported_criteria`, `missing_criteria`, `contradictory_evidence`, `next_trl`,
`next_trl_gaps`, `criteria_set_id`, `policy_version`, `signals`, `error`,
`document_id`, and the full `evidence` list. The API may only add the keys
`document` and `processing`.

- 5 corpus documents (assessed / insufficient / contradictory / PDF): **pass**
- malformed PDF (both adapters fail identically): **pass**

## 6. Security verification (Stage 8)

`tests/test_api_security.py` (13 tests):

- No shell execution during upload (os.system/popen/subprocess monkeypatched to
  raise — request still succeeds).
- Staged filename is never user-controlled: temp dir under system temp,
  contents only `upload.*`, cleaned up; `../../evil.pdf` and `/etc/passwd.pdf`
  neutralized (no file lands in CWD, `source_name` is `passwd.pdf`).
- Error bodies are clean JSON: no `Traceback`, no `site-packages`, no local
  paths, no leaked exception internals (including a monkeypatched 500).
- `/etc/passwd` and friends → 404/405, never served; injected `criteria_path`
  / `config_path` form fields ignored (config comes from code constants only).
- Static guard: no `shell=True`, `os.system`, `os.popen`, `eval(`, `exec(` in
  any `src/trl_intelligence/api/*.py`.

Plus API tests covering every 4xx code, and service tests for traversal and
temp cleanup.

## 7. End-to-end over a real socket (Stage 9)

`tests/test_e2e.py` spawns a real `uvicorn` subprocess on a free port and
talks to it with `httpx` (5 tests, 2.74 s):

1. `/health` returns service identity.
2. Markdown upload: API JSON equals CLI JSON field-for-field over the wire.
3. PDF upload: `source_type:"pdf"`, 1 page, assessed.
4. `.exe` upload → 415 `unsupported_type`, no traceback.
5. Malformed PDF → 200 `status:"failed"` with controlled error.

Server is terminated (SIGTERM → SIGKILL fallback) in fixture teardown; no
stray processes remain.

## 8. Dependencies added (approved)

| Package | Why | Where |
|---|---|---|
| `httpx` | FastAPI TestClient + E2E client | `pyproject.toml` dev extra |
| `python-multipart` | multipart upload parsing | `pyproject.toml` api extra |
| `react`, `react-dom`, `vite`, `@vitejs/plugin-react` | frontend stack (explicitly requested) | `frontend/package.json` |

No other Python or npm dependencies were introduced. Frontend has no test
runner (would require vitest/testing-library — not installed to avoid
unreported dependencies); UI correctness is covered by build + live smoke +
E2E proxy tests.

## 9. Changes made in M4

New files:

- `src/trl_intelligence/api/__init__.py`, `service.py`, `app.py`
- `frontend/` — `package.json`, `package-lock.json`, `vite.config.js`,
  `index.html`, `src/main.jsx`, `src/App.jsx`, `src/index.css`, `src/App.css`
- `tests/test_api.py`, `tests/test_api_service.py`,
  `tests/test_api_equivalence.py`, `tests/test_api_security.py`,
  `tests/test_e2e.py`
- `docs/milestone4_application.md` (this file)

Modified files:

- `README.md` — local run commands (CLI, API, frontend, smoke curl)
- `pyproject.toml` — `python-multipart` (api extra), `httpx` (dev extra)
- `.gitignore` — `frontend/node_modules/`, `frontend/dist/`

Not modified: every M1/M2/M3 assessment file (`src/trl_intelligence/` outside
`api/`, `configs/`, corpus, M1–M3 test suites). The only pre-existing tracked
files touched are `README.md`, `pyproject.toml`, `.gitignore`, and the three
M2 package `__init__.py` files were already modified before M4 started.

No commits were made (`git log` still shows only the initial skeleton commit
`59a22e5`).

## 10. Test count

| Suite | Tests |
|---|---|
| M1–M3 (frozen, unchanged) | 419 |
| `test_api_service.py` | 13 |
| `test_api.py` | 15 |
| `test_api_equivalence.py` | 6 |
| `test_api_security.py` | 13 |
| `test_e2e.py` | 5 |
| **Total** | **471 passed** |

## 11. Ruff result

```
$ ruff check src tests scripts
All checks passed!
exit 0
```

## 12. Honest limitations

- No authentication, persistence, or multi-user support (out of scope by
  design).
- 20 MB upload cap; PDF-only magic-byte check (other types trusted to the
  extension/content-type allowlists).
- Frontend is a single-page demo: no routing, no state persistence, no
  offline caching; it requires both servers running.
- `POST /api/v1/assess` returns document-level failures as HTTP 200 by design
  (the request succeeded; the document did not) — clients must check
  `status` in the body.
- Swagger UI at `/docs` loads its assets from a CDN (useful online, not
  offline).
- No frontend unit tests (see §8); covered indirectly by build, live smoke
  test, and socket-level E2E.

## Final gate

| Check | Result |
|---|---|
| `pytest -q` | **471 passed** |
| `ruff check src tests scripts` | **exit 0** |
| CLI/API equivalence (6 tests) | pass |
| Real-socket E2E (5 tests) | pass |
| Security suite (13 tests) | pass |
| `npm run build` | pass (17 modules) |
| Live proxy smoke (health + assess on :5173) | pass |
| M1–M3 logic modified | none |
| Commits made | none |
