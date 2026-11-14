# TechPort dataset source (Option A)

- **Source API:** `POST https://techport.nasa.gov/api/projects/search` (single-use nonce via `GET https://techport.nasa.gov/api/nonce`)
- **Crawl date:** 2026-11-02 (UTC, from page `retrieved_at` fields)
- **Pages:** 43 × offset 0..21000 step 500, limit 500
- **Records:** 21,161 unique `projectId` values; 0 duplicates
- **Crawl state:** see `crawl_state.json` (`total`: 21161, `pages_written`: 43, `failed_offsets`: [])
- **Population summary:** `population_report.json`
- **Crawler reference:** `crawl.py`
- **License note:** data.gov catalog lists License as USA.gov government works (`https://www.usa.gov/government-works`). Public API access does not equal a formal redistribution EULA; third-party contractor text may appear in descriptions. Raw pages are **not** committed to git (`data/raw/*` ignored).
- **Retrieval UA:** `TRL-Intelligence-Feasibility/1.0`
- **Rate limit:** ≥1s between search requests (as crawled)

Do not commit contents of `pages/` to git.
