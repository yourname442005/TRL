import { useRef, useState } from 'react'
import './App.css'

const TRL_STEPS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

function formatPercent(value) {
  if (typeof value !== 'number') return '—'
  return `${(value * 100).toFixed(1)}%`
}

function EvidenceItem({ item }) {
  return (
    <li className="evidence-item">
      <div className="evidence-head">
        <span className={`chip chip-${item.polarity === 'supports' ? 'support' : item.polarity === 'contradicts' ? 'contra' : 'neutral'}`}>
          {item.criterion_id}
        </span>
        <span className="chip chip-neutral">{item.polarity}</span>
        <span className="chip chip-neutral">{item.strength}</span>
        <span className="evidence-conf">confidence {formatPercent(item.confidence)}</span>
      </div>
      <blockquote className="evidence-quote">“{item.quote}”</blockquote>
      <div className="evidence-source">
        <span>page {item.page_number ?? '—'}</span>
        <span>section: {item.section_heading || '—'}</span>
        <span>source: {item.block_id || item.document_id}</span>
      </div>
    </li>
  )
}

function GapRow({ gap }) {
  const rec = gap.recommendation
  return (
    <tr className={gap.blocker ? 'gap-blocker' : ''}>
      <td>{gap.criterion_id}</td>
      <td>{gap.requirement}</td>
      <td>
        <span className={`chip chip-${gap.status === 'supported' ? 'support' : gap.status === 'partial' ? 'neutral' : 'missing'}`}>
          {gap.status}
        </span>
        {gap.blocker && <span className="chip chip-contra">blocker</span>}
      </td>
      <td>
        {rec ? (
          <span>
            <strong>{rec.action}</strong> — {rec.specifics}
          </span>
        ) : (
          '—'
        )}
      </td>
    </tr>
  )
}

export default function App() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const inputRef = useRef(null)

  async function handleAssess(event) {
    event.preventDefault()
    if (!file) {
      setResult(null)
      setError('Choose a PDF file to assess.')
      return
    }
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const response = await fetch('/api/v1/assess', { method: 'POST', body: form })
      const body = await response.json().catch(() => null)
      if (!body) {
        setError(`Upload failed (HTTP ${response.status}).`)
        return
      }
      if (!response.ok || body.status === 'failed') {
        setError(body.error || `Assessment failed (HTTP ${response.status}).`)
        return
      }
      setResult(body)
    } catch {
      setError('Could not reach the API. Is the backend running on port 8000?')
    } finally {
      setLoading(false)
    }
  }

  function handleReset() {
    setFile(null)
    setResult(null)
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  const meta = result?.document
  const processing = result?.processing

  return (
    <div className="app">
      <header className="app-header">
        <h1>TRL Intelligence</h1>
        <p className="subtitle">Evidence-driven technology readiness assessment</p>
      </header>

      <form className="upload-form" onSubmit={handleAssess}>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.txt,.md,application/pdf,text/plain,text/markdown"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          disabled={loading}
        />
        <button type="submit" disabled={loading || !file}>
          {loading ? 'Assessing…' : 'Assess document'}
        </button>
        <button type="button" className="secondary" onClick={handleReset} disabled={loading}>
          Reset
        </button>
      </form>

      {loading && (
        <div className="processing" role="status">
          <span className="spinner" aria-hidden="true" />
          Processing document — extraction, evidence matching, aggregation…
        </div>
      )}

      {error && (
        <div className="error-box" role="alert">
          {error}
        </div>
      )}

      {result && (
        <main className="result">
          <section className="card">
            <div className="trl-row">
              <div className="trl-badge">
                <span className="trl-label">Predicted TRL</span>
                <span className="trl-value">{result.predicted_trl ?? '—'}</span>
              </div>
              <div className="trl-meta">
                <span className={`chip chip-${result.status === 'assessed' ? 'support' : 'neutral'}`}>
                  {result.status}
                </span>
                <span>
                  confidence <strong>{formatPercent(result.confidence)}</strong> ({result.confidence_band})
                </span>
                <span>
                  next TRL <strong>{result.next_trl ?? '—'}</strong>
                </span>
              </div>
            </div>
            <div className="trl-scale" aria-label="TRL scale 1 to 9">
              {TRL_STEPS.map((step) => (
                <div
                  key={step}
                  className={`trl-step ${step === result.predicted_trl ? 'active' : ''} ${
                    step < result.predicted_trl ? 'below' : ''
                  }`}
                >
                  <span className="trl-step-number">{step}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="card">
            <h2>Criteria coverage</h2>
            <div className="criteria-group">
              <h3>Supported ({result.supported_criteria.length})</h3>
              <div className="chips">
                {result.supported_criteria.map((id) => (
                  <span key={id} className="chip chip-support">{id}</span>
                ))}
              </div>
            </div>
            <div className="criteria-group">
              <h3>Missing ({result.missing_criteria.length})</h3>
              <div className="chips">
                {result.missing_criteria.length === 0 ? (
                  <span className="chip chip-neutral">none</span>
                ) : (
                  result.missing_criteria.map((id) => (
                    <span key={id} className="chip chip-missing">{id}</span>
                  ))
                )}
              </div>
            </div>
            <div className="criteria-group">
              <h3>Contradictory evidence ({result.contradictory_evidence.length})</h3>
              <div className="chips">
                {result.contradictory_evidence.length === 0 ? (
                  <span className="chip chip-neutral">none</span>
                ) : (
                  result.contradictory_evidence.map((id) => (
                    <span key={id} className="chip chip-contra">{id}</span>
                  ))
                )}
              </div>
            </div>
          </section>

          <section className="card">
            <h2>
              Path to TRL {result.next_trl ?? '—'}
              <span className="h2-sub">{result.next_trl_gaps.length} gap(s)</span>
            </h2>
            {result.next_trl_gaps.length === 0 ? (
              <p className="muted">No open gaps recorded for the next level.</p>
            ) : (
              <table className="gaps-table">
                <thead>
                  <tr>
                    <th>Criterion</th>
                    <th>Requirement</th>
                    <th>Status</th>
                    <th>Recommendation</th>
                  </tr>
                </thead>
                <tbody>
                  {result.next_trl_gaps.map((gap) => (
                    <GapRow key={gap.criterion_id} gap={gap} />
                  ))}
                </tbody>
              </table>
            )}
          </section>

          <section className="card">
            <h2>
              Evidence
              <span className="h2-sub">{result.evidence.length} item(s)</span>
            </h2>
            {result.evidence.length === 0 ? (
              <p className="muted">No evidence items were extracted.</p>
            ) : (
              <ul className="evidence-list">
                {result.evidence.map((item) => (
                  <EvidenceItem key={item.evidence_id} item={item} />
                ))}
              </ul>
            )}
          </section>

          <section className="card meta-card">
            <h2>Run metadata</h2>
            <dl className="meta-grid">
              <div>
                <dt>Document ID</dt>
                <dd className="mono">{result.document_id}</dd>
              </div>
              <div>
                <dt>Policy version</dt>
                <dd className="mono">{result.policy_version}</dd>
              </div>
              <div>
                <dt>Criteria set</dt>
                <dd className="mono">{result.criteria_set_id}</dd>
              </div>
              <div>
                <dt>Extractor</dt>
                <dd className="mono">{processing?.extractor_version ?? '—'}</dd>
              </div>
              <div>
                <dt>Processing time</dt>
                <dd className="mono">{processing?.processing_time_ms ?? '—'} ms</dd>
              </div>
              <div>
                <dt>Source</dt>
                <dd className="mono">
                  {meta?.source_name} ({meta?.source_type}, {meta?.page_count} page(s), {meta?.byte_size} bytes)
                </dd>
              </div>
            </dl>
          </section>
        </main>
      )}

      <footer className="app-footer">
        offline assessment engine · API on :8000 · UI on :5173
      </footer>
    </div>
  )
}
