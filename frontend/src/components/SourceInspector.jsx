import React, { useState } from 'react'
import { Book, Globe, ChevronDown, ChevronUp, Star } from 'lucide-react'

function RelevanceScoreBadge({ score }) {
  const s = Number(score) || 0
  const tier = s >= 7.5 ? 'high' : s >= 5.0 ? 'medium' : 'low'
  const label = s >= 7.5 ? 'High' : s >= 5.0 ? 'Med' : 'Low'
  return (
    <span className={`relevance-score-badge ${tier}`} title={`Relevance score: ${s.toFixed(1)}/10`}>
      <Star size={9} />
      {s.toFixed(1)} · {label}
    </span>
  )
}

function BookSourceCard({ src }) {
  const [expanded, setExpanded] = useState(false)
  const hasLongSummary = src.summary && src.summary.length > 160

  return (
    <div className="source-card">
      {/* Title + score row */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '8px' }}>
        <div className="source-title">
          <Book size={13} color="#38bdf8" style={{ flexShrink: 0, marginTop: '1px' }} />
          <span>{src.title}</span>
        </div>
        {src.relevance_score !== undefined && (
          <RelevanceScoreBadge score={src.relevance_score} />
        )}
      </div>

      {/* Pages + level */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
        <span className="source-pages">{src.pages}</span>
        {src.level && (
          <span className="traversal-badge" style={{ textTransform: 'capitalize' }}>
            {src.level}
          </span>
        )}
        {src.rerank_method && (
          <span className="traversal-badge" style={{ color: '#6366f1' }}>
            {src.rerank_method === 'cross_encoder' ? '⊕ Cross-Enc' : '⊞ LLM-Rank'}
          </span>
        )}
      </div>

      {/* Summary with expand/collapse */}
      {src.summary && (
        <>
          <p className={`source-snippet ${!expanded && hasLongSummary ? 'collapsed' : ''}`}
             style={{ marginTop: '8px' }}>
            {src.summary}
          </p>
          {hasLongSummary && (
            <button
              className="source-expand-btn"
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? (
                <span style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
                  <ChevronUp size={11} /> Show less
                </span>
              ) : (
                <span style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
                  <ChevronDown size={11} /> Show more
                </span>
              )}
            </button>
          )}
        </>
      )}

      {/* Author stance */}
      {src.stance && (
        <span className="stance-tag">
          📌 {src.stance.length > 120 ? src.stance.slice(0, 120) + '...' : src.stance}
        </span>
      )}
    </div>
  )
}

function WebSourceCard({ web }) {
  const [expanded, setExpanded] = useState(false)
  const hasLong = web.snippet && web.snippet.length > 160

  return (
    <div className="source-card source-card--web">
      <div className="source-title web">
        <Globe size={13} color="#a78bfa" style={{ flexShrink: 0, marginTop: '1px' }} />
        <a
          href={web.url}
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: 'inherit', textDecoration: 'none' }}
        >
          {web.title || web.url}
        </a>
      </div>
      {web.url && (
        <div className="source-pages" style={{ marginTop: '4px' }}>
          <a href={web.url} target="_blank" rel="noopener noreferrer"
             style={{ color: '#6366f1', textDecoration: 'none', fontSize: '0.68rem' }}>
            {web.url.length > 50 ? web.url.slice(0, 50) + '…' : web.url}
          </a>
        </div>
      )}
      {web.snippet && (
        <>
          <p className={`source-snippet ${!expanded && hasLong ? 'collapsed' : ''}`}
             style={{ marginTop: '8px' }}>
            {web.snippet}
          </p>
          {hasLong && (
            <button className="source-expand-btn" style={{ color: '#a78bfa' }}
                    onClick={() => setExpanded(!expanded)}>
              {expanded ? (
                <span style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
                  <ChevronUp size={11} /> Show less
                </span>
              ) : (
                <span style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
                  <ChevronDown size={11} /> Show more
                </span>
              )}
            </button>
          )}
        </>
      )}
    </div>
  )
}

export default function SourceInspector({ bookSources, webSources }) {
  if (bookSources.length === 0 && webSources.length === 0) {
    return (
      <p style={{ fontSize: '0.82rem', color: '#64748b', padding: '8px 0' }}>
        No sources yet. Send a query to see retrieved passages here.
      </p>
    )
  }

  return (
    <div>
      {/* Book Sources */}
      {bookSources.length > 0 && (
        <>
          <h3 style={{
            fontSize: '0.78rem', color: '#94a3b8', textTransform: 'uppercase',
            letterSpacing: '0.06em', marginBottom: '10px', fontWeight: 700
          }}>
            📚 Book Corpus ({bookSources.length} chunks)
          </h3>
          {bookSources.map((src, idx) => (
            <BookSourceCard key={`book-${idx}`} src={src} />
          ))}
        </>
      )}

      {/* Web Sources */}
      {webSources.length > 0 && (
        <>
          <h3 style={{
            fontSize: '0.78rem', color: '#94a3b8', textTransform: 'uppercase',
            letterSpacing: '0.06em', margin: '18px 0 10px', fontWeight: 700
          }}>
            🌐 Tavily Web Search ({webSources.length} results)
          </h3>
          {webSources.map((web, idx) => (
            <WebSourceCard key={`web-${idx}`} web={web} />
          ))}
        </>
      )}
    </div>
  )
}
