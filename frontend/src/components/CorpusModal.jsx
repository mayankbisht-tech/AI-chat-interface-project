import React, { useState } from 'react'
import { X, BookOpen, Search } from 'lucide-react'

export default function CorpusModal({ isOpen, onClose, books }) {
  const [search, setSearch] = useState('')

  if (!isOpen) return null

  const filtered = books.filter(b => {
    const q = search.toLowerCase()
    return (
      !q ||
      b.title?.toLowerCase().includes(q) ||
      b.stance?.toLowerCase().includes(q) ||
      b.topic_tags?.some(t => t.toLowerCase().includes(q))
    )
  })

  return (
    <div className="corpus-modal-overlay" onClick={onClose}>
      <div
        className="corpus-modal"
        onClick={e => e.stopPropagation()}
        role="dialog"
        aria-label="28 Personal Finance Books Corpus"
      >
        {/* Header */}
        <div className="corpus-modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <BookOpen color="#38bdf8" size={20} />
            <h2 style={{ fontFamily: 'Outfit', fontSize: '1.1rem', color: '#fff' }}>
              28 Personal Finance Books Corpus
            </h2>
            <span style={{
              fontSize: '0.65rem', fontWeight: 700, padding: '2px 6px', borderRadius: '5px',
              background: 'rgba(56, 189, 248, 0.12)', color: '#38bdf8',
              border: '1px solid rgba(56, 189, 248, 0.2)'
            }}>
              {books.length} books indexed
            </span>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close corpus browser">
            <X size={20} />
          </button>
        </div>

        {/* Search bar */}
        <div style={{ padding: '12px 24px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
          <div style={{ position: 'relative' }}>
            <Search size={14} color="#64748b" style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)' }} />
            <input
              id="corpus-search"
              type="text"
              placeholder="Search by title, topic, or stance..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{
                width: '100%',
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: '8px',
                padding: '8px 12px 8px 34px',
                color: '#e2e8f0',
                fontSize: '0.83rem',
                outline: 'none',
                fontFamily: 'Plus Jakarta Sans, sans-serif',
              }}
            />
          </div>
        </div>

        {/* Books Grid */}
        <div className="corpus-modal-body">
          {filtered.length === 0 && (
            <div style={{ color: '#64748b', fontSize: '0.85rem', gridColumn: '1/-1' }}>
              No books match "{search}"
            </div>
          )}
          {filtered.map((b, idx) => (
            <div key={idx} className="corpus-book-card">
              <div>
                <h3 className="corpus-book-title">{b.title}</h3>
                <p className="corpus-book-stance">
                  {b.stance || b.summary?.slice(0, 120)}
                </p>
              </div>
              <div className="corpus-book-tags">
                {(b.topic_tags || []).map((tag, tIdx) => (
                  <span key={tIdx} className="corpus-tag">#{tag}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
