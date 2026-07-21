import React, { useState } from 'react'
import { GitBranch, BookOpen, ChevronDown, ChevronRight } from 'lucide-react'

const LEVEL_COLORS = {
  book:       '#f59e0b',
  chapter:    '#38bdf8',
  section:    '#8b5cf6',
  subsection: '#10b981',
  part:       '#6366f1',
}

function LevelDot({ level }) {
  const color = LEVEL_COLORS[level] || '#64748b'
  return (
    <span
      className={`level-dot ${level}`}
      style={{ background: color }}
      title={level}
    />
  )
}

function BookGroup({ bookTitle, leaves }) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div className="traversal-section">
      {/* Collapsible book header */}
      <div
        className="traversal-section-header"
        onClick={() => setCollapsed(!collapsed)}
        role="button"
        aria-expanded={!collapsed}
      >
        <BookOpen size={13} color="#38bdf8" style={{ flexShrink: 0 }} />
        <span className="traversal-book-name">{bookTitle}</span>
        <span style={{ fontSize: '0.65rem', color: '#64748b', marginRight: '4px' }}>
          {leaves.length} leaf{leaves.length !== 1 ? 's' : ''}
        </span>
        {collapsed
          ? <ChevronRight size={13} color="#64748b" />
          : <ChevronDown  size={13} color="#64748b" />
        }
      </div>

      {/* Leaf nodes */}
      {!collapsed && leaves.map((item, idx) => (
        <div key={idx} className="traversal-card">
          <div className="traversal-leaf">
            <LevelDot level={item.level} />
            <span style={{ flex: 1 }}>{item.leaf_title || item.title || '(untitled)'}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '5px' }}>
            <span className="traversal-badge">{item.level || 'node'}</span>
            <span className="traversal-pages">{item.pages}</span>
          </div>
        </div>
      ))}
    </div>
  )
}

export default function TraversalVisualizer({ candidateBooks, trace }) {
  // Group trace items by book_title
  const bookGroups = {}
  for (const item of trace) {
    const key = item.book_title || 'Unknown Book'
    if (!bookGroups[key]) bookGroups[key] = []
    bookGroups[key].push(item)
  }

  const isEmpty = candidateBooks.length === 0 && trace.length === 0

  return (
    <div>
      {/* Level 0: Book Routing */}
      <div style={{ marginBottom: '16px' }}>
        <h3 style={{
          fontSize: '0.72rem', color: '#64748b', textTransform: 'uppercase',
          letterSpacing: '0.07em', marginBottom: '8px', fontWeight: 700
        }}>
          Level 0 — Book Routing ({candidateBooks.length})
        </h3>
        {candidateBooks.length > 0 ? (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
            {candidateBooks.map((title, idx) => (
              <span
                key={idx}
                style={{
                  fontSize: '0.7rem',
                  padding: '3px 8px',
                  borderRadius: '5px',
                  background: 'rgba(56, 189, 248, 0.1)',
                  color: '#38bdf8',
                  border: '1px solid rgba(56, 189, 248, 0.2)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                }}
              >
                <BookOpen size={10} />
                {title}
              </span>
            ))}
          </div>
        ) : (
          <p style={{ fontSize: '0.78rem', color: '#64748b' }}>
            {isEmpty
              ? 'No traversal active. Send a query to observe PageIndex retrieval.'
              : 'Routing in progress...'}
          </p>
        )}
      </div>

      {/* Legend */}
      {trace.length > 0 && (
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '12px' }}>
          {Object.entries(LEVEL_COLORS).map(([level, color]) => (
            <span key={level} style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.65rem', color: '#64748b' }}>
              <span style={{ width: '7px', height: '7px', borderRadius: '50%', background: color, display: 'inline-block' }} />
              {level}
            </span>
          ))}
        </div>
      )}

      {/* Tree traversal grouped by book */}
      <h3 style={{
        fontSize: '0.72rem', color: '#64748b', textTransform: 'uppercase',
        letterSpacing: '0.07em', marginBottom: '10px', fontWeight: 700
      }}>
        Recursive Tree Traversal ({trace.length} nodes)
      </h3>

      {Object.entries(bookGroups).map(([bookTitle, leaves]) => (
        <BookGroup key={bookTitle} bookTitle={bookTitle} leaves={leaves} />
      ))}

      {trace.length === 0 && candidateBooks.length > 0 && (
        <p style={{ fontSize: '0.78rem', color: '#64748b' }}>
          Traversal in progress...
        </p>
      )}
    </div>
  )
}
