import React, { useState, useEffect } from 'react'
import { X, Settings2, Save, RotateCcw, CheckCircle, AlertCircle } from 'lucide-react'

const DEFAULTS = {
  num_subqueries: 4,
  top_k_rerank: 8,
  tree_expansion_breadth: 3,
  max_books_to_route: 4,
  max_chapters_per_book: 3,
  max_sections_per_chapter: 3,
  token_budget_book_routing: 8000,
  token_budget_chapter_select: 4000,
  token_budget_section_select: 2000,
  token_budget_answer_synthesis: 12000,
  context_token_budget: 4000,
  cross_encoder_enabled: false,
  deep_research_cross_encoder: true,
  min_relevance_score: 6.0,
  enable_web_search_fallback: true,
}

export default function ConfigPanel({ isOpen, onClose }) {
  const [config, setConfig] = useState(DEFAULTS)
  const [saveStatus, setSaveStatus] = useState(null) // 'success' | 'error' | null
  const [isSaving, setIsSaving] = useState(false)

  useEffect(() => {
    if (isOpen) {
      fetchConfig()
    }
  }, [isOpen])

  const fetchConfig = async () => {
    try {
      const res = await fetch('/api/config')
      if (res.ok) {
        const data = await res.json()
        setConfig(prev => ({ ...prev, ...data }))
      }
    } catch (e) {
      console.error('Failed to fetch config:', e)
    }
  }

  const handleSave = async () => {
    setIsSaving(true)
    setSaveStatus(null)
    try {
      const res = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_subqueries: Number(config.num_subqueries),
          top_k_rerank: Number(config.top_k_rerank),
          tree_expansion_breadth: Number(config.tree_expansion_breadth),
          max_books_to_route: Number(config.max_books_to_route),
          max_chapters_per_book: Number(config.max_chapters_per_book),
          max_sections_per_chapter: Number(config.max_sections_per_chapter),
          token_budget_book_routing: Number(config.token_budget_book_routing),
          token_budget_chapter_select: Number(config.token_budget_chapter_select),
          token_budget_section_select: Number(config.token_budget_section_select),
          token_budget_answer_synthesis: Number(config.token_budget_answer_synthesis),
          context_token_budget: Number(config.context_token_budget),
          cross_encoder_enabled: config.cross_encoder_enabled,
          deep_research_cross_encoder: config.deep_research_cross_encoder,
          min_relevance_score: Number(config.min_relevance_score),
          enable_web_search_fallback: config.enable_web_search_fallback,
        }),
      })
      if (res.ok) {
        setSaveStatus('success')
        setTimeout(() => setSaveStatus(null), 3000)
      } else {
        setSaveStatus('error')
      }
    } catch (e) {
      setSaveStatus('error')
    } finally {
      setIsSaving(false)
    }
  }

  const handleReset = () => {
    setConfig(DEFAULTS)
    setSaveStatus(null)
  }

  const setField = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }))
    setSaveStatus(null)
  }

  if (!isOpen) return null

  return (
    <>
      <div className="config-panel-overlay" onClick={onClose} />
      <aside className="config-panel" role="dialog" aria-label="RAG Pipeline Configuration">
        {/* Header */}
        <div className="config-panel-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Settings2 size={18} color="#38bdf8" />
            <span className="config-panel-title">Pipeline Configuration</span>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close config panel">
            <X size={18} />
          </button>
        </div>

        {/* Body */}
        <div className="config-panel-body">

          {/* ── Retrieval Settings ─────────────────────────────────────────── */}
          <div>
            <div className="config-section-label">Retrieval Settings</div>

            <div className="config-field">
              <label htmlFor="cfg-subqueries">Sub-Queries per Question</label>
              <div className="field-desc">Multi-query reformulations generated per user query (3-5 recommended)</div>
              <input
                id="cfg-subqueries"
                type="number"
                className="config-input"
                min={1} max={8}
                value={config.num_subqueries}
                onChange={e => setField('num_subqueries', e.target.value)}
              />
            </div>

            <div className="config-field">
              <label htmlFor="cfg-topk">Top-K After Re-Ranking</label>
              <div className="field-desc">Chunks kept after reranking for answer synthesis (6-12 recommended)</div>
              <input
                id="cfg-topk"
                type="number"
                className="config-input"
                min={2} max={20}
                value={config.top_k_rerank}
                onChange={e => setField('top_k_rerank', e.target.value)}
              />
            </div>

            <div className="config-field">
              <label htmlFor="cfg-max-books">Max Books Routed (Level 0)</label>
              <div className="field-desc">Max books selected from the 28-book corpus per query (1-4)</div>
              <input
                id="cfg-max-books"
                type="number"
                className="config-input"
                min={1} max={8}
                value={config.max_books_to_route}
                onChange={e => setField('max_books_to_route', e.target.value)}
              />
            </div>

            <div className="config-field">
              <label htmlFor="cfg-breadth">Tree Expansion Breadth</label>
              <div className="field-desc">Max nodes explored per level during recursive tree traversal</div>
              <input
                id="cfg-breadth"
                type="number"
                className="config-input"
                min={1} max={6}
                value={config.tree_expansion_breadth}
                onChange={e => setField('tree_expansion_breadth', e.target.value)}
              />
            </div>

            <div className="config-field">
              <label htmlFor="cfg-min-score">Relevance Gate Threshold (0-10)</label>
              <div className="field-desc">Minimum relevance score for a chunk to pass the hard gate</div>
              <input
                id="cfg-min-score"
                type="number"
                className="config-input"
                min={0} max={10} step={0.5}
                value={config.min_relevance_score}
                onChange={e => setField('min_relevance_score', e.target.value)}
              />
            </div>
          </div>

          {/* ── Token Budgets ─────────────────────────────────────────────── */}
          <div>
            <div className="config-section-label">Token Budgets (grok-3 = 131K ctx)</div>

            <div className="config-field">
              <label htmlFor="cfg-tok-routing">Level 0 Book Routing</label>
              <div className="field-desc">Tokens for feeding all 28 book summaries to LLM</div>
              <input
                id="cfg-tok-routing"
                type="number"
                className="config-input"
                min={1000} max={20000} step={500}
                value={config.token_budget_book_routing}
                onChange={e => setField('token_budget_book_routing', e.target.value)}
              />
            </div>

            <div className="config-field">
              <label htmlFor="cfg-tok-chapter">Level 1 Chapter Selection</label>
              <div className="field-desc">Tokens for chapter summaries per candidate book</div>
              <input
                id="cfg-tok-chapter"
                type="number"
                className="config-input"
                min={500} max={10000} step={500}
                value={config.token_budget_chapter_select}
                onChange={e => setField('token_budget_chapter_select', e.target.value)}
              />
            </div>

            <div className="config-field">
              <label htmlFor="cfg-tok-section">Level 2 Section Selection</label>
              <div className="field-desc">Tokens for section summaries per selected chapter</div>
              <input
                id="cfg-tok-section"
                type="number"
                className="config-input"
                min={500} max={6000} step={250}
                value={config.token_budget_section_select}
                onChange={e => setField('token_budget_section_select', e.target.value)}
              />
            </div>

            <div className="config-field">
              <label htmlFor="cfg-tok-synthesis">Answer Synthesis</label>
              <div className="field-desc">Tokens for leaf content in the final synthesis prompt</div>
              <input
                id="cfg-tok-synthesis"
                type="number"
                className="config-input"
                min={4000} max={32000} step={1000}
                value={config.token_budget_answer_synthesis}
                onChange={e => setField('token_budget_answer_synthesis', e.target.value)}
              />
            </div>
          </div>

          {/* ── Reranking ─────────────────────────────────────────────────── */}
          <div>
            <div className="config-section-label">Reranking</div>

            <div className="config-field">
              <div className="config-toggle">
                <input
                  type="checkbox"
                  id="cfg-cross-encoder-normal"
                  checked={config.cross_encoder_enabled}
                  onChange={e => setField('cross_encoder_enabled', e.target.checked)}
                />
                <label htmlFor="cfg-cross-encoder-normal" style={{ cursor: 'pointer' }}>
                  Cross-Encoder in Normal Agent
                </label>
              </div>
              <div className="field-desc" style={{ marginTop: '4px' }}>
                Off = LLM pointwise (fast, &lt;20s). On = cross-encoder (slower, more accurate).
              </div>
            </div>

            <div className="config-field">
              <div className="config-toggle">
                <input
                  type="checkbox"
                  id="cfg-cross-encoder-deep"
                  checked={config.deep_research_cross_encoder}
                  onChange={e => setField('deep_research_cross_encoder', e.target.checked)}
                />
                <label htmlFor="cfg-cross-encoder-deep" style={{ cursor: 'pointer' }}>
                  Cross-Encoder in Deep Research Agent
                </label>
              </div>
              <div className="field-desc" style={{ marginTop: '4px' }}>
                Recommended ON for deep mode (no latency limit).
              </div>
            </div>
          </div>

          {/* ── Web Search ────────────────────────────────────────────────── */}
          <div>
            <div className="config-section-label">Web Search (Tavily)</div>
            <div className="config-field">
              <div className="config-toggle">
                <input
                  type="checkbox"
                  id="cfg-web-search"
                  checked={config.enable_web_search_fallback}
                  onChange={e => setField('enable_web_search_fallback', e.target.checked)}
                />
                <label htmlFor="cfg-web-search" style={{ cursor: 'pointer' }}>
                  Enable Tavily Web Search Fallback
                </label>
              </div>
              <div className="field-desc" style={{ marginTop: '4px' }}>
                Triggers when corpus relevance gate fails or query needs current data.
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="config-panel-footer">
          <div style={{ display: 'flex', flexDirection: 'column', flex: 1, gap: '8px' }}>
            {saveStatus && (
              <div className={`config-save-status ${saveStatus}`}>
                {saveStatus === 'success' ? (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '5px', justifyContent: 'center' }}>
                    <CheckCircle size={13} /> Saved successfully
                  </span>
                ) : (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '5px', justifyContent: 'center' }}>
                    <AlertCircle size={13} /> Save failed — check console
                  </span>
                )}
              </div>
            )}
            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                className="reset-config-btn"
                onClick={handleReset}
                title="Reset all values to defaults"
              >
                <RotateCcw size={14} />
              </button>
              <button
                className="save-config-btn"
                onClick={handleSave}
                disabled={isSaving}
              >
                {isSaving ? 'Saving...' : (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px', justifyContent: 'center' }}>
                    <Save size={14} /> Save Configuration
                  </span>
                )}
              </button>
            </div>
          </div>
        </div>
      </aside>
    </>
  )
}
