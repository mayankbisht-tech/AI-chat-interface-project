import React from 'react'
import { Bot, Zap, BookOpen, Layers, Settings2 } from 'lucide-react'

export default function Header({ agentMode, setAgentMode, useBooks, setUseBooks, onOpenCorpusModal, onOpenConfig, totalBooks, provider }) {
  const providerLabel = {
    gemini: 'Gemini 2.0 Flash',
    groq: 'Llama 3.3 70B',
    xai: 'grok-3',
    openai: 'GPT-4o',
  }[provider] || 'Gemini 2.0 Flash'

  return (
    <header className="app-header">
      <div className="brand-section">
        <div className="brand-icon">
          <Bot size={20} color="#ffffff" />
        </div>
        <div>
          <h1 className="brand-title">FinIndex RAG</h1>
          <span className="badge-vectorless">Vectorless · {providerLabel}</span>
        </div>
      </div>

      <div className="header-controls">
        {/* Direct LLM vs Book Search Toggle */}
        <button
          id="btn-toggle-use-books"
          className={`use-books-btn ${useBooks ? 'active' : ''}`}
          onClick={() => setUseBooks(!useBooks)}
          title={useBooks ? "Book Search is ON (retrieves excerpts from 28 finance books)" : "Direct LLM Answer is default (Click to search 28 finance books)"}
        >
          <BookOpen size={14} />
          <span>{useBooks ? '📖 Search Books: ON' : '⚡ Search Books: OFF (Direct LLM)'}</span>
        </button>

        {/* Agent Mode Toggle */}
        <div className="mode-toggle">
          <button
            id="btn-normal-agent"
            className={`mode-btn ${agentMode === 'normal' ? 'active' : ''}`}
            onClick={() => setAgentMode('normal')}
            title="Normal Agent — fast direct or single-pass retrieval"
          >
            <Zap size={14} />
            Normal
          </button>
          <button
            id="btn-deep-research-agent"
            className={`mode-btn ${agentMode === 'deep_research' ? 'active deep' : ''}`}
            onClick={() => setAgentMode('deep_research')}
            title="Deep Research Agent — multi-sub-question report"
          >
            <Layers size={14} />
            Deep Research
          </button>
        </div>

        {/* Corpus Browser */}
        <button
          id="btn-corpus-modal"
          className="corpus-btn"
          onClick={onOpenCorpusModal}
          title={`Browse all ${totalBooks || 28} books in the corpus`}
        >
          <BookOpen size={15} color="#38bdf8" />
          <span>{totalBooks || 28} Books</span>
        </button>

        {/* Settings / Config */}
        <button
          id="btn-config-panel"
          className="settings-btn"
          onClick={onOpenConfig}
          title="Configure RAG pipeline parameters"
        >
          <Settings2 size={15} />
        </button>
      </div>
    </header>
  )
}
