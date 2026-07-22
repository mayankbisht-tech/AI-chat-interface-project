import React, { useState, useEffect, useRef } from 'react'
import Header from './components/Header'
import ChatMessage from './components/ChatMessage'
import TraversalVisualizer from './components/TraversalVisualizer'
import SourceInspector from './components/SourceInspector'
import CorpusModal from './components/CorpusModal'
import ConfigPanel from './components/ConfigPanel'
import { Send, RefreshCw, PlusCircle } from 'lucide-react'

// Generate a stable session ID per browser tab
function makeSessionId() {
  const stored = sessionStorage.getItem('fin_session_id')
  if (stored) return stored
  const id = `sess_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
  sessionStorage.setItem('fin_session_id', id)
  return id
}

const SESSION_ID = makeSessionId()

const WELCOME_MSG = {
  id: 1,
  sender: 'assistant',
  text: `Hello! I'm your Personal Finance Assistant.\n\n⚡ **Direct LLM Mode (Default)**: Ask any financial question and I will answer directly using expert LLM reasoning.\n\n📖 **Search Book Corpus (Optional)**: Click **"Search Books: OFF"** in the header above to turn ON RAG search over 28 personal finance books whenever you want book-based excerpts.\n\n**Ask me anything about:**\n- 📈 Investing (index funds, stocks, portfolio allocation)\n- 💳 Debt payoff (Debt Snowball, FIRE, leverage)\n- 🏠 Real estate & financial independence\n- 🧠 Money mindset, budgeting & retirement`,
  mode: 'normal',
}

export default function App() {
  const [agentMode, setAgentMode] = useState('normal')
  const [useBooks, setUseBooks] = useState(false)
  const [provider, setProvider] = useState('gemini')
  const [messages, setMessages] = useState([WELCOME_MSG])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [statusMessage, setStatusMessage] = useState('')
  const [streamingMsgId, setStreamingMsgId] = useState(null)

  // Side panel state
  const [candidateBooks, setCandidateBooks] = useState([])
  const [traversalTrace, setTraversalTrace] = useState([])
  const [bookSources, setBookSources] = useState([])
  const [webSources, setWebSources] = useState([])
  const [activeTab, setActiveTab] = useState('traversal')

  // Modal state
  const [corpusBooks, setCorpusBooks] = useState([])
  const [isCorpusOpen, setIsCorpusOpen] = useState(false)
  const [isConfigOpen, setIsConfigOpen] = useState(false)

  const messagesEndRef = useRef(null)

  useEffect(() => { fetchCorpus(); fetchHealth() }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, statusMessage])

  const fetchHealth = async () => {
    try {
      const res = await fetch('/api/health')
      if (res.ok) {
        const data = await res.json()
        setProvider(data.provider || 'gemini')
      }
    } catch (e) {
      console.warn('Health check failed:', e)
    }
  }

  const fetchCorpus = async () => {
    try {
      const res = await fetch('/api/corpus')
      const data = await res.json()
      setCorpusBooks(data.books || [])
    } catch (e) {
      console.error('Failed to fetch corpus:', e)
    }
  }

  const handleNewChat = async () => {
    // Clear backend memory for this session
    try {
      await fetch(`/api/memory/${SESSION_ID}`, { method: 'DELETE' })
    } catch (e) {
      console.warn('Memory clear failed:', e)
    }
    // Reset UI
    setMessages([WELCOME_MSG])
    setCandidateBooks([])
    setTraversalTrace([])
    setBookSources([])
    setWebSources([])
    setInput('')
    setStatusMessage('')
  }

  const handleSend = async (queryText = input, skipVagueness = false) => {
    if (!queryText.trim() || isLoading) return

    const userMsg = { id: Date.now(), sender: 'user', text: queryText }
    const assistantMsgId = Date.now() + 1

    setMessages(prev => [...prev, userMsg])
    setInput('')
    setIsLoading(true)
    setStatusMessage('Initializing retrieval...')
    setStreamingMsgId(null)

    // Reset side panel for new query
    setCandidateBooks([])
    setTraversalTrace([])
    setBookSources([])
    setWebSources([])

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: queryText,
          mode: agentMode,
          skip_vagueness: true,
          use_books: useBooks,
          session_id: SESSION_ID,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let assistantMsgCreated = false
      let sseBuffer = '' // Buffer for incomplete SSE lines

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        // Append new chunk to buffer and split on SSE line boundaries
        sseBuffer += decoder.decode(value, { stream: true })
        const lines = sseBuffer.split('\n')

        // Keep the last (possibly incomplete) line in the buffer
        sseBuffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const rawJson = line.slice(6).trim()
          if (!rawJson) continue

          let data
          try {
            data = JSON.parse(rawJson)
          } catch {
            // Partial JSON from chunked transfer — will be completed in next chunk
            sseBuffer = line + '\n' + sseBuffer
            continue
          }

          switch (data.event) {
            case 'status':
              setStatusMessage(data.message || '')
              break

            case 'clarification':
              setMessages(prev => [
                ...prev,
                {
                  id: assistantMsgId,
                  sender: 'assistant',
                  isVague: true,
                  questions: data.questions || [],
                  mode: agentMode,
                },
              ])
              setIsLoading(false)
              setStatusMessage('')
              return

            case 'traversal':
              setCandidateBooks(data.candidate_books || [])
              setTraversalTrace(data.trace || [])
              // Auto-switch to traversal tab when data arrives
              setActiveTab('traversal')
              break

            case 'sources':
              setBookSources(data.sources || [])
              setWebSources(data.web_sources || [])
              // Auto-switch to sources tab when sources arrive
              if ((data.sources?.length || 0) + (data.web_sources?.length || 0) > 0) {
                setActiveTab('sources')
              }
              break

            case 'answer_chunk':
              if (!assistantMsgCreated) {
                assistantMsgCreated = true
                setStreamingMsgId(assistantMsgId)
                setMessages(prev => [
                  ...prev,
                  {
                    id: assistantMsgId,
                    sender: 'assistant',
                    text: data.chunk || '',
                    mode: agentMode,
                    isStreaming: true,
                  },
                ])
              } else {
                setMessages(prev =>
                  prev.map(msg =>
                    msg.id === assistantMsgId
                      ? { ...msg, text: msg.text + (data.chunk || '') }
                      : msg
                  )
                )
              }
              break

            case 'done':
              // Remove streaming flag from the completed message
              setMessages(prev =>
                prev.map(msg =>
                  msg.id === assistantMsgId
                    ? { ...msg, isStreaming: false }
                    : msg
                )
              )
              setStreamingMsgId(null)
              setIsLoading(false)
              setStatusMessage('')
              break

            case 'error':
              setMessages(prev => [
                ...prev,
                {
                  id: Date.now(),
                  sender: 'assistant',
                  text: `⚠️ Error: ${data.message || 'An error occurred during retrieval.'}`,
                  mode: agentMode,
                },
              ])
              setIsLoading(false)
              setStatusMessage('')
              break

            default:
              break
          }
        }
      }
    } catch (e) {
      console.error('Chat error:', e)
      setMessages(prev => [
        ...prev,
        {
          id: Date.now(),
          sender: 'assistant',
          text: '⚠️ Connection error. Please check that the backend is running on port 8000.',
          mode: agentMode,
        },
      ])
    } finally {
      setIsLoading(false)
      setStatusMessage('')
      setStreamingMsgId(null)
      // Ensure streaming flag is cleared
      setMessages(prev =>
        prev.map(msg => msg.isStreaming ? { ...msg, isStreaming: false } : msg)
      )
    }
  }

  const isDeepMode = agentMode === 'deep_research'

  return (
    <div className="app-container">
      <Header
        agentMode={agentMode}
        setAgentMode={setAgentMode}
        useBooks={useBooks}
        setUseBooks={setUseBooks}
        onOpenCorpusModal={() => setIsCorpusOpen(true)}
        onOpenConfig={() => setIsConfigOpen(true)}
        provider={provider}
        totalBooks={corpusBooks.length}
      />

      <div className="main-content">
        {/* ── Left: Chat Area ─────────────────────────────────────────── */}
        <div className="chat-area">
          <div className="messages-list">
            {messages.map(msg => (
              <ChatMessage
                key={msg.id}
                message={msg}
                onProvideContext={ans => handleSend(ans, true)}
              />
            ))}

            {/* Loading / status indicator */}
            {isLoading && (
              <div className="loading-indicator">
                <div className={`message-avatar avatar-assistant${isDeepMode ? ' deep' : ''}`}>
                  <RefreshCw size={16} />
                </div>
                <div className="status-bubble">
                  <div className={`status-spinner${isDeepMode ? ' deep' : ''}`} />
                  <span>{statusMessage || 'Reasoning over 28 finance books via PageIndex tree traversal...'}</span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="input-container">
            <form
              className="input-form"
              onSubmit={e => { e.preventDefault(); handleSend() }}
            >
              <button
                type="button"
                className="new-chat-btn"
                onClick={handleNewChat}
                title="Start a new conversation (clears memory)"
                aria-label="New chat"
              >
                <PlusCircle size={16} />
                <span>New Chat</span>
              </button>
              <input
                id="chat-input"
                className={`chat-input${isDeepMode ? ' deep-mode' : ''}`}
                placeholder={
                  isDeepMode
                    ? 'Ask a deep financial question (multi-book, cross-referenced, real-time)...'
                    : 'Ask any personal finance question...'
                }
                value={input}
                onChange={e => setInput(e.target.value)}
                disabled={isLoading}
              />
              <button
                id="chat-send-btn"
                className={`send-btn${isDeepMode ? ' deep' : ''}`}
                type="submit"
                disabled={isLoading || !input.trim()}
                aria-label="Send message"
              >
                <Send size={15} />
              </button>
            </form>
          </div>
        </div>

        {/* ── Right: Side Panel ────────────────────────────────────────── */}
        <aside className="side-panel">
          <div className="panel-tabs">
            <button
              id="tab-traversal"
              className={`panel-tab ${activeTab === 'traversal' ? 'active' : ''}`}
              onClick={() => setActiveTab('traversal')}
            >
              Tree ({traversalTrace.length})
            </button>
            <button
              id="tab-sources"
              className={`panel-tab ${activeTab === 'sources' ? 'active' : ''}`}
              onClick={() => setActiveTab('sources')}
            >
              Sources ({bookSources.length + webSources.length})
            </button>
          </div>

          <div className="panel-content">
            {activeTab === 'traversal' ? (
              <TraversalVisualizer
                candidateBooks={candidateBooks}
                trace={traversalTrace}
              />
            ) : (
              <SourceInspector
                bookSources={bookSources}
                webSources={webSources}
              />
            )}
          </div>
        </aside>
      </div>

      {/* Modals */}
      <CorpusModal
        isOpen={isCorpusOpen}
        onClose={() => setIsCorpusOpen(false)}
        books={corpusBooks}
      />

      <ConfigPanel
        isOpen={isConfigOpen}
        onClose={() => setIsConfigOpen(false)}
      />
    </div>
  )
}
