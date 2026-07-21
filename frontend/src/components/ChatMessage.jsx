import React from 'react'
import ReactMarkdown from 'react-markdown'
import ClarificationWidget from './ClarificationWidget'
import { Bot, User } from 'lucide-react'

export default function ChatMessage({ message, onProvideContext }) {
  const isUser = message.sender === 'user'
  const isDeep = message.mode === 'deep_research'
  const isStreaming = message.isStreaming === true

  return (
    <div
      className={`message-wrapper ${isUser ? 'message-user' : 'message-assistant'}`}
      id={`msg-${message.id}`}
    >
      {/* Avatar */}
      <div
        className={`message-avatar ${
          isUser
            ? 'avatar-user'
            : isDeep
            ? 'avatar-assistant deep'
            : 'avatar-assistant'
        }`}
      >
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>

      {/* Bubble */}
      <div className="message-bubble">
        {/* Mode badge (only on assistant messages) */}
        {!isUser && message.mode && (
          <span className={`message-mode-badge ${isDeep ? 'deep' : 'normal'}`}>
            {isDeep ? 'Deep Research' : 'Normal'}
          </span>
        )}

        {/* Content */}
        {message.isVague ? (
          <ClarificationWidget
            questions={message.questions}
            onProvideContext={onProvideContext}
          />
        ) : (
          <div style={{ paddingRight: !isUser && message.mode ? '80px' : '0' }}>
            <ReactMarkdown
              components={{
                // Open links in new tab
                a: ({ node, ...props }) => (
                  <a {...props} target="_blank" rel="noopener noreferrer" />
                ),
              }}
            >
              {message.text || ''}
            </ReactMarkdown>
            {/* Streaming cursor — visible only while message is being built */}
            {isStreaming && (
              <span
                className={`typing-cursor${isDeep ? ' deep' : ''}`}
                aria-hidden="true"
              />
            )}
          </div>
        )}
      </div>
    </div>
  )
}
