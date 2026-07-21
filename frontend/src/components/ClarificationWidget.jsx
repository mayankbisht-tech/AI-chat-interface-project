import React from 'react'
import { HelpCircle, ArrowRight } from 'lucide-react'

export default function ClarificationWidget({ questions, onProvideContext }) {
  return (
    <div className="clarification-box">
      <div className="clarification-title">
        <HelpCircle size={18} color="#fbbf24" />
        <span>Vagueness Detected: Please clarify 1-2 details for personalized guidance</span>
      </div>
      <ul className="clarification-questions">
        {questions.map((q, idx) => (
          <li key={idx}>{q}</li>
        ))}
      </ul>
      <button 
        className="corpus-btn" 
        style={{ borderColor: 'rgba(245, 158, 11, 0.4)', color: '#fbbf24', marginTop: '8px' }}
        onClick={() => onProvideContext("Answer generally based on classic book principles")}
      >
        <span>Answer Generally Instead</span>
        <ArrowRight size={14} />
      </button>
    </div>
  )
}
