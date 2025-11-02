import { Link } from 'react-router-dom';
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './ChatInterface.css';

const ChatInterface = ({ userRole, onLogout }) => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(`session_${Date.now()}`);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Welcome message
    setMessages([{
      type: 'agent',
      content: 'Hello! I am your AI Loan Assistant. I can help you with:\n\n• Loan application processing\n• Document verification\n• Credit assessment\n• Loan status inquiries\n• General loan information\n\nHow can I assist you today?',
      timestamp: new Date().toLocaleTimeString()
    }]);
  }, []);

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage = {
      type: 'user',
      content: inputMessage,
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setLoading(true);

    try {
      const response = await axios.post('/api/chat', {
        message: inputMessage,
        session_id: sessionId,
        user_role: userRole
      });

      const agentMessage = {
        type: 'agent',
        content: response.data.response,
        timestamp: new Date().toLocaleTimeString(),
        agentType: response.data.agent_type,
        confidence: response.data.confidence,
        explainability: response.data.explainability
      };

      setMessages(prev => [...prev, agentMessage]);
    } catch (error) {
      const errorMessage = {
        type: 'error',
        content: 'Sorry, I encountered an error. Please try again or contact support.',
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="page-container">
      <div className="sidebar">
        <h2>Loan ERP</h2>
        <div className="nav-menu">
          <Link to="/dashboard" className="nav-item">Dashboard</Link>
          <Link to="/chat" className="nav-item active">Chat Assistant</Link>
          <Link to="/analytics" className="nav-item">Analytics</Link>
          {userRole === 'admin' && (
            <Link to="/upload" className="nav-item">Data Upload</Link>
          )}
        </div>
        <button onClick={onLogout} className="logout-btn">Logout</button>
      </div>

      <div className="main-content">
        <div className="chat-container">
          <div className="chat-header">
            <h2>AI Loan Assistant</h2>
            <span className="status-badge">Online</span>
          </div>

          <div className="messages-container">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.type}`}>
                <div className="message-content">
                  <p style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</p>
                  {msg.explainability && (
                    <div className="explainability-section">
                      <h4>Explanation:</h4>
                      <p>{msg.explainability.reasoning}</p>
                      {msg.explainability.factors && (
                        <div className="factors">
                          <strong>Key Factors:</strong>
                          <ul>
                            {msg.explainability.factors.map((factor, idx) => (
                              <li key={idx}>{factor}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {msg.confidence && (
                        <div className="confidence">
                          Confidence: <span className="confidence-value">{(msg.confidence * 100).toFixed(1)}%</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
                <span className="message-time">{msg.timestamp}</span>
              </div>
            ))}
            {loading && (
              <div className="message agent">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-container">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message here... (Press Enter to send)"
              rows="3"
              disabled={loading}
            />
            <button
              onClick={handleSendMessage}
              disabled={loading || !inputMessage.trim()}
              className="send-btn"
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
