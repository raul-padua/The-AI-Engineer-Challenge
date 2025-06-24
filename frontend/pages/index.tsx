import React, { useState, FormEvent, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';

export default function Home() {
  const [developerMessage, setDeveloperMessage] = useState(
    'You are a professional instructor, that writes answers in professional kind tone to students that are eager to learn.'
  );
  const [userInput, setUserInput] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('gpt-4.1-mini');
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant'; content: string }[]>([]);
  const [loading, setLoading] = useState(false);

  // Ref to chat history container for auto-scroll
  const chatRef = useRef<HTMLDivElement>(null);

  // Always scroll to bottom when messages change or new chunks arrive
  useEffect(() => {
    const el = chatRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, loading]);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);

    // Add user message to chat history
    setMessages((prev) => [...prev, { role: 'user', content: userInput }]);

    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000';
      // Build chat history payload
      const chatPayload = [
        { role: 'system', content: developerMessage },
        ...messages.map((m) => ({ role: m.role, content: m.content })),
        { role: 'user', content: userInput },
      ];

      const res = await fetch(`${baseUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: chatPayload,
          model,
          api_key: apiKey,
        }),
      });

      if (!res.ok || !res.body) {
        throw new Error(await res.text());
      }

      // Push placeholder assistant message (empty)
      setMessages((prev) => [...prev, { role: 'assistant', content: '' }]);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let assistantContent = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunkValue = decoder.decode(value);
        assistantContent += chunkValue;
        setMessages((prev) => {
          const updated = [...prev];
          const lastIdx = updated.length - 1;
          if (lastIdx >= 0) {
            updated[lastIdx].content = assistantContent;
          }
          return updated;
        });
      }
    } catch (error: any) {
      console.error(error);
      setMessages((prev) => [...prev, { role: 'assistant', content: `Error: ${error.message || 'Unknown error'}` }]);
    } finally {
      setLoading(false);
      setUserInput('');
    }
  };

  return (
    <main
      style={{
        fontFamily: 'sans-serif',
        maxWidth: 720,
        margin: '0 auto',
        padding: '2rem',
        lineHeight: 1.5,
      }}
    >
      <h1 style={{ textAlign: 'center' }}>🧑‍💻 AI Engineer Challenge Chat</h1>

      {/* Settings */}
      <section
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '0.75rem',
          marginBottom: '1.5rem',
        }}
      >
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          Developer Message
          <textarea
            rows={3}
            value={developerMessage}
            onChange={(e) => setDeveloperMessage(e.target.value)}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          Model
          <input
            type="text"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          OpenAI API Key
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
          />
        </label>
      </section>

      {/* Chat history */}
      <div
        style={{
          border: '1px solid #ccc',
          borderRadius: 8,
          padding: '1rem',
          minHeight: 240,
          maxHeight: '60vh',
          overflowY: 'auto',
          backgroundColor: '#fafafa',
        }}
        ref={chatRef}
      >
        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              backgroundColor: msg.role === 'user' ? '#000' : 'transparent',
              color: msg.role === 'user' ? '#fff' : '#000',
              padding: '0.5rem 0.75rem',
              borderRadius: 6,
              marginBottom: '0.5rem',
            }}
          >
            <ReactMarkdown>{msg.content}</ReactMarkdown>
          </div>
        ))}
        {loading && messages.length > 0 && messages[messages.length - 1].role === 'assistant' && messages[messages.length - 1].content === '' && (
          <div
            style={{
              backgroundColor: 'transparent',
              padding: '0.5rem 0.75rem',
              borderRadius: 6,
              marginBottom: '0.5rem',
              fontStyle: 'italic',
              color: '#555',
            }}
          >
            typing...
          </div>
        )}
      </div>

      {/* User input */}
      <form
        onSubmit={handleSubmit}
        style={{
          display: 'flex',
          gap: '0.5rem',
          marginTop: '1rem',
        }}
      >
        <input
          type="text"
          placeholder="Type your message..."
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          style={{ flex: 1, padding: '0.75rem' }}
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading || !userInput.trim()}
          style={{
            padding: '0.75rem 1.5rem',
            fontWeight: 'bold',
            backgroundColor: '#111',
            color: '#fff',
            border: 'none',
            borderRadius: 6,
            cursor: 'pointer',
          }}
        >
          {loading ? '...' : 'Send'}
        </button>
      </form>
    </main>
  );
} 