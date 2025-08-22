import React, { useState, FormEvent, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';

interface DocumentInfo {
  loaded_documents: string[];
  total_chunks: number;
  vector_count: number;
}

export default function Home() {
  const [developerMessage, setDeveloperMessage] = useState(
    'You are a professional instructor, that writes answers in professional kind tone to students that are eager to learn.'
  );
  const [userInput, setUserInput] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('gpt-4.1-mini');
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant'; content: string; sources?: string[] }[]>([]);
  const [loading, setLoading] = useState(false);
  
  // RAG/Fusion state
  const [ragMode, setRagMode] = useState(false);
  const [fusionMode, setFusionMode] = useState(false);
  const [documentInfo, setDocumentInfo] = useState<DocumentInfo>({ loaded_documents: [], total_chunks: 0, vector_count: 0 });
  const [uploadingPdf, setUploadingPdf] = useState(false);
  // Fusion settings
  const [includeWeb, setIncludeWeb] = useState(true);
  const [fusionK, setFusionK] = useState(6);
  const [numQueries, setNumQueries] = useState(4);
  const [webResults, setWebResults] = useState(3);

  // In production (Vercel), always use same-origin API routes
  const apiBaseUrl = process.env.NODE_ENV === 'production' ? '' : (process.env.NEXT_PUBLIC_API_BASE_URL || '');

  // Ref to chat history container for auto-scroll
  const chatRef = useRef<HTMLDivElement>(null);

  // Always scroll to bottom when messages change or new chunks arrive
  useEffect(() => {
    const el = chatRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, loading]);

  // Load document info when API key changes
  useEffect(() => {
    if (apiKey && (ragMode || fusionMode)) {
      loadDocumentInfo();
    }
  }, [apiKey, ragMode, fusionMode]);

  const loadDocumentInfo = async () => {
    if (!apiKey) return;
    
    try {
      const res = await fetch(`${apiBaseUrl}/api/rag/documents?api_key=${encodeURIComponent(apiKey)}`);
      if (res.ok) {
        const info = await res.json();
        setDocumentInfo(info);
      }
    } catch (error) {
      console.error('Error loading document info:', error);
    }
  };

  const handlePdfUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0 || !apiKey) return;

    setUploadingPdf(true);
    
    try {
      const formData = new FormData();
      
      // Append each file with the name 'files' (matching backend expectation)
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
      }
      formData.append('api_key', apiKey);

      const res = await fetch(`${apiBaseUrl}/api/rag/upload`, {
        method: 'POST',
        body: formData,
      });

      const result = await res.json();
      
      if (res.ok) {
        if (result.status === 'success') {
          alert(`✅ All PDFs uploaded successfully!\n\nFiles processed: ${result.successful_files.join(', ')}\nTotal chunks created: ${result.total_chunks_created}\nTotal characters: ${result.total_characters}`);
        } else if (result.status === 'partial_success') {
          alert(`⚠️ Partial success!\n\nSuccessful: ${result.successful_files.join(', ')}\nFailed: ${result.failed_files.join(', ')}\nTotal chunks created: ${result.total_chunks_created}`);
        }
        loadDocumentInfo();
      } else {
        alert(`❌ Error: ${result.detail || 'Failed to upload PDFs'}`);
      }
    } catch (error: any) {
      alert(`❌ Error uploading PDFs: ${error.message}`);
    } finally {
      setUploadingPdf(false);
      // Reset file input
      event.target.value = '';
    }
  };

  const clearDocuments = async () => {
    if (!apiKey) return;
    
    try {
      const res = await fetch(`${apiBaseUrl}/api/rag/documents?api_key=${encodeURIComponent(apiKey)}`, {
        method: 'DELETE',
      });
      
      if (res.ok) {
        alert('📄 All documents cleared successfully!');
        setDocumentInfo({ loaded_documents: [], total_chunks: 0, vector_count: 0 });
        setMessages([]);
      }
    } catch (error) {
      alert('❌ Error clearing documents');
    }
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);

    // Add user message to chat history
    setMessages((prev) => [...prev, { role: 'user', content: userInput }]);

    try {
      
      if (ragMode) {
        // RAG Chat
        const res = await fetch(`${apiBaseUrl}/api/rag/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_message: userInput,
            model,
            api_key: apiKey,
            k: 3
          }),
        });

        if (!res.ok) {
          const errorData = await res.json();
          throw new Error(errorData.detail || 'RAG chat failed');
        }

        const result = await res.json();
        setMessages((prev) => [...prev, {
          role: 'assistant',
          content: result.response,
          sources: result.sources
        }]);
      } else if (fusionMode) {
        // Fusion Chat (RAG-Fusion with optional web)
        const res = await fetch(`${apiBaseUrl}/api/rag/fusion_chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_message: userInput,
            model,
            api_key: apiKey,
            k: fusionK,
            num_queries: numQueries,
            include_web: includeWeb,
            web_results: webResults,
          }),
        });

        if (!res.ok) {
          const errorData = await res.json();
          throw new Error(errorData.detail || 'Fusion chat failed');
        }

        const result = await res.json();
        setMessages((prev) => [...prev, {
          role: 'assistant',
          content: result.response,
          sources: result.sources
        }]);
        
      } else {
        // Regular Chat
        const chatPayload = [
          { role: 'system', content: developerMessage },
          ...messages.map((m) => ({ role: m.role, content: m.content })),
          { role: 'user', content: userInput },
        ];

        const res = await fetch(`${apiBaseUrl}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: chatPayload,
            model,
            api_key: apiKey,
            stream: process.env.NODE_ENV !== 'production',
          }),
        });

        if (process.env.NODE_ENV === 'production') {
          if (!res.ok) throw new Error(await res.text());
          const data = await res.json();
          setMessages((prev) => [...prev, { role: 'assistant', content: data.response || '' }]);
          return;
        } else {
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
        }
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
        maxWidth: 800,
        margin: '0 auto',
        padding: '2rem',
        lineHeight: 1.5,
      }}
    >
      <h1 style={{ textAlign: 'center' }}>🦎 The AI Chameleon Assistant {ragMode ? '📄 + Documents' : fusionMode ? '🧪 Fusion' : ''}</h1>

      {/* Mode Toggle Buttons */}
      <div style={{ textAlign: 'center', marginBottom: '1.5rem', display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
        <button
          onClick={() => { setRagMode(false); setFusionMode(false); }}
          style={{
            padding: '0.75rem 1.5rem',
            fontSize: '1rem',
            backgroundColor: (!ragMode && !fusionMode) ? '#0066cc' : '#e9ecef',
            color: (!ragMode && !fusionMode) ? 'white' : '#495057',
            border: '2px solid #0066cc',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: 'bold',
            minWidth: '180px',
          }}
        >
          💬 Regular Chat
        </button>
        <button
          onClick={() => { setRagMode(true); setFusionMode(false); }}
          style={{
            padding: '0.75rem 1.5rem',
            fontSize: '1rem',
            backgroundColor: ragMode ? '#0066cc' : '#e9ecef',
            color: ragMode ? 'white' : '#495057',
            border: '2px solid #0066cc',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: 'bold',
            minWidth: '180px',
          }}
        >
          📄 Upload Documents
        </button>
        <button
          onClick={() => { setFusionMode(true); setRagMode(false); }}
          style={{
            padding: '0.75rem 1.5rem',
            fontSize: '1rem',
            backgroundColor: fusionMode ? '#0066cc' : '#e9ecef',
            color: fusionMode ? 'white' : '#495057',
            border: '2px solid #0066cc',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: 'bold',
            minWidth: '180px',
          }}
        >
          🧪 Fusion (RAG‑Fusion)
        </button>
      </div>

      {/* Document Upload Controls */}
      {ragMode && (
        <section
          style={{
            border: '2px solid #0066cc',
            borderRadius: '8px',
            padding: '1rem',
            marginBottom: '1.5rem',
            backgroundColor: '#f0f8ff',
          }}
        >
          <h3 style={{ margin: '0 0 1rem 0', color: '#0066cc' }}>📄 Document Management</h3>
          
          {/* Document Info */}
          <div style={{ marginBottom: '1rem', fontSize: '0.9rem' }}>
            <strong>Loaded Documents:</strong> {documentInfo.loaded_documents.length > 0 ? documentInfo.loaded_documents.join(', ') : 'None'}
            <br />
            <strong>Total Chunks:</strong> {documentInfo.total_chunks} | <strong>Vectors:</strong> {documentInfo.vector_count}
          </div>

          {/* Upload Controls */}
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
            <input
              type="file"
              accept=".pdf"
              multiple
              onChange={handlePdfUpload}
              disabled={uploadingPdf || !apiKey}
              style={{ flex: 1, minWidth: '200px' }}
            />
            <button
              onClick={clearDocuments}
              disabled={!apiKey || documentInfo.loaded_documents.length === 0}
              style={{
                padding: '0.5rem 1rem',
                backgroundColor: '#dc3545',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              Clear All
            </button>
          </div>
          
          {uploadingPdf && <p style={{ margin: '0.5rem 0', fontStyle: 'italic' }}>🔄 Processing PDF(s)...</p>}
          {!apiKey && <p style={{ margin: '0.5rem 0', color: '#dc3545' }}>⚠️ Please enter your API key first</p>}
        </section>
      )}

      {/* Settings */}
      <section
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '0.75rem',
          marginBottom: '1.5rem',
        }}
      >
        {!ragMode && !fusionMode && (
          <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            Developer Message
            <textarea
              rows={3}
              value={developerMessage}
              onChange={(e) => setDeveloperMessage(e.target.value)}
            />
          </label>
        )}
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

      {/* Fusion Settings */}
      {fusionMode && (
        <section
          style={{
            border: '2px solid #0066cc',
            borderRadius: '8px',
            padding: '1rem',
            marginBottom: '1.5rem',
            backgroundColor: '#f0f8ff',
          }}
        >
          <h3 style={{ margin: '0 0 1rem 0', color: '#0066cc' }}>🧪 Fusion Settings</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '0.75rem' }}>
            <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              k (chunks)
              <input type="number" value={fusionK} min={1} max={20} onChange={(e) => setFusionK(parseInt(e.target.value || '1'))} />
            </label>
            <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              num_queries
              <input type="number" value={numQueries} min={1} max={10} onChange={(e) => setNumQueries(parseInt(e.target.value || '1'))} />
            </label>
            <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              include web snippets
              <input type="checkbox" checked={includeWeb} onChange={(e) => setIncludeWeb(e.target.checked)} />
            </label>
            <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              web_results
              <input type="number" value={webResults} min={0} max={10} onChange={(e) => setWebResults(parseInt(e.target.value || '0'))} />
            </label>
          </div>
          {documentInfo.loaded_documents.length === 0 && includeWeb && (
            <p style={{ marginTop: '0.75rem', fontSize: '0.9rem' }}>ℹ️ No PDFs loaded. Fusion will use web snippets only.</p>
          )}
        </section>
      )}

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
          <div key={idx}>
            <div
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
            {msg.sources && msg.sources.length > 0 && (
              <div style={{ 
                fontSize: '0.8rem', 
                color: '#666', 
                marginBottom: '0.5rem',
                paddingLeft: '0.75rem',
                borderLeft: '3px solid #0066cc'
              }}>
                <strong>📄 Sources used:</strong>
                {msg.sources.map((source, i) => (
                  <div key={i} style={{ marginTop: '0.25rem' }}>• {source}</div>
                ))}
              </div>
            )}
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
          placeholder={ragMode ? "Ask a question about your documents..." : fusionMode ? "Ask a question (fusion)..." : "Type your message..."}
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          style={{ flex: 1, padding: '0.75rem' }}
          disabled={loading || (ragMode && documentInfo.loaded_documents.length === 0)}
        />
        <button
          type="submit"
          disabled={loading || !userInput.trim() || !apiKey || (ragMode && documentInfo.loaded_documents.length === 0)}
          style={{
            padding: '0.75rem 1.5rem',
            fontWeight: 'bold',
            backgroundColor: ragMode ? '#0066cc' : fusionMode ? '#0066cc' : '#111',
            color: '#fff',
            border: 'none',
            borderRadius: 6,
            cursor: 'pointer',
          }}
        >
          {loading ? '...' : ragMode ? '🔍 Ask' : fusionMode ? '🧪 Fuse' : 'Send'}
        </button>
      </form>
      
      {ragMode && documentInfo.loaded_documents.length === 0 && (
        <p style={{ textAlign: 'center', color: '#666', marginTop: '1rem' }}>
          📄 Upload PDFs to start chatting with them!
        </p>
      )}
    </main>
  );
} 