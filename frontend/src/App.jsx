import { useState, useRef, useEffect, useCallback } from 'react';

const API = 'http://localhost:8000';

// ── SVG Icons ──────────────────────────────────────────────────────────────────

const BotIcon = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="#7C6FFF" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="11" width="18" height="10" rx="2" />
    <path d="M9 11V8a3 3 0 0 1 6 0v3" />
    <circle cx="9" cy="16" r="1" fill="#7C6FFF" stroke="none" />
    <circle cx="15" cy="16" r="1" fill="#7C6FFF" stroke="none" />
    <path d="M12 3v2" />
  </svg>
);

const SendIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
);

const PaperclipIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
  </svg>
);

const SpinnerIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="animate-spin">
    <path d="M21 12a9 9 0 1 1-6.219-8.56" />
  </svg>
);

const ChevronIcon = ({ open }) => (
  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
    style={{ transform: open ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}>
    <polyline points="6 9 12 15 18 9" />
  </svg>
);

// ── Pre-populated starter conversation ────────────────────────────────────────

const INITIAL_MESSAGES = [
  {
    id: '0',
    role: 'bot',
    content: "Hi, I'm Aether — your document intelligence assistant. Upload a PDF with the paperclip icon and ask me anything about it. I'll retrieve the most relevant sections and cite my sources. Or just ask me something directly — I'll answer from my own knowledge.",
    usedRetrieval: false,
  },
  {
    id: '1',
    role: 'user',
    content: "How do you decide whether to search the document or answer directly?",
  },
  {
    id: '2',
    role: 'bot',
    content: "I'm a ReAct agent — I reason about your question first, then decide whether calling the query_document tool will improve my answer. If your question is about general knowledge I don't need retrieval. If it requires specific information from an uploaded document, I'll call the tool and show you which chunks I used.",
    usedRetrieval: false,
  },
];

// ── Typing indicator ──────────────────────────────────────────────────────────

function TypingIndicator() {
  return (
    <div className="flex items-start gap-3 msg-fade-in">
      <div
        className="flex-shrink-0 mt-0.5 w-7 h-7 rounded-full flex items-center justify-center"
        style={{ background: '#1C1929' }}
      >
        <BotIcon />
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-xs font-medium" style={{ color: '#8B8A96' }}>Aether</span>
        <div className="flex items-center gap-1.5 h-7 px-0.5">
          <span className="typing-dot" style={{ animationDelay: '0ms' }} />
          <span className="typing-dot" style={{ animationDelay: '160ms' }} />
          <span className="typing-dot" style={{ animationDelay: '320ms' }} />
        </div>
      </div>
    </div>
  );
}

// ── Source chunks expandable section ─────────────────────────────────────────

function Sources({ sources }) {
  const [open, setOpen] = useState(false);
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen(v => !v)}
        className="flex items-center gap-1.5 text-[11px] font-medium transition-colors"
        style={{ color: open ? '#7C6FFF' : '#8B8A96' }}
      >
        <ChevronIcon open={open} />
        {open ? 'Hide' : 'Show'} {sources.length} source{sources.length !== 1 ? 's' : ''}
      </button>

      {open && (
        <div className="mt-2 flex flex-col gap-2 msg-fade-in">
          {sources.map((src, i) => (
            <div
              key={i}
              className="p-3 rounded-lg text-xs leading-relaxed"
              style={{ background: '#111116', border: '1px solid #252530', color: '#8B8A96' }}
            >
              {src.content}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Individual message ────────────────────────────────────────────────────────

function Message({ msg }) {
  if (msg.role === 'user') {
    return (
      <div className="flex justify-end msg-fade-in">
        <div
          className="max-w-[72%] px-4 py-2.5 rounded-2xl rounded-br-sm text-sm leading-relaxed text-white"
          style={{ background: '#7C6FFF' }}
        >
          {msg.content}
        </div>
      </div>
    );
  }

  // Badge shown on every bot message:
  //   "retrieved" → agent called the MCP tool (RAG was used)
  //   "direct"    → agent answered from its own knowledge
  // This is the key agent-specific signal: did the LLM decide to use the tool or not?
  const badge = msg.usedRetrieval !== undefined ? (
    <span
      className="text-[10px] px-1.5 py-0.5 rounded-full font-medium"
      style={
        msg.usedRetrieval
          ? { background: 'rgba(124,111,255,0.14)', color: '#7C6FFF', border: '1px solid rgba(124,111,255,0.28)' }
          : { background: 'rgba(139,138,150,0.1)', color: '#8B8A96', border: '1px solid rgba(139,138,150,0.18)' }
      }
    >
      {msg.usedRetrieval ? '▸ retrieved' : '▸ direct'}
    </span>
  ) : null;

  const errorBadge = msg.error ? (
    <span
      className="text-[10px] px-1.5 py-0.5 rounded-full font-medium"
      style={{ background: 'rgba(255,80,80,0.1)', color: '#FF6B6B', border: '1px solid rgba(255,80,80,0.2)' }}
    >
      error
    </span>
  ) : null;

  return (
    <div className="flex items-start gap-3 msg-fade-in">
      <div
        className="flex-shrink-0 mt-0.5 w-7 h-7 rounded-full flex items-center justify-center"
        style={{ background: '#1C1929' }}
      >
        <BotIcon />
      </div>

      <div className="flex flex-col gap-1.5" style={{ maxWidth: 'min(80%, 640px)' }}>
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium" style={{ color: '#8B8A96' }}>Aether</span>
          {badge}
          {errorBadge}
        </div>

        <p className="text-sm leading-relaxed" style={{ color: msg.error ? '#FF6B6B' : '#EEEDF2' }}>
          {msg.content}
        </p>

        <Sources sources={msg.sources} />
      </div>
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [messages, setMessages] = useState(INITIAL_MESSAGES);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [collection, setCollection] = useState(null);   // ChromaDB collection from upload
  const [isUploading, setIsUploading] = useState(false);

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll to bottom when new messages appear or typing indicator changes
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  // Reset textarea height when the input is cleared after sending
  useEffect(() => {
    if (input === '' && textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [input]);

  const handleUpload = useCallback(async (file) => {
    if (!file || !file.name.toLowerCase().endsWith('.pdf')) return;
    setIsUploading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API}/upload`, { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }
      const data = await res.json();
      setCollection(data.collection_name);
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: 'bot',
        content: `Document ready: "${data.collection_name}" — ${data.chunks_stored} chunks indexed. Ask me anything about it.`,
        usedRetrieval: false,
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: 'bot',
        content: `Upload failed: ${err.message}`,
        error: true,
      }]);
    } finally {
      setIsUploading(false);
    }
  }, []);

  const handleSend = useCallback(async () => {
    const question = input.trim();
    if (!question || isTyping) return;

    setInput('');
    setMessages(prev => [...prev, {
      id: crypto.randomUUID(),
      role: 'user',
      content: question,
    }]);
    setIsTyping(true);

    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // Pass collection so the agent knows which ChromaDB collection to search.
        // If null, the agent will still work — it just won't call query_document.
        body: JSON.stringify({ question, collection }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }

      const data = await res.json();
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: 'bot',
        content: data.answer,
        usedRetrieval: data.used_retrieval,
        sources: data.sources,
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: 'bot',
        content: `Could not reach the API: ${err.message}. Make sure the server is running on port 8000.`,
        error: true,
      }]);
    } finally {
      setIsTyping(false);
    }
  }, [input, isTyping, collection]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const canSend = input.trim().length > 0 && !isTyping;

  return (
    <div
      className="flex flex-col"
      style={{ height: '100dvh', background: '#0D0D0F', fontFamily: "'Inter', sans-serif" }}
    >

      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <header
        className="flex items-center gap-3 px-5 py-3.5 flex-shrink-0"
        style={{ borderBottom: '1px solid #1C1C24' }}
      >
        {/* Bot avatar with pulse dot */}
        <div className="relative flex-shrink-0">
          <div
            className="w-8 h-8 rounded-full flex items-center justify-center"
            style={{ background: '#1C1929' }}
          >
            <BotIcon size={16} />
          </div>
          <span
            className="absolute -bottom-px -right-px w-2.5 h-2.5 rounded-full pulse-dot"
            style={{ background: '#7C6FFF', border: '2px solid #0D0D0F' }}
          />
        </div>

        <div className="flex flex-col leading-none gap-0.5">
          <span className="text-sm font-semibold tracking-tight" style={{ color: '#EEEDF2' }}>Aether</span>
          <span className="text-[11px]" style={{ color: '#8B8A96' }}>Document Intelligence</span>
        </div>

        <div className="ml-auto flex items-center gap-2">
          {/* Show current active collection in the header */}
          {collection && (
            <span
              className="text-[11px] px-2 py-1 rounded-md font-medium max-w-[160px] truncate"
              style={{ background: 'rgba(124,111,255,0.1)', color: '#7C6FFF', border: '1px solid rgba(124,111,255,0.2)' }}
              title={collection}
            >
              {collection}
            </span>
          )}
          <span
            className="flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1 rounded-full"
            style={{ background: 'rgba(52,199,89,0.1)', color: '#34C759' }}
          >
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: '#34C759' }} />
            Online
          </span>
        </div>
      </header>

      {/* ── Message feed ────────────────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto px-5 py-6 flex flex-col gap-6">
        {messages.map(msg => <Message key={msg.id} msg={msg} />)}
        {isTyping && <TypingIndicator />}
        <div ref={messagesEndRef} />
      </main>

      {/* ── Input area ──────────────────────────────────────────────────────── */}
      <footer
        className="flex-shrink-0 px-5 pb-5 pt-3"
        style={{ borderTop: '1px solid #1C1C24' }}
      >
        <div
          className="flex items-end gap-2 px-3 py-2.5 rounded-2xl"
          style={{ background: '#141418', border: '1px solid #252530' }}
        >
          {/* PDF upload button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            title={collection ? `Active: ${collection} — click to replace` : 'Upload a PDF'}
            className="flex-shrink-0 p-1.5 rounded-lg transition-colors"
            style={{
              color: isUploading ? '#4A4958' : collection ? '#7C6FFF' : '#8B8A96',
            }}
          >
            {isUploading ? <SpinnerIcon /> : <PaperclipIcon />}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            className="hidden"
            onChange={e => {
              handleUpload(e.target.files[0]);
              e.target.value = '';
            }}
          />

          {/* Auto-resizing textarea */}
          <textarea
            ref={textareaRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onInput={e => {
              e.target.style.height = 'auto';
              e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
            }}
            placeholder="Ask a question…"
            rows={1}
            className="flex-1 resize-none bg-transparent text-sm leading-relaxed"
            style={{
              color: '#EEEDF2',
              caretColor: '#7C6FFF',
              maxHeight: '120px',
              overflowY: 'hidden',
            }}
          />

          {/* Send button — glows when active */}
          <button
            onClick={handleSend}
            disabled={!canSend}
            className="flex-shrink-0 p-1.5 rounded-lg transition-all duration-150"
            style={{
              background: canSend ? '#7C6FFF' : 'transparent',
              color: canSend ? '#fff' : '#3A3A4A',
              boxShadow: canSend ? '0 0 14px rgba(124,111,255,0.45)' : 'none',
            }}
          >
            <SendIcon />
          </button>
        </div>

        <p className="text-center text-[11px] mt-2" style={{ color: '#3A3A4A' }}>
          Enter to send · Shift+Enter for new line
        </p>
      </footer>
    </div>
  );
}
