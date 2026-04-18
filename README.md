# 🔍 Multi-Document RAG Agent

A production-grade Agentic AI system that lets you upload documents and have a conversation with them. Powered by a **LangGraph ReAct agent** that autonomously decides when to search your documents, browse the web, or summarize content — with real-time streaming and persistent memory.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Agentic reasoning** | LangGraph ReAct loop — agent decides tool use autonomously |
| **Hybrid retrieval** | BM25 sparse + ChromaDB dense, merged and deduplicated |
| **FlashRank reranking** | Cross-encoder reranks top candidates for higher precision |
| **Multi-tool chaining** | Can combine document search + web search in a single answer |
| **Real-time streaming** | Token-by-token SSE streaming with live tool activity indicators |
| **Persistent memory** | SQLite-backed LangGraph checkpoints survive server restarts |
| **Web search** | Tavily API integration for current events and live data |
| **Multi-format ingestion** | PDF, TXT, DOCX support |
| **Source citations** | Every answer shows which document and page it came from |
| **Two frontends** | Streamlit (local) + Gradio (HuggingFace Spaces) |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit / Gradio UI               │
│          (streaming SSE · tool badges · citations)   │
└────────────────────────┬────────────────────────────┘
                         │ HTTP / SSE
┌────────────────────────▼────────────────────────────┐
│               FastAPI Backend  (:8000)               │
│  /upload  /chat  /history  /clear  /stats  /health  │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│            LangGraph ReAct Agent                     │
│   ┌──────────────┐  ┌────────────┐  ┌────────────┐  │
│   │document_search│  │ web_search │  │ summarizer │  │
│   └──────┬───────┘  └─────┬──────┘  └─────┬──────┘  │
│          │                │               │          │
│   ┌──────▼───────┐  ┌─────▼──────┐        │          │
│   │ HybridRetriever│  │  Tavily   │        │          │
│   │ BM25 + Chroma │  │   API     │        │          │
│   │ + FlashRank   │  └───────────┘        │          │
│   └───────────────┘                       │          │
│                                           │          │
│   ┌──────────────────────────────────────┐│          │
│   │   SQLite Checkpoint (LangGraph)      ││          │
│   │   agent_memory.db  — persistent      ││          │
│   └──────────────────────────────────────┘│          │
└─────────────────────────────────────────────────────┘
```

---

## 🗂 Project Structure

```
rag-agent/
├── backend/
│   ├── main.py            # FastAPI app — 6 endpoints
│   ├── rag_pipeline.py    # LangGraph ReAct agent + SSE streaming
│   ├── retriever.py       # Hybrid BM25 + ChromaDB + FlashRank reranker
│   ├── memory.py          # Per-session memory + SQLite rehydration on restart
│   ├── tools.py           # @tool definitions (document_search, web_search, summarizer)
│   ├── config.py          # Pydantic settings — reads from .env
│   └── __init__.py
├── frontend/
│   ├── app.py             # Streamlit UI (streaming, tool badges, citations)
│   └── gradio_app.py      # Gradio UI for HuggingFace Spaces
├── .env.example           # Template — copy to .env and fill in keys
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start (Local)

### 1. Clone and set up environment

```bash
git clone https://github.com/YOUR_USERNAME/rag-agent.git
cd rag-agent

# Create virtual environment (using uv — recommended)
uv venv --python 3.11
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

uv pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
```

### 3. Run the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 4. Run the frontend (new terminal)

```bash
# From project root
streamlit run frontend/app.py --server.port 8501
```

Open **http://localhost:8501**

---

## 🐳 Docker

```bash
# Copy and fill in your .env first
cp .env.example .env

# Build and start all services
docker compose up --build
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| Gradio UI | http://localhost:7860 |
| FastAPI docs | http://localhost:8000/docs |

To stop:

```bash
docker compose down
```

---

## 🤗 HuggingFace Spaces Deployment

1. Create a new Space — choose **Gradio** SDK
2. Set the entry point to `frontend/gradio_app.py`
3. Add secrets in Space Settings:
   - `GROQ_API_KEY`
   - `TAVILY_API_KEY`
4. Push the repo (exclude `backend/chroma_db/` and `backend/agent_memory.db` from your commit)

The Gradio frontend embeds the backend directly — no separate FastAPI process needed on Spaces.

---

## ⚙️ Configuration

All settings are controlled via environment variables or the `.env` file:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `TAVILY_API_KEY` | — | Tavily search key (optional — web search disabled without it) |
| `LLM_PROVIDER` | `groq` | `groq` or `ollama` |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace sentence-transformers model |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence path |
| `CHUNK_SIZE` | `512` | Document chunk size (tokens) |
| `CHUNK_OVERLAP` | `64` | Chunk overlap (tokens) |
| `TOP_K_RETRIEVAL` | `6` | Number of chunks returned after reranking |
| `MEMORY_WINDOW_SIZE` | `10` | Sliding window — last N turns kept in context |
| `API_HOST` | `0.0.0.0` | FastAPI bind address |
| `API_PORT` | `8000` | FastAPI port |
| `ALLOWED_ORIGINS` | `http://localhost:8501,...` | CORS allowed origins |

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Chunk count, active sessions, LLM info |
| `POST` | `/upload` | Upload and index a PDF, TXT, or DOCX file |
| `POST` | `/chat` | Send message — streaming SSE or JSON response |
| `GET` | `/history` | Get conversation history for a session |
| `DELETE` | `/clear` | Clear memory and/or vector store |

### `/chat` request body

```json
{
  "message": "What are the key findings?",
  "session_id": "abc-123",
  "stream": true
}
```

### SSE event types (stream=true)

```
{ "type": "session_id", "session_id": "..." }
{ "type": "tool_start",  "tool": "document_search", "input": "..." }
{ "type": "tool_end",    "tool": "document_search" }
{ "type": "token",       "content": "The key..." }
{ "type": "done",        "sources": [...], "tool_calls": [...] }
{ "type": "error",       "content": "..." }
```

---

## 🧠 How It Works

### Retrieval pipeline

1. **BM25** (sparse) — keyword frequency matching across all indexed chunks
2. **ChromaDB** (dense) — semantic similarity via `all-MiniLM-L6-v2` embeddings
3. **Merge** — results deduplicated by content, BM25 results prioritized
4. **FlashRank** — cross-encoder reranks the merged candidate pool, returning only the most semantically relevant chunks to the LLM

### Agent loop

The LangGraph ReAct agent receives the user query and a system prompt listing available tools. It then:

1. Decides which tool(s) to call based on the query
2. Executes tools (document search, web search, summarizer) — can chain multiple
3. Synthesizes a final cited answer from tool outputs
4. Saves the full reasoning trace as a checkpoint to SQLite

### Memory & persistence

- **In-session**: `ChatMessageHistory` keeps a sliding window of the last 10 turns for LLM context
- **Cross-restart**: LangGraph `SqliteSaver` persists full agent state to `backend/agent_memory.db`. On server restart, sessions are rehydrated from the checkpoint DB — `/history` and conversation context are fully restored

---

## 📦 Dependencies

### Core

- `langchain` / `langchain-community` / `langchain-core`
- `langchain-groq` — Groq LLM integration
- `langchain-huggingface` — HuggingFace embeddings
- `langchain-chroma` — ChromaDB vector store
- `langgraph` — ReAct agent orchestration
- `langgraph-checkpoint-sqlite` — SQLite persistence for agent state

### Retrieval

- `chromadb` — persistent vector store
- `rank-bm25` — sparse BM25 retrieval
- `sentence-transformers` — embedding model
- `flashrank` — cross-encoder reranking

### Document processing

- `pypdf` — PDF loading
- `python-docx` — DOCX loading
- `unstructured[docx]` — advanced DOCX parsing

### API & frontend

- `fastapi` + `uvicorn` — backend API
- `streamlit` — local frontend
- `gradio` — HuggingFace Spaces frontend
- `httpx` — async HTTP client for SSE streaming

---

## 🔧 Known Limitations

- **Groq rate limits** — free tier allows ~30 req/min. If you hit 429 errors, wait a few seconds between messages
- **FlashRank cold start** — the reranker downloads its model on first use (~50 MB). This adds ~10s on first query in a fresh environment. Subsequent queries are fast
- **Gradio frontend** — runs the backend in-process (no FastAPI). Streaming is synchronous; the Streamlit UI gives a better streaming experience for local use
- **DOCX support** — requires the `unstructured` package. If ingestion fails for `.docx` files, ensure `unstructured[docx]` is installed

---

## 🗺 Roadmap

- [ ] Add support for URLs as input sources (web scraping + ingestion)
- [ ] Multi-user auth with isolated vector store namespaces per user
- [ ] Streaming support in Gradio frontend
- [ ] Evaluation harness with RAGAS metrics
- [ ] Support for images inside PDFs (multimodal RAG)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.