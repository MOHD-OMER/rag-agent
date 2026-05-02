# 🔍 Multi-Document RAG Agent

> A production-grade Agentic AI system that lets you upload documents and have a conversation with them. Powered by a **LangGraph ReAct agent** that autonomously decides when to search your documents, browse the web, or summarize content — with real-time streaming and persistent memory.

---

## 📋 Table of Contents

- [What is RAG?](#-what-is-rag)
- [How RAG Works — Step by Step](#️-how-rag-works--step-by-step)
- [What is Agentic RAG?](#-what-is-agentic-rag)
- [Features](#-features)
- [Tech Stack](#️-tech-stack)
- [Screenshots](#-screenshots)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Quick Start (Local)](#-quick-start-local)
- [Docker](#-docker)
- [Configuration](#️-configuration)
- [API Reference](#-api-reference)
- [Key Concepts Explained](#-key-concepts-explained)
- [Known Limitations](#-known-limitations)
- [License](#-license)

---

## 🧠 What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that combines the power of large language models (LLMs) with external knowledge retrieval. Instead of relying solely on what the model memorized during training, RAG gives the model the ability to **look up relevant information first**, then generate a grounded, accurate answer.

Think of it like the difference between asking someone a question from pure memory vs. letting them quickly look it up in the right book before answering.

```
Without RAG:  User Question  →  LLM  →  Answer (from training only, may hallucinate)

With RAG:     User Question  →  Retrieve relevant chunks  →  LLM + context  →  Grounded Answer
```

### Why RAG?

| Problem | RAG Solution |
|---|---|
| LLMs hallucinate facts | Answers are grounded in actual documents |
| Training data has a cutoff date | Inject fresh, up-to-date documents at runtime |
| LLMs can't access private data | Index and retrieve your own files securely |
| Context window limits | Only inject the most relevant chunks, not the whole document |
| No source traceability | Every answer cites its exact document and page |

---

## ⚙️ How RAG Works — Step by Step

### Phase 1: Indexing (Offline)

Before you can query your documents, they need to be processed and stored in a way that makes retrieval fast and meaningful.

```
PDF / TXT / DOCX
       │
       ▼
  Text Extraction
       │
       ▼
  Chunking (512 tokens, 64 overlap)
       │
       ▼
  Embedding (all-MiniLM-L6-v2)
       │
       ▼
  ChromaDB (vector store)  +  BM25 index (keyword index)
```

1. **Text Extraction** — Raw text is pulled from uploaded files (PDF, TXT, DOCX).
2. **Chunking** — The text is split into overlapping windows. Overlap of 64 tokens ensures context is never lost at a chunk boundary.
3. **Embedding** — Each chunk is converted into a dense vector using `all-MiniLM-L6-v2`. Semantically similar text ends up geometrically close in vector space.
4. **Storage** — Vectors are stored in ChromaDB for semantic search; raw chunks are also indexed with BM25 for keyword matching.

---

### Phase 2: Retrieval (At Query Time)

When you ask a question, the system finds the most relevant chunks before passing anything to the LLM.

```
User Query
    │
    ├──► BM25 Search     (keyword matching)      ─┐
    │                                              ├──► Merge & Deduplicate
    └──► ChromaDB Search (semantic similarity)   ─┘
                                                   │
                                                   ▼
                                           FlashRank Reranker
                                     (cross-encoder rescores each candidate)
                                                   │
                                                   ▼
                                         Top-K Relevant Chunks
```

This project uses **Hybrid Retrieval** — two fundamentally different strategies working together:

- **BM25 (sparse)** — a classical IR algorithm scoring chunks by keyword frequency. Great for exact matches and named entities.
- **ChromaDB (dense)** — semantic similarity search via embeddings. Great for conceptual and paraphrase matches even when keywords differ.
- **FlashRank (reranker)** — a cross-encoder that re-reads the query alongside each candidate, producing a much more accurate final ranking than either retrieval method alone.

---

### Phase 3: Generation

The retrieved chunks are injected into the LLM's context window alongside the user's question, allowing the model to produce a precise, cited answer.

```
System Prompt
+ Retrieved Chunks  (with source + page metadata)
+ Conversation History  (last N turns)
+ User Query
        │
        ▼
   Groq LLM  (llama-3.1-8b-instant)
        │
        ▼
  Streamed Answer with Source Citations
```

---

## 🤖 What is Agentic RAG?

Standard RAG is a fixed pipeline: retrieve → generate. **Agentic RAG** goes further — it gives the LLM the ability to **reason about which tool to call**, chain multiple tools together, and loop until it has enough information to answer confidently.

This project uses a **LangGraph ReAct agent**, following the ReAct (Reasoning + Acting) paradigm:

```
User Query
    │
    ▼
[Think]  What do I need to answer this?
    │
    ├── Need info from uploaded docs?      → call document_search()
    ├── Need current / live data?          → call web_search()
    ├── Content is long, needs distilling? → call summarizer()
    └── Have enough info already?          → generate final answer
    │
    ▼
[Act]  Execute chosen tool(s)
    │
    ▼
[Observe]  Process tool output
    │
    ▼
[Loop or Answer]  Repeat if needed, else stream cited response
```

The agent can **chain multiple tools in a single query** — for example, pulling background context from your documents and then hitting the web for the latest update, all in one turn.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Agentic reasoning** | LangGraph ReAct loop — agent decides tool use autonomously |
| **Hybrid retrieval** | BM25 sparse + ChromaDB dense, merged and deduplicated |
| **FlashRank reranking** | Cross-encoder reranks top candidates for higher precision |
| **Multi-tool chaining** | Combines document search + web search in a single answer |
| **Real-time streaming** | Token-by-token SSE streaming with live tool activity indicators |
| **Persistent memory** | SQLite-backed LangGraph checkpoints survive server restarts |
| **Web search** | Tavily API integration for current events and live data |
| **Multi-format ingestion** | PDF, TXT, and DOCX support |
| **Source citations** | Every answer shows which document and page it came from |
| **Two frontends** | Streamlit (local) + Gradio (HuggingFace Spaces) |
| **Dockerized** | Single `docker compose up` to run everything |

---

## 🛠️ Tech Stack

### AI / LLM
| Library | Role |
|---|---|
| `langchain` / `langchain-core` | Chain abstractions, prompt templates, tool definitions |
| `langchain-groq` | Groq LLM provider integration |
| `langchain-ollama` | Optional local LLM via Ollama |
| `langgraph` | ReAct agent graph, stateful execution loop |
| `langgraph-checkpoint-sqlite` | Persistent agent memory across server restarts |

### Retrieval & Embeddings
| Library | Role |
|---|---|
| `chromadb` + `langchain-chroma` | Dense vector store for semantic retrieval |
| `sentence-transformers` | `all-MiniLM-L6-v2` embedding model |
| `rank-bm25` | Sparse keyword retrieval (BM25 algorithm) |
| `flashrank` | Cross-encoder reranker for precision boosting |

### Document Processing
| Library | Role |
|---|---|
| `pypdf` | PDF text extraction |
| `python-docx` | DOCX parsing |
| `unstructured` | Advanced ingestion (DOCX, tables, structured content) |

### Backend & API
| Library | Role |
|---|---|
| `fastapi` | REST + SSE streaming API server |
| `uvicorn` | ASGI server |
| `python-multipart` | File upload handling |
| `pydantic` / `pydantic-settings` | Data validation and config management |
| `python-dotenv` | Environment variable loading |

### Search & Web
| Library | Role |
|---|---|
| `tavily-python` | Real-time web search API |
| `httpx` | Async HTTP client |

### Frontend
| Library | Role |
|---|---|
| `streamlit` | Local interactive chat UI with streaming support |
| `gradio` | HuggingFace Spaces-compatible UI |

### Infrastructure
| Tool | Role |
|---|---|
| `Docker` / `docker-compose` | Containerized multi-service deployment |
| `SQLite` (via `aiosqlite`) | Persistent conversation checkpoints |
| `numpy` / `pandas` | Data utilities |

---

## 📸 Screenshots

### Dashboard & Document Upload
![Upload](assets/screenshot_upload.png)

### Multi-tool Agentic Response
> Agent autonomously chains `web_search` · `document_search` · `summarizer` in a single query

![Chat](assets/screenshot_chat.png)

### Live Web Search via Tavily API
> Real-time results for current events — agent decides when documents aren't enough

![Web Search](assets/screenshot_websearch.png)

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
│   │HybridRetriever│  │  Tavily   │        │          │
│   │ BM25 + Chroma │  │   API     │        │          │
│   │ + FlashRank   │  └───────────┘        │          │
│   └───────────────┘                                  │
│                                                      │
│   ┌──────────────────────────────────────────────┐   │
│   │   SQLite Checkpoint (LangGraph)              │   │
│   │   agent_memory.db  — persists across restart │   │
│   └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

---

## 🗂 Project Structure

```
rag-agent/
├── backend/
│   ├── main.py            # FastAPI app — 6 REST + SSE endpoints
│   ├── rag_pipeline.py    # LangGraph ReAct agent + SSE streaming logic
│   ├── retriever.py       # Hybrid BM25 + ChromaDB + FlashRank reranker
│   ├── memory.py          # Per-session memory + SQLite rehydration on restart
│   ├── tools.py           # @tool definitions (document_search, web_search, summarizer)
│   ├── config.py          # Pydantic settings — reads from .env
│   └── __init__.py
├── frontend/
│   ├── app.py             # Streamlit UI (streaming, tool badges, citations)
│   └── gradio_app.py      # Gradio UI for HuggingFace Spaces deployment
├── assets/                # Screenshots used in README
├── .env.example           # Template — copy to .env and fill keys
├── requirements.txt       # All Python dependencies
├── Dockerfile             # Multi-stage Docker image
├── docker-compose.yml     # Orchestrates backend + both frontends
└── README.md
```

---

## 🚀 Quick Start (Local)

### 1. Clone and set up environment

```bash
git clone https://github.com/MOHD-OMER/rag-agent.git
cd rag-agent

conda create -n rag-agent python=3.11.15
conda activate rag-agent
pip install -r requirements.txt
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

> `TAVILY_API_KEY` is optional — web search will be disabled if not provided.

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

The easiest way to run the full stack — backend + both frontends — in one command:

```bash
cp .env.example .env
# Fill in your API keys in .env, then:
docker compose up --build
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| Gradio UI | http://localhost:7860 |
| FastAPI docs | http://localhost:8000/docs |

---

## ⚙️ Configuration

All settings are loaded from `.env` via Pydantic settings (`config.py`).

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `TAVILY_API_KEY` | — | Tavily search key (optional) |
| `LLM_PROVIDER` | `groq` | `groq` or `ollama` |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace sentence-transformers model |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence path |
| `CHUNK_SIZE` | `512` | Document chunk size in tokens |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `TOP_K_RETRIEVAL` | `6` | Number of chunks returned after reranking |
| `MEMORY_WINDOW_SIZE` | `10` | Sliding window — last N turns kept in context |

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

### SSE Event Types (`stream=true`)

When streaming is enabled on `/chat`, the server sends newline-delimited JSON events:

```jsonc
{ "type": "session_id",  "session_id": "abc123" }
{ "type": "tool_start",  "tool": "document_search", "input": "..." }
{ "type": "tool_end",    "tool": "document_search" }
{ "type": "token",       "content": "The key finding is..." }
{ "type": "done",        "sources": [...], "tool_calls": [...] }
{ "type": "error",       "content": "Something went wrong" }
```

---

## 🧩 Key Concepts Explained

### Chunking & Overlap

Large documents cannot fit in an LLM's context window, so they are split into smaller pieces called **chunks**. With `CHUNK_SIZE=512` and `CHUNK_OVERLAP=64`, each chunk shares 64 tokens with its neighbor — preventing important information from being silently severed at a boundary.

### Embeddings

An **embedding** is a numerical representation of text as a high-dimensional vector. `all-MiniLM-L6-v2` maps sentences into a 384-dimensional space where semantically similar sentences are geometrically close. This is what enables semantic search — finding relevant chunks even when the user's wording differs entirely from the document.

### BM25 vs. Dense Retrieval

| | BM25 (Sparse) | ChromaDB (Dense) |
|---|---|---|
| Method | TF-IDF keyword frequency scoring | Cosine similarity on embeddings |
| Strength | Exact keyword & named entity matches | Conceptual / paraphrase matching |
| Weakness | Misses synonyms and semantics | May miss rare or specific terms |
| **Combined** | **Hybrid gives the best of both worlds** | |

### Reranking with FlashRank

Initial retrieval returns candidates quickly but imprecisely. **FlashRank** uses a cross-encoder that reads the query and each candidate chunk *together*, scoring them far more accurately than embedding-distance alone. It only runs on the small top-K pool, so the added precision costs minimal latency.

### ReAct Agent Loop

**ReAct** (Reason + Act) is a prompting strategy where the LLM alternates between *thinking* about what to do next and *acting* by calling a tool. This lets the system tackle complex, multi-step queries rather than attempting everything in a single shot. LangGraph manages the execution graph, state transitions, and tool routing.

### Session Memory & Checkpointing

Each user session maintains a sliding window of the last `MEMORY_WINDOW_SIZE` conversation turns in context. The full reasoning trace — including all tool calls, intermediate thoughts, and outputs — is checkpointed to SQLite via LangGraph. Conversations survive server restarts without losing history.

### SSE Streaming

The `/chat` endpoint supports **Server-Sent Events**, allowing the frontend to display tokens as they are generated rather than waiting for the full response. Tool-use events (`tool_start`, `tool_end`) are interleaved with tokens, so the UI can show live activity badges like `🔍 Searching documents...` while the agent is still working.

---

## 🔧 Known Limitations

- **Groq rate limits** — free tier is ~30 req/min. If you hit a 429, wait a few seconds and retry.
- **FlashRank cold start** — the reranker model (~50 MB) downloads on first use, adding ~10 seconds to the first query.
- **DOCX support** — requires `unstructured[docx]` which installs system-level dependencies. Use Docker if you run into install issues.
- **Single-user design** — all sessions share the same vector store. Multi-tenant namespacing is on the roadmap.
---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
