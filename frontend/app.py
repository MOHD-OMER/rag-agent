"""
Streamlit Frontend for the Multi-Document RAG Chatbot.

Features:
- File uploader sidebar (PDF, TXT, DOCX)
- Chat interface with message bubbles
- Source citation display
- Tool activity indicator
- Clear chat / full reset buttons
- Session persistence
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Optional

import httpx
import streamlit as st

# ============================================================
# Config
# ============================================================

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
SUPPORTED_TYPES = ["pdf", "txt", "docx", "doc"]

# ============================================================
# Page Setup
# ============================================================

st.set_page_config(
    page_title="RAG Agent — Multi-Document AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,300&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  header[data-testid="stHeader"] { display: none; }
  .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

  .rag-header {
    background: linear-gradient(135deg, #0d0d0d 0%, #1a1a2e 50%, #0d0d0d 100%);
    border: 1px solid rgba(99, 179, 237, 0.3);
    border-radius: 12px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .rag-header h1 {
    color: #e2e8f0; font-size: 1.4rem; font-weight: 700;
    margin: 0; letter-spacing: -0.02em;
  }
  .rag-header span.accent { color: #63b3ed; }
  .rag-header .badge {
    background: rgba(99, 179, 237, 0.15); color: #63b3ed;
    font-size: 0.7rem; font-family: 'DM Mono', monospace;
    padding: 2px 8px; border-radius: 20px;
    border: 1px solid rgba(99, 179, 237, 0.3); letter-spacing: 0.05em;
  }

  .chat-wrapper { display: flex; flex-direction: column; gap: 1rem; }
  .msg-row { display: flex; gap: 0.8rem; align-items: flex-start; }
  .msg-row.user { flex-direction: row-reverse; }

  .avatar {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
  }
  .avatar.user-av { background: linear-gradient(135deg, #2d3748, #4a5568); border: 1px solid rgba(255,255,255,0.1); }
  .avatar.ai-av   { background: linear-gradient(135deg, #1a365d, #2b6cb0); border: 1px solid rgba(99,179,237,0.3); }

  .bubble {
    max-width: 78%; padding: 0.9rem 1.1rem; border-radius: 12px;
    font-size: 0.91rem; line-height: 1.6; word-wrap: break-word;
  }
  .bubble.user-bubble {
    background: linear-gradient(135deg, #2d3748, #374151);
    color: #e2e8f0; border: 1px solid rgba(255,255,255,0.08);
    border-top-right-radius: 3px;
  }
  .bubble.ai-bubble {
    background: linear-gradient(135deg, #0f1f3d, #162444);
    color: #cbd5e0; border: 1px solid rgba(99,179,237,0.2);
    border-top-left-radius: 3px;
  }

  .sources-box {
    margin-top: 0.6rem; padding: 0.6rem 0.9rem;
    background: rgba(99,179,237,0.05); border: 1px solid rgba(99,179,237,0.2);
    border-radius: 8px; font-size: 0.78rem; color: #90cdf4;
    max-width: 78%; margin-left: 44px;
  }
  .sources-box .src-title {
    font-family: 'DM Mono', monospace; color: #63b3ed; font-weight: 500;
    margin-bottom: 0.3rem; font-size: 0.72rem;
    letter-spacing: 0.06em; text-transform: uppercase;
  }
  .src-item { display: flex; gap: 0.5rem; margin-bottom: 0.2rem; font-family: 'DM Mono', monospace; }
  .src-file { color: #63b3ed; }
  .src-page { color: #4a5568; }

  .tool-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: rgba(237,137,54,0.1); color: #ed8936;
    border: 1px solid rgba(237,137,54,0.3);
    padding: 2px 8px; border-radius: 4px;
    font-family: 'DM Mono', monospace; font-size: 0.7rem; margin-right: 0.3rem;
  }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d0d 0%, #111827 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
  }
  .sidebar-section {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 1rem; margin-bottom: 1rem;
  }
  .sidebar-section h3 {
    color: #90cdf4; font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    font-family: 'DM Mono', monospace; margin-bottom: 0.7rem;
  }

  .file-pill {
    background: rgba(99,179,237,0.08); border: 1px solid rgba(99,179,237,0.2);
    border-radius: 6px; padding: 4px 10px; font-size: 0.78rem; color: #90cdf4;
    font-family: 'DM Mono', monospace; margin-bottom: 0.3rem;
    display: flex; justify-content: space-between;
  }
  .file-pill .chunks { color: #4a5568; font-size: 0.7rem; }

  .stat-row { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
  .stat-card {
    flex: 1; background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px; padding: 0.7rem; text-align: center;
  }
  .stat-card .sv { color: #63b3ed; font-size: 1.3rem; font-weight: 700; }
  .stat-card .sk { color: #4a5568; font-size: 0.68rem; font-family: 'DM Mono', monospace; text-transform: uppercase; }

  .stTextInput input {
    background: #1a1a2e !important; border: 1px solid rgba(99,179,237,0.25) !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  .stTextInput input:focus {
    border-color: #63b3ed !important;
    box-shadow: 0 0 0 2px rgba(99,179,237,0.15) !important;
  }

  .stButton > button {
    border-radius: 7px !important; font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important; font-size: 0.85rem !important;
  }
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a365d, #2b6cb0) !important;
    border: 1px solid rgba(99,179,237,0.4) !important; color: #e2e8f0 !important;
  }

  .empty-state { text-align: center; padding: 3rem 1rem; color: #4a5568; }
  .empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }
  .empty-state h3 { color: #718096; font-size: 1rem; font-weight: 500; }
  .empty-state p { font-size: 0.85rem; }

  .chat-scroll {
    height: 58vh; overflow-y: auto; padding-right: 0.5rem;
    scrollbar-width: thin; scrollbar-color: #2d3748 transparent;
  }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================

def init_state():
    defaults = {
        "session_id": str(uuid.uuid4()),
        "messages": [],
        "uploaded_files": [],
        "total_chunks": 0,
        # FIX: track whether the last user message has already received a
        # response, so a dropped SSE connection can't re-fire the same query.
        "awaiting_response": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ============================================================
# API Helpers
# ============================================================

def api_upload(file_bytes: bytes, filename: str) -> dict:
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{API_BASE}/upload",
            files={"file": (filename, file_bytes)},
        )
        resp.raise_for_status()
        return resp.json()


def api_stats() -> dict:
    try:
        with httpx.Client(timeout=5.0) as client:
            return client.get(f"{API_BASE}/stats").json()
    except Exception:
        return {
            "total_documents": 0,
            "active_sessions": 0,
            "llm_provider": "—",
            "embedding_model": "—",
        }


def api_clear(clear_vs: bool = False) -> dict:
    with httpx.Client(timeout=10.0) as client:
        return client.delete(
            f"{API_BASE}/clear",
            params={
                "session_id": st.session_state.session_id,
                "clear_vectorstore": clear_vs,
            },
        ).json()


def api_chat_stream(message: str):
    """
    Generator that consumes the SSE stream from the FastAPI backend.
    Yields parsed JSON events.
    """
    try:
        with httpx.stream(
            "POST",
            f"{API_BASE}/chat",
            json={
                "message": message,
                "session_id": st.session_state.session_id,
                "stream": True,
            },
            timeout=180.0,
        ) as response:
            if response.status_code != 200:
                yield {"type": "error", "content": f"API Error: {response.status_code}"}
                return

            for line in response.iter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        yield data
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield {"type": "error", "content": str(e)}


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 1rem; text-align:center;">
      <div style="font-size:2rem;">🔍</div>
      <div style="color:#63b3ed; font-weight:700; font-size:1rem; letter-spacing:-0.02em;">RAG Agent</div>
      <div style="color:#4a5568; font-size:0.72rem; font-family:'DM Mono',monospace; margin-top:2px;">MULTI-DOCUMENT AI</div>
    </div>
    """, unsafe_allow_html=True)

    stats = api_stats()
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card">
        <div class="sv">{stats.get('total_documents', 0)}</div>
        <div class="sk">Chunks</div>
      </div>
      <div class="stat-card">
        <div class="sv">{stats.get('active_sessions', 0)}</div>
        <div class="sk">Sessions</div>
      </div>
    </div>
    <div style="font-size:0.72rem; color:#4a5568; font-family:'DM Mono',monospace; margin-bottom:1rem; text-align:center;">
      LLM: {stats.get('llm_provider','—').upper()} &nbsp;|&nbsp; Embeddings: {stats.get('embedding_model','—')}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><h3>📎 Upload Documents</h3>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop files here",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        for uf in uploaded:
            already = any(f["name"] == uf.name for f in st.session_state.uploaded_files)
            if not already:
                with st.spinner(f"Indexing {uf.name}…"):
                    try:
                        result = api_upload(uf.read(), uf.name)
                        st.session_state.uploaded_files.append({
                            "name": uf.name,
                            "chunks": result["chunks_added"],
                        })
                        st.session_state.total_chunks = result["total_chunks"]
                        st.success(f"✓ {uf.name} — {result['chunks_added']} chunks")
                    except Exception as e:
                        st.error(f"Upload failed: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.uploaded_files:
        st.markdown('<div class="sidebar-section"><h3>📚 Indexed Files</h3>', unsafe_allow_html=True)
        for f in st.session_state.uploaded_files:
            icon = "📄" if f["name"].endswith(".pdf") else ("📝" if f["name"].endswith(".txt") else "📋")
            st.markdown(f"""
            <div class="file-pill">
              <span>{icon} {f['name'][:28]}{'…' if len(f['name'])>28 else ''}</span>
              <span class="chunks">{f['chunks']}c</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><h3>⚙️ Settings</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear Chat", use_container_width=True):
            api_clear(clear_vs=False)
            st.session_state.messages = []
            st.session_state.awaiting_response = False
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
    with col2:
        if st.button("💣 Full Reset", use_container_width=True, help="Clears chat + vector store"):
            api_clear(clear_vs=True)
            st.session_state.messages = []
            st.session_state.uploaded_files = []
            st.session_state.total_chunks = 0
            st.session_state.awaiting_response = False
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><h3>💡 Example Queries</h3>', unsafe_allow_html=True)
    examples = [
        "Summarize all uploaded documents",
        "What are the key findings?",
        "Compare the main topics across docs",
        "Search the web for recent AI news",
        "What does the document say about X?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:15]}"):
            st.session_state["pending_query"] = ex
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# Main Area
# ============================================================

st.markdown("""
<div class="rag-header">
  <span style="font-size:1.5rem;">🔍</span>
  <div>
    <h1>Multi-Document <span class="accent">RAG</span> Agent</h1>
  </div>
  <span class="badge">HYBRID RETRIEVAL</span>
  <span class="badge">AGENTIC AI</span>
  <span class="badge" style="margin-left:auto;">SESSION: """ + st.session_state.session_id[:8] + """</span>
</div>
""", unsafe_allow_html=True)


def render_message(msg: dict):
    role = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])
    tool_calls = msg.get("tool_calls", [])

    if role == "user":
        st.markdown(f"""
        <div class="msg-row user">
          <div class="avatar user-av">👤</div>
          <div class="bubble user-bubble">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        tool_html = ""
        if tool_calls:
            icons = {"document_search": "📄", "web_search": "🌐", "summarizer": "✂️"}
            for tc in tool_calls:
                icon = icons.get(tc, "🔧")
                tool_html += f'<span class="tool-badge">{icon} {tc}</span>'
            tool_html = f'<div style="margin-bottom:0.4rem;">{tool_html}</div>'

        st.markdown(f"""
        <div class="msg-row">
          <div class="avatar ai-av">🤖</div>
          <div class="bubble ai-bubble">{tool_html}{content}</div>
        </div>
        """, unsafe_allow_html=True)

        if sources:
            src_items = ""
            for s in sources[:4]:
                src_name = s.get("source", "Unknown")[:30]
                src_page = s.get("page", "?")
                src_items += f'<div class="src-item"><span class="src-file">📄 {src_name}</span><span class="src-page">p.{src_page}</span></div>'
            st.markdown(f"""
            <div class="sources-box">
              <div class="src-title">⛓ Sources Used</div>
              {src_items}
            </div>
            """, unsafe_allow_html=True)


# ── Chat history ──
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">🔍</div>
          <h3>Upload documents and start asking questions</h3>
          <p>This agent uses hybrid BM25 + dense retrieval.<br>
          It can search your documents, browse the web, and summarize content.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            render_message(msg)

# ── Input ──
st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

pending_query = st.session_state.pop("pending_query", "").strip()
user_input = pending_query or (st.chat_input("Ask anything about your documents or the web…") or "").strip()

if user_input:
    # FIX: Only accept new input when not already waiting for a response.
    # This prevents a dropped SSE stream from re-queuing the same message.
    if not st.session_state.awaiting_response:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.awaiting_response = True
        st.rerun()

# FIX: Only enter the response loop when the flag is set AND the last message
# is from the user. If the stream completes (done/error), the flag is cleared
# before rerun — so a subsequent rerun never re-enters this block.
if (
    st.session_state.awaiting_response
    and st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
):
    last_query = st.session_state.messages[-1]["content"]

    with chat_container:
        with st.chat_message("assistant", avatar="🤖"):
            status_placeholder = st.empty()
            token_placeholder = st.empty()
            sources_placeholder = st.empty()

            full_content = ""
            tool_calls = []
            sources = []
            stream_completed = False  # FIX: track clean completion

            for event in api_chat_stream(last_query):
                etype = event.get("type")

                if etype == "session_id":
                    st.session_state.session_id = event["session_id"]

                elif etype == "tool_start":
                    tname = event["tool"]
                    tool_calls.append(tname)
                    status_placeholder.info(f"🔧 Agent is using: {tname}...")

                elif etype == "tool_end":
                    status_placeholder.empty()

                elif etype == "token":
                    full_content += event["content"]
                    token_placeholder.markdown(full_content + "▌")

                elif etype == "done":
                    token_placeholder.markdown(full_content)
                    sources = event.get("sources", [])
                    if sources:
                        src_html = ""
                        for s in sources[:4]:
                            name = s.get("source", "Unknown")[:30]
                            page = s.get("page", "?")
                            src_html += f'<div class="src-item"><span class="src-file">📄 {name}</span><span class="src-page">p.{page}</span></div>'
                        sources_placeholder.markdown(f"""
                        <div class="sources-box" style="margin-left:0; max-width:100%;">
                          <div class="src-title">⛓ Sources Used</div>
                          {src_html}
                        </div>
                        """, unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_content,
                        "sources": sources,
                        "tool_calls": list(set(tool_calls)),
                    })
                    stream_completed = True  # FIX: mark clean exit

                elif etype == "error":
                    status_placeholder.empty()
                    st.error(event["content"])
                    # FIX: treat error as a completed exchange so we don't re-fire
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚠️ {event['content']}",
                        "sources": [],
                        "tool_calls": list(set(tool_calls)),
                    })
                    stream_completed = True
                    break

            # FIX: If the stream dropped without a done/error event (e.g. Groq
            # 429 mid-stream, network reset), still close the exchange so the
            # same query is never re-sent on the next rerun.
            if not stream_completed:
                error_msg = "⚠️ Response stream interrupted. Please try again."
                token_placeholder.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "tool_calls": list(set(tool_calls)),
                })

            # FIX: Always clear the flag after the stream loop exits,
            # regardless of how it ended.
            st.session_state.awaiting_response = False
            st.rerun()