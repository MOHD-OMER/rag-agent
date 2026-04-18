"""
Gradio Frontend for HuggingFace Spaces deployment.
Mirrors all features of the Streamlit app with HF-compatible UI.

Deploy to HF Spaces:
  1. Set secrets: GROQ_API_KEY, TAVILY_API_KEY
  2. Set SDK: gradio
  3. Entry point: frontend/gradio_app.py
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import List, Tuple

import gradio as gr

# When running on HF Spaces, embed the backend directly
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# ============================================================
# Singletons — import once from backend modules
# ============================================================

from retriever import get_retriever   # noqa: E402  (after sys.path insert)
from memory import get_memory_manager  # noqa: E402

_ingested_files: List[dict] = []

SESSION_ID = str(uuid.uuid4())


# ============================================================
# Core Functions
# ============================================================

def upload_files(files) -> str:
    """Ingest uploaded files and return status message."""
    if not files:
        return "No files provided."

    # FIX: correct import path (langchain_text_splitters, not langchain.text_splitter)
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        UnstructuredWordDocumentLoader,
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    results = []
    retriever = get_retriever()

    for f in files:
        filename = Path(f.name).name
        ext = Path(filename).suffix.lower()

        loader_map = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".doc": UnstructuredWordDocumentLoader,
        }
        loader_cls = loader_map.get(ext)
        if not loader_cls:
            results.append(f"❌ {filename}: Unsupported format")
            continue

        try:
            loader = loader_cls(f.name)
            raw_docs = loader.load()
            for doc in raw_docs:
                doc.metadata["source"] = filename
                doc.metadata.setdefault("page", doc.metadata.get("page", 0))

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=64,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            chunks = splitter.split_documents(raw_docs)
            count = retriever.add_documents(chunks)

            _ingested_files.append({"name": filename, "chunks": count})
            results.append(f"✅ {filename}: {count} chunks indexed")
        except Exception as e:
            results.append(f"❌ {filename}: {str(e)[:80]}")

    return "\n".join(results)


def get_file_status() -> str:
    if not _ingested_files:
        return "No documents uploaded yet."
    lines = [f"📄 {f['name']} — {f['chunks']} chunks" for f in _ingested_files]
    return "\n".join(lines)


def chat(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """Process a chat message and return updated history."""
    if not message.strip():
        return "", history

    from rag_pipeline import run_agent
    memory_manager = get_memory_manager()
    session = memory_manager.get_or_create(SESSION_ID)

    try:
        result = run_agent(message, session, SESSION_ID)
        answer = result["answer"]

        sources = result.get("sources", [])
        tool_calls = result.get("tool_calls", [])

        if tool_calls:
            tools_str = " · ".join(f"`{t}`" for t in tool_calls)
            answer = f"*Tools used: {tools_str}*\n\n{answer}"

        if sources:
            src_lines = [
                f"  • {s.get('source', '?')} (p.{s.get('page', '?')})"
                for s in sources[:4]
            ]
            answer += "\n\n**Sources:**\n" + "\n".join(src_lines)

        history.append((message, answer))
    except Exception as e:
        history.append((message, f"⚠️ Error: {str(e)}"))

    return "", history


def clear_chat():
    memory_manager = get_memory_manager()
    memory_manager.clear_session(SESSION_ID)
    return []


def full_reset():
    memory_manager = get_memory_manager()
    memory_manager.clear_session(SESSION_ID)
    retriever = get_retriever()
    retriever.clear()
    _ingested_files.clear()
    return [], "All documents and chat history cleared."


# ============================================================
# Gradio UI
# ============================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;700&display=swap');

body, .gradio-container { font-family: 'DM Sans', sans-serif !important; }

.gr-panel { background: #0d1117 !important; border-color: rgba(99,179,237,0.15) !important; }

#title-bar {
  background: linear-gradient(135deg, #0d0d0d, #1a1a2e);
  border: 1px solid rgba(99,179,237,0.25);
  border-radius: 12px;
  padding: 1.2rem 1.5rem;
  margin-bottom: 1rem;
  text-align: center;
}
#title-bar h1 { color: #e2e8f0; font-size: 1.5rem; margin: 0; }
#title-bar p  { color: #4a5568; font-size: 0.85rem; margin: 0.3rem 0 0; font-family: 'DM Mono', monospace; }

.upload-box { border: 2px dashed rgba(99,179,237,0.3) !important; border-radius: 10px !important; }
.file-status { font-family: 'DM Mono', monospace; font-size: 0.82rem; color: #90cdf4; }

.chat-bubble-user   { background: #2d3748 !important; }
.chat-bubble-bot    { background: #0f1f3d !important; }
"""

with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="RAG Agent") as demo:
    gr.HTML("""
    <div id="title-bar">
      <h1>🔍 Multi-Document RAG Agent</h1>
      <p>HYBRID RETRIEVAL · AGENTIC AI</p>
    </div>
    """)

    with gr.Row():
        # ── Sidebar ──
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### 📎 Upload Documents")
            file_upload = gr.File(
                label="Drop PDF / TXT / DOCX files",
                file_count="multiple",
                file_types=[".pdf", ".txt", ".docx"],
                elem_classes=["upload-box"],
            )
            upload_btn = gr.Button("📥 Index Documents", variant="primary")
            upload_status = gr.Textbox(
                label="Upload Status",
                lines=4,
                interactive=False,
                elem_classes=["file-status"],
            )

            gr.Markdown("### 📚 Indexed Files")
            file_status_box = gr.Textbox(
                value=get_file_status,
                label="",
                lines=4,
                interactive=False,
                elem_classes=["file-status"],
                every=5,
            )

            with gr.Row():
                clear_btn = gr.Button("🗑 Clear Chat", scale=1)
                reset_btn = gr.Button("💣 Full Reset", scale=1, variant="stop")

            gr.Markdown("""
            ---
            **Example Queries:**
            - Summarize all documents
            - What are the key findings?
            - Search web for latest AI news
            - Compare topics across documents
            """)

        # ── Chat ──
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=520,
                bubble_full_width=False,
                show_label=False,
                avatar_images=("👤", "🤖"),
                render_markdown=True,
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask anything about your documents or the web…",
                    show_label=False,
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Send →", variant="primary", scale=1)

    # ── Wire events ──
    upload_btn.click(upload_files, inputs=file_upload, outputs=upload_status)
    msg_input.submit(chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
    send_btn.click(chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
    clear_btn.click(clear_chat, outputs=chatbot)
    reset_btn.click(full_reset, outputs=[chatbot, upload_status])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
    )
