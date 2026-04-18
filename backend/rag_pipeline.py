"""
Agentic RAG Pipeline — Powered by LangGraph.

This module implements a true ReAct agent that can autonomously decide 
between document retrieval, web search, and summarization tools.
Supports multi-tool sequences and real-time SSE streaming.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Use AsyncSqliteSaver for async streaming support.
# We create the underlying aiosqlite connection manually so we can keep it
# open for the full process lifetime without relying on context manager cleanup.
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite

from config import get_settings
from memory import SessionMemory
from tools import ALL_TOOLS

logger = logging.getLogger(__name__)
settings = get_settings()

SYSTEM_PROMPT = (
    "You are an expert research assistant with access to a document library and the web.\n\n"
    "Your goal is to provide accurate, well-structured, and cited answers to user questions.\n\n"
    "Available Tools:\n"
    "1. document_search: Use this FIRST for any questions about uploaded documents.\n"
    "2. web_search: Use this for real-time news, recent events, or when the documents lack information.\n"
    "3. summarizer: Use this if the retrieved context is very long or if the user explicitly asks for a summary.\n\n"
    "Guidelines:\n"
    "- ALWAYS cite your sources (e.g., [Source: filename, page X] or [Source: Website Title]).\n"
    "- You can use multiple tools in sequence if needed (e.g., search documents, then search the web to supplement).\n"
    "- If you cannot find the answer in the documents or the web, say so clearly.\n"
    "- Maintain a professional, helpful tone."
)

# Module-level singletons
_agent_instance = None
_agent_memory: Optional[AsyncSqliteSaver] = None
_aiosqlite_conn = None          # keep the raw aiosqlite connection alive
_agent_init_error: Optional[str] = None


def _get_llm():
    """Get configured LLM instance."""
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0.1,
            max_tokens=1024,
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.1,
        )


async def _get_agent():
    """
    Lazily initialize the LangGraph ReAct agent.

    Opens an aiosqlite connection manually and holds a reference to it at
    module level so it stays open for the full process lifetime — avoiding
    the 'Cannot operate on a closed database' error that occurs when the
    AsyncSqliteSaver context manager exits prematurely.
    """
    global _agent_instance, _agent_memory, _aiosqlite_conn, _agent_init_error

    if _agent_init_error:
        raise RuntimeError(f"Agent failed to initialize: {_agent_init_error}")

    if _agent_instance is None:
        try:
            # Open the aiosqlite connection and hold it at module level
            _aiosqlite_conn = await aiosqlite.connect(settings.agent_checkpoint_db)
            _aiosqlite_conn.row_factory = aiosqlite.Row

            # Build the saver directly from the open connection
            _agent_memory = AsyncSqliteSaver(_aiosqlite_conn)

            llm = _get_llm()
            _agent_instance = create_react_agent(
                llm,
                ALL_TOOLS,
                checkpointer=_agent_memory,
                prompt=SYSTEM_PROMPT,
            )
            logger.info("LangGraph ReAct agent initialized successfully.")
        except Exception as e:
            _agent_init_error = str(e)
            logger.error("Agent initialization failed: %s", e, exc_info=True)
            raise RuntimeError(f"Agent failed to initialize: {e}") from e

    return _agent_instance


def _to_ascii(text: str) -> str:
    """Strip obvious bad characters to avoid UI/terminal mojibake."""
    if not text:
        return ""
    return text.replace("\uFFFD", "")


async def run_agent(
    query: str,
    session_memory: SessionMemory,
    session_id: str = "default",
) -> Dict[str, Any]:
    """
    Execute the agent asynchronously.
    Must be awaited — use inside async FastAPI endpoints directly.
    """
    try:
        agent = await _get_agent()
        config = {"configurable": {"thread_id": session_id}}
        inputs = {"messages": [HumanMessage(content=query)]}
        result = await agent.ainvoke(inputs, config=config)

        final_msg = result["messages"][-1]
        answer = final_msg.content if hasattr(final_msg, "content") else str(final_msg)

        tool_calls = []
        sources = []

        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(tc["name"])

            if msg.type == "tool":
                content = str(msg.content)
                if "Source:" in content:
                    for line in content.split("\n"):
                        if "Source:" in line:
                            parts = line.split("|")
                            src = parts[0].replace("Source:", "").strip()
                            page = parts[1].replace("Page:", "").strip() if len(parts) > 1 else ""
                            sources.append({"source": src, "page": page, "snippet": ""})

        unique_sources = []
        seen = set()
        for s in sources:
            key = f"{s['source']}-{s['page']}"
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        session_memory.add_exchange(query, answer, unique_sources)
        return {
            "answer": answer,
            "sources": unique_sources,
            "tool_calls": list(set(tool_calls)),
        }

    except Exception as e:
        logger.error("Agent execution error: %s", e, exc_info=True)
        return {"answer": f"Error: {str(e)}", "sources": [], "tool_calls": []}


async def run_agent_stream(
    query: str,
    session_memory: SessionMemory,
    session_id: str = "default",
) -> AsyncGenerator[str, None]:
    """
    Execute the agent in streaming mode.
    Yields SSE events for tokens and tool activity.
    """
    try:
        agent = await _get_agent()
        config = {"configurable": {"thread_id": session_id}}
        inputs = {"messages": [HumanMessage(content=query)]}

        full_answer = ""
        tool_calls = []
        sources = []

        async for event in agent.astream_events(inputs, config=config, version="v2"):
            kind = event["event"]

            if kind == "on_tool_start":
                tool_name = event["name"]
                tool_input = str(event["data"].get("input", ""))[:50]
                tool_calls.append(tool_name)
                yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name, 'input': tool_input})}\n\n"

            elif kind == "on_tool_end":
                tool_name = event["name"]
                if tool_name == "document_search":
                    output = str(event["data"].get("output", ""))
                    for line in output.split("\n"):
                        if "Source:" in line:
                            parts = line.split("|")
                            src = parts[0].replace("Source:", "").strip()
                            page = parts[1].replace("Page:", "").strip() if len(parts) > 1 else ""
                            sources.append({"source": src, "page": page, "snippet": ""})
                yield f"data: {json.dumps({'type': 'tool_end', 'tool': tool_name})}\n\n"

            elif kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk is None:
                    continue
                content = chunk.content if hasattr(chunk, "content") else ""
                if content:
                    safe = _to_ascii(content)
                    full_answer += safe
                    yield f"data: {json.dumps({'type': 'token', 'content': safe})}\n\n"

        unique_sources = []
        seen = set()
        for s in sources:
            key = f"{s['source']}-{s['page']}"
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        session_memory.add_exchange(query, full_answer, unique_sources)
        yield f"data: {json.dumps({'type': 'done', 'sources': unique_sources, 'tool_calls': list(set(tool_calls))})}\n\n"

    except Exception as e:
        logger.error("Agent streaming error: %s", e, exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


def clear_checkpoint(thread_id: Optional[str] = None):
    """Clear agent checkpoints from SQLite."""
    try:
        conn = sqlite3.connect(settings.agent_checkpoint_db)
        cursor = conn.cursor()
        tables = ["checkpoints", "checkpoint_blobs", "checkpoint_writes"]

        for table in tables:
            try:
                if thread_id:
                    cursor.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
                else:
                    cursor.execute(f"DELETE FROM {table}")
            except sqlite3.OperationalError as e:
                if "no such table" not in str(e).lower():
                    logger.warning("Error clearing table %s: %s", table, e)

        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("Failed to clear checkpoints: %s", e)