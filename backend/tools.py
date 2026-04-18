"""
Agent Tools for the RAG Agent.

Tools:
  1. document_search  — hybrid retrieval from the vector store
  2. web_search       — real-time search via Tavily API
  3. summarizer       — condenses long context using the LLM
"""
from __future__ import annotations

import logging
from typing import Annotated, Any, List

from langchain_core.tools import tool
from langchain_core.documents import Document

from config import get_settings
from retriever import get_retriever

logger = logging.getLogger(__name__)
settings = get_settings()


# ============================================================
# Tool 1 — Document Search (Hybrid RAG)
# ============================================================

@tool
def document_search(
    query: Annotated[str, "The search query to find relevant document passages"]
) -> str:
    """
    Search the ingested document collection using hybrid BM25 + dense retrieval.
    Returns the most relevant passages along with their source metadata.
    Use this for questions about uploaded documents.
    """
    retriever = get_retriever()
    if retriever.document_count == 0:
        return "No documents have been uploaded yet. Please upload documents first."

    docs: List[Document] = retriever.retrieve(query)
    if not docs:
        return "No relevant passages found for this query."

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        results.append(
            f"[Passage {i}] Source: {source} | Page: {page}\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(results)


# ============================================================
# Tool 2 — Web Search (Tavily)
# ============================================================

@tool
def web_search(
    query: Annotated[str, "The web search query for real-time information"]
) -> str:
    """
    Search the web for current, real-time information using Tavily.
    Use this when the documents don't contain the needed information,
    or when the user asks about recent events / live data.
    """
    if not settings.tavily_api_key or settings.tavily_api_key == "your_tavily_api_key_here":
        return (
            "Web search is not configured. "
            "Set TAVILY_API_KEY in your .env file to enable real-time search."
        )

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=True,
        )

        results = []
        if response.get("answer"):
            results.append(f"Quick answer: {response['answer']}\n")

        for r in response.get("results", [])[:4]:
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            content = r.get("content", "")[:400]
            results.append(f"**{title}**\n{url}\n{content}")

        return "\n\n---\n\n".join(results) if results else "No web results found."

    except ImportError:
        return "Tavily package not installed. Run: pip install tavily-python"
    except Exception as e:
        logger.error("Web search error: %s", e)
        return f"Web search failed: {str(e)}"


# ============================================================
# Tool 3 — Summarizer
# ============================================================

@tool
def summarizer(
    text: Annotated[str, "The long text content to summarize"]
) -> str:
    """
    Condense a long piece of text into a concise, structured summary.
    Use this when context is too long or when the user explicitly asks for a summary.
    """
    if not text or len(text.strip()) < 50:
        return "Text is too short to summarize."

    max_chars = 8000
    truncated = text[:max_chars]
    if len(text) > max_chars:
        truncated += "\n\n[... text truncated for summarization ...]"

    try:
        llm = _get_llm()
        prompt = (
            "Please provide a concise, structured summary of the following text. "
            "Include:\n"
            "- Main topic/purpose\n"
            "- Key points (bullet list)\n"
            "- Important facts or conclusions\n\n"
            f"TEXT:\n{truncated}\n\n"
            "SUMMARY:"
        )
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error("Summarizer error: %s", e)
        sentences = text.split(". ")
        return ". ".join(sentences[:5]) + "."


# ============================================================
# Helper
# ============================================================

def _get_llm() -> Any:
    """Get the configured LLM instance."""
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0.3,
        )
    else:
        # FIX: use langchain_ollama instead of deprecated langchain_community.llms.Ollama
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.3,
        )


# Exported tool list
ALL_TOOLS = [document_search, web_search, summarizer]
