"""
Hybrid Retriever: BM25 (sparse) + ChromaDB (dense) + FlashRank reranking.
Manual merge replaces EnsembleRetriever for broad LangChain compatibility.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import chromadb
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# FIX: Import FlashrankRerank lazily and guard against missing/broken install.
# On restricted Docker environments the model may fail to download at startup,
# which would crash the entire retriever. We degrade gracefully to no reranking.
_flashrank_available = False
_FlashrankRerank = None

try:
    from langchain_community.document_compressors import FlashrankRerank as _FR
    _FlashrankRerank = _FR
    _flashrank_available = True
    logger.info("FlashRank reranker loaded successfully.")
except ImportError:
    logger.warning(
        "FlashrankRerank not available (langchain_community or flashrank not installed). "
        "Retrieval will proceed without reranking."
    )
except Exception as e:
    logger.warning("FlashrankRerank failed to load (%s). Falling back to no reranking.", e)


class HybridRetriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        self._vectorstore: Optional[Chroma] = None
        self._all_documents: List[Document] = []
        self._bm25_retriever = None

        # FIX: Initialize reranker separately with its own try/except so a
        # model-download failure at runtime doesn't bring down the whole retriever.
        self._reranker = None
        if _flashrank_available and _FlashrankRerank is not None:
            try:
                self._reranker = _FlashrankRerank(
                    model="ms-marco-MiniLM-L-12-v2",
                    top_n=settings.top_k_retrieval,
                )
                logger.info("FlashRank reranker initialized.")
            except Exception as e:
                logger.warning(
                    "FlashRank reranker failed to initialize (model download issue?): %s. "
                    "Retrieval will proceed without reranking.", e
                )
                self._reranker = None

        self._load_existing_collection()

    def add_documents(self, documents: List[Document]) -> int:
        if not documents:
            return 0
        self._all_documents.extend(documents)
        if self._vectorstore is None:
            self._vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                client=self._chroma_client,
                collection_name=settings.chroma_collection_name,
            )
        else:
            self._vectorstore.add_documents(documents)
        self._build_ensemble()
        logger.info("Added %d chunks. Total: %d", len(documents), len(self._all_documents))
        return len(documents)

    def retrieve(self, query: str, k: int | None = None) -> List[Document]:
        k = k or settings.top_k_retrieval
        if self._vectorstore is None:
            logger.warning("No documents indexed yet.")
            return []
        try:
            # BM25 results
            bm25_docs = []
            if self._bm25_retriever is not None:
                bm25_docs = self._bm25_retriever.invoke(query)

            # Dense results — fetch more candidates so reranker has room to work
            candidate_k = max(k * 3, 20)
            dense_docs = self._vectorstore.similarity_search(query, k=candidate_k)

            # Merge: deduplicate by content, BM25 first then dense
            seen = set()
            merged = []
            for doc in bm25_docs + dense_docs:
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)

            # Rerank if available and we have multiple candidates
            if self._reranker is not None and len(merged) > 1:
                try:
                    reranked = self._reranker.compress_documents(merged, query)
                    logger.debug("Reranked %d → %d docs.", len(merged), len(reranked))
                    return list(reranked)[:k]
                except Exception as re:
                    logger.warning(
                        "Reranking failed, falling back to merged list: %s", re
                    )

            return merged[:k]

        except Exception as e:
            logger.error("Retrieval error: %s", e)
            return []

    def dense_only_retrieve(self, query: str, k: int | None = None) -> List[Document]:
        k = k or settings.top_k_retrieval
        if self._vectorstore is None:
            return []
        return self._vectorstore.similarity_search(query, k=k)

    def clear(self) -> None:
        try:
            self._chroma_client.delete_collection(settings.chroma_collection_name)
        except Exception:
            pass
        self._vectorstore = None
        self._all_documents = []
        self._bm25_retriever = None
        logger.info("Vector store cleared.")

    @property
    def document_count(self) -> int:
        return len(self._all_documents)

    def _load_existing_collection(self) -> None:
        try:
            existing = self._chroma_client.get_collection(
                name=settings.chroma_collection_name
            )
            if existing.count() > 0:
                self._vectorstore = Chroma(
                    client=self._chroma_client,
                    collection_name=settings.chroma_collection_name,
                    embedding_function=self.embeddings,
                )
                raw = existing.get(include=["documents", "metadatas"])
                for text, meta in zip(raw["documents"], raw["metadatas"]):
                    self._all_documents.append(
                        Document(page_content=text, metadata=meta or {})
                    )
                self._build_ensemble()
                logger.info(
                    "Loaded existing collection with %d chunks.", len(self._all_documents)
                )
        except Exception as e:
            logger.debug("No existing collection found: %s", e)

    def _build_ensemble(self) -> None:
        if not self._all_documents:
            return
        self._bm25_retriever = BM25Retriever.from_documents(
            self._all_documents,
            k=settings.top_k_retrieval,
        )


_hybrid_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever