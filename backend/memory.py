"""
Conversation Memory Management.

Per-session chat history using LangChain's ChatMessageHistory.
Each session_id gets its own memory window to support multiple concurrent users.

FIX: After a server restart, the in-RAM MemoryManager is empty even though
LangGraph's SQLite checkpoints still exist. This module now rehydrates the
structured message log from the checkpoint DB when a known session is requested,
so /history returns correct data and the sliding-window context is restored.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ChatMessage(BaseModel):
    role: str          # "user" | "assistant"
    content: str
    timestamp: str
    sources: Optional[List[dict]] = None


class SessionMemory:
    """Wraps LangChain ChatMessageHistory + structured log for one session."""

    def __init__(self, session_id: str, window_size: int):
        self.session_id = session_id
        self.window_size = window_size
        self.lc_memory = ChatMessageHistory()
        self.message_log: List[ChatMessage] = []
        self.created_at = datetime.utcnow().isoformat()

    def add_exchange(
        self,
        user_message: str,
        ai_response: str,
        sources: Optional[List[dict]] = None,
    ) -> None:
        """Record a human/AI exchange in both LangChain memory and the log."""
        self.lc_memory.add_user_message(user_message)
        self.lc_memory.add_ai_message(ai_response)

        ts = datetime.utcnow().isoformat()
        self.message_log.append(
            ChatMessage(role="user", content=user_message, timestamp=ts)
        )
        self.message_log.append(
            ChatMessage(
                role="assistant",
                content=ai_response,
                timestamp=ts,
                sources=sources,
            )
        )

    def get_history(self) -> List[ChatMessage]:
        return self.message_log

    def get_lc_history(self) -> List[BaseMessage]:
        """Return last N messages (sliding window) for use in the agent."""
        messages = self.lc_memory.messages
        return messages[-(self.window_size * 2):]

    def clear(self) -> None:
        self.lc_memory = ChatMessageHistory()
        self.message_log = []

    @property
    def turn_count(self) -> int:
        return len(self.message_log) // 2


# ============================================================
# SQLite checkpoint rehydration helpers
# ============================================================

def _load_messages_from_checkpoint(session_id: str) -> List[ChatMessage]:
    """
    Attempt to read conversation turns from the LangGraph SQLite checkpoint DB.

    LangGraph stores full message state in the checkpoints table as a blob.
    We use a best-effort approach: read the latest checkpoint for the thread
    and reconstruct HumanMessage / AIMessage pairs into ChatMessage objects.
    Returns an empty list if the DB doesn't exist, the table is missing, or
    deserialization fails — so callers never need to handle exceptions.
    """
    try:
        import pickle  # LangGraph uses pickle for checkpoint blobs

        conn = sqlite3.connect(settings.agent_checkpoint_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Fetch the most recent checkpoint blob for this thread
        cursor.execute(
            """
            SELECT checkpoint FROM checkpoints
            WHERE thread_id = ?
            ORDER BY checkpoint_id DESC
            LIMIT 1
            """,
            (session_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return []

        state = pickle.loads(row["checkpoint"])  # noqa: S301
        raw_messages = state.get("channel_values", {}).get("messages", [])

        result: List[ChatMessage] = []
        ts = datetime.utcnow().isoformat()

        for msg in raw_messages:
            if isinstance(msg, HumanMessage):
                result.append(ChatMessage(role="user", content=msg.content, timestamp=ts))
            elif isinstance(msg, AIMessage) and msg.content:
                result.append(ChatMessage(role="assistant", content=msg.content, timestamp=ts))

        logger.info(
            "Rehydrated %d messages for session %s from checkpoint DB.",
            len(result), session_id,
        )
        return result

    except sqlite3.OperationalError:
        # Table doesn't exist yet — fresh DB, nothing to rehydrate
        return []
    except Exception as e:
        logger.debug("Could not rehydrate session %s from checkpoint: %s", session_id, e)
        return []


def _session_exists_in_checkpoint(session_id: str) -> bool:
    """Return True if the SQLite checkpoint DB has any record for this thread_id."""
    try:
        conn = sqlite3.connect(settings.agent_checkpoint_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM checkpoints WHERE thread_id = ? LIMIT 1", (session_id,)
        )
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    except Exception:
        return False


# ============================================================
# MemoryManager
# ============================================================

class MemoryManager:
    """Registry of per-session memory objects."""

    def __init__(self):
        self._sessions: Dict[str, SessionMemory] = {}

    def get_or_create(self, session_id: str) -> SessionMemory:
        if session_id not in self._sessions:
            session = SessionMemory(
                session_id=session_id,
                window_size=settings.memory_window_size,
            )

            # FIX: Rehydrate from SQLite checkpoint if the session existed before
            # a server restart. This restores /history and the sliding-window context.
            prior_messages = _load_messages_from_checkpoint(session_id)
            if prior_messages:
                session.message_log = prior_messages
                # Also restore LangChain in-memory history for context window
                for msg in prior_messages:
                    if msg.role == "user":
                        session.lc_memory.add_user_message(msg.content)
                    else:
                        session.lc_memory.add_ai_message(msg.content)

            self._sessions[session_id] = session
            logger.info(
                "Session %s loaded (%d prior turns).",
                session_id, session.turn_count,
            )
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[SessionMemory]:
        """
        Return an existing in-RAM session, or attempt to rehydrate from
        the checkpoint DB if the session_id is known but not yet loaded.
        """
        if session_id in self._sessions:
            return self._sessions[session_id]

        # FIX: Before returning None (which causes /history to 404), check
        # the checkpoint DB — the session may have survived a server restart.
        if _session_exists_in_checkpoint(session_id):
            return self.get_or_create(session_id)

        return None

    def clear_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            return True
        return False

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)


# Singleton
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager