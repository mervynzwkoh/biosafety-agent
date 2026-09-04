"""Conversation trajectory store with history management and windowing."""

import json
import os
import sqlite3
from typing import Dict, List, Optional

from biosafety_defense.defense_agent.schemas import ConversationTurn


class ConversationStore:
    """Manages dialogue history across turns for each conversation."""

    def __init__(self, db_path: Optional[str] = None):
        self._memory_store: Dict[str, List[ConversationTurn]] = {}
        self.db_path = db_path
        if db_path:
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initialize SQLite schema if database path provided."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    turn INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_id ON conversation_turns(conversation_id)"
            )

    def load(self, conversation_id: str) -> List[ConversationTurn]:
        """Load conversation turns in chronological order."""
        if conversation_id in self._memory_store:
            return list(self._memory_store[conversation_id])

        if self.db_path and os.path.exists(self.db_path):
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT turn, role, content FROM conversation_turns WHERE conversation_id = ? ORDER BY id ASC",
                    (conversation_id,),
                )
                rows = cursor.fetchall()
                turns = [ConversationTurn(turn=r[0], role=r[1], content=r[2]) for r in rows]
                self._memory_store[conversation_id] = turns
                return list(turns)

        return []

    def append(
        self,
        conversation_id: str,
        turn: int,
        user_text: str,
        delivered_assistant_text: str,
    ) -> None:
        """
        Append user message and delivered assistant response to conversation history.
        Per Section 24, delivered_assistant_text is what the user actually received.
        """
        if conversation_id not in self._memory_store:
            self._memory_store[conversation_id] = self.load(conversation_id)

        user_turn = ConversationTurn(turn=turn, role="user", content=user_text)
        asst_turn = ConversationTurn(
            turn=turn, role="assistant", content=delivered_assistant_text
        )

        self._memory_store[conversation_id].extend([user_turn, asst_turn])

        if self.db_path:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO conversation_turns (conversation_id, turn, role, content) VALUES (?, ?, ?, ?)",
                    (conversation_id, turn, "user", user_text),
                )
                conn.execute(
                    "INSERT INTO conversation_turns (conversation_id, turn, role, content) VALUES (?, ?, ?, ?)",
                    (conversation_id, turn, "assistant", delivered_assistant_text),
                )

    def get_context_window(
        self,
        conversation_id: str,
        max_raw_turns: int = 10,
    ) -> List[ConversationTurn]:
        """
        Return recent raw turns per Section 25.
        Older turns are represented via persistent safety state summary.
        """
        full_history = self.load(conversation_id)
        if len(full_history) <= max_raw_turns * 2:
            return full_history
        # Return the last max_raw_turns (each turn is user + assistant)
        return full_history[-(max_raw_turns * 2) :]
