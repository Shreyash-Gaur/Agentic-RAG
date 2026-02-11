# backend/services/memory_service.py
"""
MemoryService (SQLite-backed, thread-safe)

Features:
 - Keeps per-conversation history in memory for fast read.
 - Persists all turns to SQLite for durability.
 - Bounded history (max_history) kept in-memory per conversation.
 - Safe for multi-threaded FastAPI usage (uses RLock).
 - Simple API: add_turn, get_history, clear_history, export_history, import_history, close.
"""

from __future__ import annotations
import sqlite3
import threading
import json
import time
import os
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger("agentic-rag.memory")

DEFAULT_DB_PATH = "backend/db/memory/memory_store.sqlite"


class MemoryService:
    def __init__(
        self,
        max_history: int = 20,
        use_sqlite: bool = True,
        db_path: str = DEFAULT_DB_PATH,
        preload: bool = True,
    ):
        """
        Args:
            max_history: max number of turns to keep in-memory per conversation.
            use_sqlite: persist turns to SQLite when True.
            db_path: path to sqlite file.
            preload: if True, preload existing DB rows into in-memory store on startup.
        """
        self.max_history = int(max_history)
        self.use_sqlite = bool(use_sqlite)
        self.db_path = db_path
        self._lock = threading.RLock()
        self._store: Dict[str, List[Dict[str, Any]]] = {}

        # Ensure DB directory exists
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._conn: Optional[sqlite3.Connection] = None
        if self.use_sqlite:
            self._init_db()
            if preload:
                try:
                    self._load_all_to_memory()
                except Exception as e:
                    logger.warning("Preload memory failed: %s", e)

    # ---------------------------
    # Database bootstrap
    # ---------------------------
    def _init_db(self) -> None:
        # Connect with WAL mode for concurrency
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable WAL
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        # Create table if not exists
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL,
                ts REAL NOT NULL,
                role TEXT,
                content TEXT,
                meta TEXT
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_id ON memory(conv_id);")
        conn.commit()
        self._conn = conn

    # ---------------------------
    # Preload DB into in-memory store
    # ---------------------------
    def _load_all_to_memory(self) -> None:
        if not self._conn:
            return
        cur = self._conn.cursor()
        cur.execute("SELECT conv_id, ts, role, content, meta FROM memory ORDER BY id ASC")
        rows = cur.fetchall()
        with self._lock:
            for r in rows:
                conv_id = r["conv_id"]
                turn = {
                    "ts": float(r["ts"]),
                    "role": r["role"],
                    "content": r["content"],
                    "meta": json.loads(r["meta"]) if r["meta"] else {},
                }
                self._store.setdefault(conv_id, []).append(turn)
            # Trim histories to max_history
            for cid, turns in list(self._store.items()):
                if len(turns) > self.max_history:
                    self._store[cid] = turns[-self.max_history :]

    # ---------------------------
    # Add turn
    # ---------------------------
    def add_turn(
        self,
        conv_id: str,
        role: str,
        content: str,
        meta: Optional[Dict[str, Any]] = None,
        ts: Optional[float] = None,
    ) -> None:
        """
        Add a conversation turn.

        Args:
            conv_id: conversation identifier (string)
            role: role string, e.g. "user" or "assistant"
            content: text content of the turn
            meta: optional dict with extra metadata (sources, scores, etc.)
            ts: optional epoch seconds timestamp (if omitted, current time used)
        """
        ts = float(ts or time.time())
        meta = meta or {}

        turn = {"ts": ts, "role": role, "content": content, "meta": meta}

        with self._lock:
            buf = self._store.setdefault(conv_id, [])
            buf.append(turn)
            # trim in-memory
            if len(buf) > self.max_history:
                self._store[conv_id] = buf[-self.max_history :]

            # persist to sqlite
            if self.use_sqlite and self._conn:
                try:
                    cur = self._conn.cursor()
                    cur.execute(
                        "INSERT INTO memory (conv_id, ts, role, content, meta) VALUES (?, ?, ?, ?, ?)",
                        (conv_id, ts, role, content, json.dumps(meta, ensure_ascii=False)),
                    )
                    self._conn.commit()
                except Exception as e:
                    logger.exception("Failed to persist memory to sqlite: %s", e)

    # ---------------------------
    # Get history
    # ---------------------------
    def get_history(self, conv_id: str) -> List[Dict[str, Any]]:
        """
        Return the in-memory history for conv_id (most recent first preserved order).
        """
        with self._lock:
            return list(self._store.get(conv_id, []))

    # ---------------------------
    # Clear history
    # ---------------------------
    def clear_history(self, conv_id: str) -> None:
        """
        Clear in-memory and persisted history for conv_id.
        """
        with self._lock:
            self._store.pop(conv_id, None)
            if self.use_sqlite and self._conn:
                try:
                    cur = self._conn.cursor()
                    cur.execute("DELETE FROM memory WHERE conv_id = ?", (conv_id,))
                    self._conn.commit()
                except Exception as e:
                    logger.exception("Failed to delete memory rows: %s", e)

    # ---------------------------
    # Export / Import
    # ---------------------------
    def export_history(self, conv_id: str) -> str:
        """
        Return JSON string of conversation history (pretty-printed).
        """
        hist = self.get_history(conv_id)
        return json.dumps(hist, ensure_ascii=False, indent=2)

    def import_history(self, conv_id: str, turns: List[Dict[str, Any]]) -> None:
        """
        Import a list of turns into a conversation.
        Each turn should be a dict with keys: ts, role, content, meta (optional).
        This will persist to DB if enabled.
        """
        if not isinstance(turns, list):
            raise ValueError("turns must be a list of turn dicts")

        with self._lock:
            # replace in-memory (keep last max_history)
            self._store[conv_id] = turns[-self.max_history :]

            if self.use_sqlite and self._conn:
                try:
                    cur = self._conn.cursor()
                    for t in self._store[conv_id]:
                        cur.execute(
                            "INSERT INTO memory (conv_id, ts, role, content, meta) VALUES (?, ?, ?, ?, ?)",
                            (
                                conv_id,
                                float(t.get("ts", time.time())),
                                t.get("role", ""),
                                t.get("content", ""),
                                json.dumps(t.get("meta", {}), ensure_ascii=False),
                            ),
                        )
                    self._conn.commit()
                except Exception as e:
                    logger.exception("Failed to import history to sqlite: %s", e)

    # ---------------------------
    # Close
    # ---------------------------
    def close(self) -> None:
        """
        Close DB connection. Call on application shutdown.
        """
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
