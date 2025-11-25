# backend/services/memory_service.py
"""
MemoryService (Production Version)
- Thread-safe
- SQLite-backed conversation history
- Supports export/import
- Works with Chainlit session_id
"""

from __future__ import annotations
import sqlite3
import threading
import json
import time
import os
from typing import List, Dict, Optional, Any

DEFAULT_DB_PATH = "backend/db/memory_store.sqlite"


class MemoryService:
    def __init__(
        self,
        max_history: int = 20,
        use_sqlite: bool = True,
        db_path: str = DEFAULT_DB_PATH,
    ):
        self.max_history = int(max_history)
        self.use_sqlite = bool(use_sqlite)
        self.db_path = db_path

        # ensure db directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self._lock = threading.RLock()
        self._store: Dict[str, List[Dict[str, Any]]] = {}

        if self.use_sqlite:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._init_db()
            self._load_all_to_memory()
        else:
            self._conn = None

    # ---------------------------
    # DB Init
    # ---------------------------
    def _init_db(self):
        cur = self._conn.cursor()
        cur.execute(
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
        cur.execute("CREATE INDEX IF NOT EXISTS idx_conv_id ON memory(conv_id);")
        self._conn.commit()

    # ---------------------------
    # Preload memory for fast access
    # ---------------------------
    def _load_all_to_memory(self):
        cur = self._conn.cursor()
        cur.execute("SELECT conv_id, ts, role, content, meta FROM memory ORDER BY id ASC")
        rows = cur.fetchall()

        for r in rows:
            turn = {
                "ts": float(r["ts"]),
                "role": r["role"],
                "content": r["content"],
                "meta": json.loads(r["meta"]) if r["meta"] else None,
            }
            self._store.setdefault(r["conv_id"], []).append(turn)

        # Trim oversize histories
        for cid in list(self._store.keys()):
            turns = self._store[cid]
            if len(turns) > self.max_history:
                self._store[cid] = turns[-self.max_history:]

    # ---------------------------
    # Public API
    # ---------------------------
    def add_turn(
        self,
        conv_id: str,
        role: str,
        content: str,
        meta: Optional[Dict[str, Any]] = None,
        ts: Optional[float] = None,
    ):
        ts = ts or time.time()
        turn = {"ts": ts, "role": role, "content": content, "meta": meta or {}}

        with self._lock:
            buf = self._store.setdefault(conv_id, [])
            buf.append(turn)

            if len(buf) > self.max_history:
                self._store[conv_id] = buf[-self.max_history:]

            if self.use_sqlite:
                cur = self._conn.cursor()
                cur.execute(
                    """
                    INSERT INTO memory (conv_id, ts, role, content, meta)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (conv_id, ts, role, content, json.dumps(meta or {})),
                )
                self._conn.commit()

    def get_history(self, conv_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._store.get(conv_id, []))

    def clear_history(self, conv_id: str):
        with self._lock:
            self._store.pop(conv_id, None)

            if self.use_sqlite:
                cur = self._conn.cursor()
                cur.execute("DELETE FROM memory WHERE conv_id = ?", (conv_id,))
                self._conn.commit()

    def export_history(self, conv_id: str) -> str:
        return json.dumps(self.get_history(conv_id), ensure_ascii=False, indent=2)

    def import_history(self, conv_id: str, turns: List[Dict[str, Any]]):
        self.clear_history(conv_id)
        with self._lock:
            self._store[conv_id] = turns[-self.max_history:]

            if self.use_sqlite:
                cur = self._conn.cursor()
                for t in self._store[conv_id]:
                    cur.execute(
                        """
                        INSERT INTO memory (conv_id, ts, role, content, meta)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            conv_id,
                            t.get("ts", time.time()),
                            t.get("role"),
                            t.get("content"),
                            json.dumps(t.get("meta", {})),
                        ),
                    )
                self._conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
