# backend/services/tests/test_memory.py
import os
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so "backend" package can be imported reliably.
# File location: backend/services/tests/test_memory.py
REPO_ROOT = Path(__file__).resolve().parents[3]  # go up from tests -> services -> backend -> repo root
sys.path.insert(0, str(REPO_ROOT))

import time
import pytest

from backend.services.memory_service import MemoryService

TEST_DB = "backend/db/test_memory.sqlite"

def setup_function():
    # ensure fresh DB for each test run
    try:
        os.remove(TEST_DB)
    except FileNotFoundError:
        pass

def teardown_function():
    try:
        os.remove(TEST_DB)
    except FileNotFoundError:
        pass

def test_memory_service_basic():
    # create service with preload disabled for deterministic test
    m = MemoryService(max_history=3, use_sqlite=True, db_path=TEST_DB, preload=False)
    m.add_turn("t1", "user", "q1")
    m.add_turn("t1", "assistant", "a1")
    h = m.get_history("t1")

    assert isinstance(h, list)
    assert len(h) == 2
    assert h[0]["role"] == "user"
    assert "q1" in h[0]["content"]
    assert h[1]["role"] == "assistant"

    # clear and verify empty
    m.clear_history("t1")
    assert m.get_history("t1") == []

    # add many turns and ensure trimming to max_history
    for i in range(6):
        m.add_turn("t2", "user", f"msg-{i}")
    h2 = m.get_history("t2")
    assert len(h2) == 3  # max_history
    # newest should be last 3 messages (msg-3, msg-4, msg-5)
    assert "msg-5" in h2[-1]["content"]

    m.close()

def test_memory_persistence():
    # test persistence across service instances
    m1 = MemoryService(max_history=5, use_sqlite=True, db_path=TEST_DB, preload=True)
    m1.add_turn("convA", "user", "hello")
    m1.add_turn("convA", "assistant", "hi")
    m1.close()

    # new instance should preload previous rows
    m2 = MemoryService(max_history=5, use_sqlite=True, db_path=TEST_DB, preload=True)
    hist = m2.get_history("convA")
    assert len(hist) >= 2
    assert hist[0]["role"] == "user"
    m2.clear_history("convA")
    m2.close()
