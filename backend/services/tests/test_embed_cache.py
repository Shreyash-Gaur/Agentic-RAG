# backend/services/tests/test_embed_cache.py
import os
import sys
from pathlib import Path

# Ensure repo root in sys.path for imports
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from backend.services.embed_cache_service import EmbedCacheService

TEST_DB = "backend/db/test_embed_cache.sqlite"

def setup_function():
    try:
        os.remove(TEST_DB)
    except FileNotFoundError:
        pass

def teardown_function():
    try:
        os.remove(TEST_DB)
    except FileNotFoundError:
        pass

def test_embed_cache_single_and_batch():
    cache = EmbedCacheService(db_path=TEST_DB)
    text = "hello world"
    model = "test-model:1"
    assert cache.has(text, model) is False

    vec = np.arange(8, dtype="float32")
    cache.set_vector(text, model, vec)
    assert cache.has(text, model) is True

    got = cache.get_vector(text, model)
    assert got is not None
    assert got.shape == (8,)
    assert np.allclose(got, vec)

    # batch test
    texts = ["a","b", text]
    vecs = [np.ones(3,dtype="float32"), np.ones(3,dtype="float32")*2, vec]
    cache.set_batch(texts, model, vecs)
    outs, keys = cache.get_batch(texts, model)
    assert len(outs) == 3
    assert outs[0] is not None and outs[0].shape[0] == 3
    assert outs[2].shape == (8,)

    # count should be >=3
    assert cache.count() >= 3

    cache.close()
