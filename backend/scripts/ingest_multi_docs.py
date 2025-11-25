#!/usr/bin/env python3
# backend/scripts/ingest_multi_docs.py
import argparse
import os
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import faiss
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # add backend to path
from tools.pdf_ingest import extract_text_from_pdf  # if present, else fallback
from tools.embedder import Embedder
from tools.embed_cache import EmbeddingCache
import tiktoken
ENC = None
try:
    ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENC = None

def tokenize(text):
    if ENC:
        return ENC.encode(text)
    return text.split()

def detokenize(tokens):
    if ENC:
        return ENC.decode(tokens)
    return " ".join(tokens)

def chunk_text(text, chunk_tokens=512, overlap=128):
    toks = tokenize(text)
    stride = chunk_tokens - overlap
    chunks = []
    if isinstance(toks, list) and toks and isinstance(toks[0], int):
        n = len(toks)
        for start in range(0, n, stride):
            end = min(start + chunk_tokens, n)
            chunk_ids = toks[start:end]
            chunks.append(detokenize(chunk_ids))
            if end == n:
                break
    else:
        n = len(toks)
        for start in range(0, n, stride):
            end = min(start + chunk_tokens, n)
            chunk_tokens_list = toks[start:end]
            chunks.append(detokenize(chunk_tokens_list))
            if end == n:
                break
    return chunks

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True)
    p.add_argument("--output-dir", default="backend/db")
    p.add_argument("--chunk-tokens", type=int, default=512)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--index-name", default="multi_docs_faiss.index")
    args = p.parse_args()

    folder = Path(args.folder)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / args.index_name
    meta_path = out_dir / (args.index_name.replace(".index","_meta.jsonl"))

    embedder = Embedder()
    cache = EmbeddingCache(cache_dir=str(out_dir / "emb_cache"))

    # collect all chunks and metas
    all_chunks = []
    all_meta = []
    chunk_id = 0
    for file in folder.iterdir():
        if file.suffix.lower() not in (".pdf",".txt"):
            continue
        if file.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(str(file))
        else:
            text = file.read_text(encoding="utf8")
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        # chunk each block
        for blk_id, blk in enumerate(blocks):
            chunks = chunk_text(blk, chunk_tokens=args.chunk_tokens, overlap=args.overlap)
            for c in chunks:
                meta = {"chunk_id": chunk_id, "doc": file.name, "block_id": blk_id}
                all_chunks.append(c)
                all_meta.append(meta)
                chunk_id += 1

    total = len(all_chunks)
    print(f"Total chunks: {total}")
    if total == 0:
        return

    # Embedding dimension from sample
    sample_vec = embedder.embed_batch([all_chunks[0]])
    dim = sample_vec.shape[1]
    # Try to load existing index to append
    if index_path.exists():
        index = faiss.read_index(str(index_path))
        existing = index.ntotal
        print("Existing index ntotal:", existing)
        # We'll append vectors
        vectors = None
    else:
        index = faiss.IndexFlatL2(dim)
        existing = 0

    # prepare storage for vectors in memory in batches
    vectors_all = np.zeros((total, dim), dtype=np.float32)
    for i in range(0, total, args.batch):
        batch_texts = all_chunks[i:i+args.batch]
        # check cache per text
        vecs = []
        to_compute = []
        compute_idx = []
        for j, txt in enumerate(batch_texts):
            cache_key = cache.key_for_text(txt)
            cached = cache.get(cache_key)
            if cached is not None:
                vecs.append(cached)
            else:
                vecs.append(None)
                to_compute.append(txt)
                compute_idx.append(j)
        if to_compute:
            computed = embedder.embed_batch(to_compute)
            # store computed into vecs and cache
            ci = 0
            for k in compute_idx:
                vecs[k] = computed[ci]
                cache.set(cache.key_for_text(batch_texts[k]), computed[ci])
                ci += 1
        # write into vectors_all
        for local_j, v in enumerate(vecs):
            vectors_all[i+local_j, :] = np.asarray(v, dtype=np.float32)

    # Add vectors to index
    if existing > 0:
        # index already has vectors, we need to create an IndexIDMap to set ids if needed.
        index.add(vectors_all)
    else:
        index.add(vectors_all)

    print("Index ntotal:", index.ntotal)
    faiss.write_index(index, str(index_path))

    # write metadata jsonl (overwrite)
    with open(meta_path, "w", encoding="utf8") as outf:
        for m, txt in zip(all_meta, all_chunks):
            mm = dict(m)
            mm["text"] = txt
            outf.write(json.dumps(mm, ensure_ascii=False) + "\n")

    print("Ingestion complete. index:", index_path, "meta:", meta_path)

if __name__ == "__main__":
    main()
