#!/usr/bin/env python3
"""
Ingest a single book PDF into chunked FAISS index + JSONL metadata.

Usage (from repo root):
  source venv/bin/activate
  python backend/scripts/ingest_book_king_faiss.py \
    --pdf "/home/shrey/projects/LLM/rag&agents/Agentic-rag/data/book/The_King_of_the_Dark_Chamber.pdf" \
    --chunk-tokens 512 --overlap 128 --batch 64

Outputs:
  backend/db/book_king_faiss.index
  backend/db/book_king_meta.jsonl
"""
import sys, os, argparse, json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import faiss

# add backend to path for repo embedder
sys.path.append("backend")

# PDF extractor
try:
    from PyPDF2 import PdfReader
except Exception as e:
    raise RuntimeError("Please install PyPDF2: pip install PyPDF2") from e

# Try project embedder first, else fallback
USE_REPO_EMBEDDER = False
try:
    from tools.embedder import Embedder
    repo_embedder = Embedder()
    print("Using project embedder: backend.tools.embedder.Embedder")

    def embed_batch(texts):
        """
        Your embedder supports:
          - embed(text)        # likely for single text
          - embed_batch(texts) # batch of texts
        We prefer embed_batch, fallback to embed for single items.
        """
        import numpy as _np

        if hasattr(repo_embedder, "embed_batch"):
            vecs = repo_embedder.embed_batch(texts)
        elif hasattr(repo_embedder, "embed"):
            # embed() probably supports only single strings → loop
            vecs = [repo_embedder.embed(t) for t in texts]
        else:
            raise AttributeError(
                "Embedder has neither embed_batch() nor embed(). "
                "Please check backend/tools/embedder.py"
            )

        # Normalize vectors to np.float32
        return _np.asarray([_np.asarray(v, dtype=_np.float32) for v in vecs], dtype=_np.float32)

    USE_REPO_EMBEDDER = True

except Exception as e:
    print("Project embedder not usable, falling back to sentence-transformers:", e)
    USE_REPO_EMBEDDER = False
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer("all-mpnet-base-v2")

    def embed_batch(texts):
        arr = st_model.encode(texts, convert_to_numpy=True)
        return arr.astype("float32")


# Tokenization: prefer tiktoken (accurate), otherwise whitespace
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def tokenize_ids(text):
        return enc.encode(text)
    def detokenize_ids(ids):
        return enc.decode(ids)
    TOKENIZER = "tiktoken"
except Exception:
    def tokenize_ids(text):
        # fallback: return list of words (strings)
        return text.split()
    def detokenize_ids(tokens):
        if not tokens:
            return ""
        if isinstance(tokens[0], int):
            # unlikely here, but handle
            return " ".join(str(t) for t in tokens)
        return " ".join(tokens)
    TOKENIZER = "whitespace"

def chunk_tokens_by_stride(text, chunk_tokens=512, overlap=128):
    toks = tokenize_ids(text)
    # if tiktoken returns ints
    stride = max(1, chunk_tokens - overlap)
    chunks = []
    if isinstance(toks, list) and toks and isinstance(toks[0], int):
        for start in range(0, len(toks), stride):
            end = min(start + chunk_tokens, len(toks))
            chunk_ids = toks[start:end]
            chunk_text = detokenize_ids(chunk_ids)
            chunks.append((start, end, chunk_text))
            if end == len(toks):
                break
    else:
        # whitespace tokens
        for start in range(0, len(toks), stride):
            end = min(start + chunk_tokens, len(toks))
            chunk_tokens_list = toks[start:end]
            chunk_text = detokenize_ids(chunk_tokens_list)
            chunks.append((start, end, chunk_text))
            if end == len(toks):
                break
    return chunks

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    texts = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        texts.append(txt)
    return "\n\n".join(texts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, required=True)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="backend/db")
    parser.add_argument("--max-pages", type=int, default=None, help="If you want to limit pages (debug)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "book_king_faiss.index"
    meta_path = out_dir / "book_king_meta.jsonl"

    print("Extracting PDF text from:", pdf_path)
    text = extract_text_from_pdf(pdf_path)
    print("Extracted text length (chars):", len(text))

    # optional: split into pseudo-doc sections by double newlines to keep chunk boundaries sane
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    # join every N paras to form larger blocks (avoid tiny chunks)
    blocks = []
    tmp = []
    tmp_chars = 0
    for p in paras:
        tmp.append(p)
        tmp_chars += len(p)
        if tmp_chars > 4000:  # approx threshold, tweakable
            blocks.append(" ".join(tmp))
            tmp = []
            tmp_chars = 0
    if tmp:
        blocks.append(" ".join(tmp))

    print(f"Created {len(blocks)} blocks to chunk (from paragraphs). Tokenizer used: {TOKENIZER}")

    # produce chunk_texts + metadata
    chunk_texts = []
    meta = []
    chunk_id = 0
    for blk_id, blk in enumerate(blocks):
        chunks = chunk_tokens_by_stride(blk, chunk_tokens=args.chunk_tokens, overlap=args.overlap)
        for (s, e, ctext) in chunks:
            meta.append({
                "chunk_id": chunk_id,
                "pdf": str(pdf_path.name),
                "block_id": blk_id,
                "start_token": int(s),
                "end_token": int(e)
            })
            chunk_texts.append(ctext)
            chunk_id += 1

    total = len(chunk_texts)
    print("Total chunks:", total)
    if total == 0:
        print("No chunks produced; exiting.")
        return

    # embed in batches
    # get dimension from sample
    sample_vec = embed_batch([chunk_texts[0]])
    dim = sample_vec.shape[1]
    print("Embedding dim:", dim)
    vectors = np.zeros((total, dim), dtype=np.float32)

    idx = 0
    for i in tqdm(range(0, total, args.batch), desc="Embedding"):
        batch_texts = chunk_texts[i:i+args.batch]
        batch_vecs = embed_batch(batch_texts)
        batch_vecs = np.asarray(batch_vecs, dtype=np.float32)
        vectors[idx: idx + batch_vecs.shape[0], :] = batch_vecs
        idx += batch_vecs.shape[0]

    # build FAISS index (exact)
    print("Building FAISS IndexFlatL2")
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    print("Index ntotal:", index.ntotal)

    # save index and metadata
    print("Saving FAISS index to:", index_path)
    faiss.write_index(index, str(index_path))

    print("Saving metadata jsonl to:", meta_path)
    with open(meta_path, "w", encoding="utf-8") as outf:
        for m, txt in zip(meta, chunk_texts):
            mm = dict(m)
            mm["text"] = txt
            outf.write(json.dumps(mm, ensure_ascii=False) + "\n")

    print("Done. Index and metadata ready.")

if __name__ == "__main__":
    main()
