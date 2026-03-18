FROM python:3.11-slim
WORKDIR /app

# Install system dependencies needed for FAISS, numpy, and sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caches unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ ./backend/

# Copy entrypoint script and make it executable
COPY backend/entrypoint.sh .
RUN chmod +x entrypoint.sh

# FIX: was backend/db/emb_cache — must match EMBEDDING_CACHE_DB path in .env
# which points to backend/db/embedding_cache/embed_cache.sqlite
RUN mkdir -p \
    backend/db/vector_data \
    backend/db/memory \
    backend/db/embedding_cache

# Create the watch directory so the watcher starts without errors
# even before the host volume is mounted
RUN mkdir -p knowledge

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
