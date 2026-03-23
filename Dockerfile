FROM python:3.11-slim
WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# dos2unix: safety net for CRLF line endings in entrypoint.sh
# (in case .gitattributes wasn't respected on the developer's machine)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# ── PyTorch — GPU — STEP 2 of 2 ──────────────────────────────────────────────
# Two blocks below: only ONE should be active at a time.
#
# CPU users (default):
#   Keep the CPU block active, leave CUDA block commented.
#
# GPU users:
#   Comment out the CPU block, uncomment the CUDA block.
#   Also uncomment the `deploy` block in docker-compose.yml (step 1).
#   Check your CUDA version with `nvidia-smi` and pick the matching wheel:
#   https://pytorch.org/get-started/locally/

# ── ACTIVE: CPU torch (~700 MB) ──────────────────────────────────────────────
# RUN pip install --no-cache-dir \
#     torch \
#     --index-url https://download.pytorch.org/whl/cpu

# ── INACTIVE: CUDA 12.4 torch (~2.3 GB) — uncomment for GPU ─────────────────
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cu124

# ── Python dependencies ───────────────────────────────────────────────────────
# This layer is cached unless requirements.txt changes.
# torch is already installed above so pip skips it here.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY backend/  ./backend/
COPY frontend/ ./frontend/

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY backend/entrypoint.sh .
# dos2unix converts CRLF → LF in case the file was checked out on Windows.
# Safety net on top of .gitattributes — belt and suspenders.
RUN dos2unix entrypoint.sh && chmod +x entrypoint.sh

# ── Runtime directories ───────────────────────────────────────────────────────
# Created here so they exist before volumes are mounted.
RUN mkdir -p \
    backend/db/vector_data \
    backend/db/memory \
    backend/db/embedding_cache \
    knowledge

EXPOSE 8000 8001

# Default command: backend (watcher + uvicorn).
# The frontend service overrides this via `command:` in docker-compose.yml.
CMD ["./entrypoint.sh"]