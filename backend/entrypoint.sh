#!/bin/bash
set -e
 
# Start the file watcher in the background.
# It watches the 'knowledge' directory — must match WATCH_DIR in .env.
# Output is logged separately so it doesn't pollute the uvicorn logs.
echo "Starting File Watcher on /app/knowledge..."
python backend/scripts/ingest_vector_watch.py \
    --watch knowledge \
    --interval 10 \
    > /app/watcher.log 2>&1 &
 
# Start the main API server in the foreground.
# Using exec replaces the shell process with uvicorn so that SIGTERM
# from Docker (on container stop) is received directly by uvicorn
# rather than being swallowed by the shell — this ensures clean shutdown.
echo "Starting Uvicorn Server..."
exec uvicorn backend.main:APP --host 0.0.0.0 --port 8000