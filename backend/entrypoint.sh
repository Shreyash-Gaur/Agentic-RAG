#!/bin/bash

# 1. Start the Watcher in the background (&)
# We log its output to a separate file so it doesn't clutter the main logs too much
echo "Starting File Watcher on /app/data..."
python backend/scripts/ingest_watch.py --watch data --interval 10 > /app/watcher.log 2>&1 &

# 2. Start the Main API Server (Uvicorn)
echo "Starting Uvicorn Server..."
exec uvicorn backend.main:APP --host 0.0.0.0 --port 8000