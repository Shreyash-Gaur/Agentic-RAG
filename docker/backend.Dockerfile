# Use python 3.11 slim for smaller image size
FROM python:3.11-slim

# Set environment variables to prevent pyc files and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (build-essential for compiling some python libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend source code into /app/backend
COPY backend/ ./backend/

# Create directory for local DBs/logs to ensure permissions
RUN mkdir -p backend/db backend/logs

# Expose port
EXPOSE 8000

# Run from /app so "backend.main" is resolvable
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]