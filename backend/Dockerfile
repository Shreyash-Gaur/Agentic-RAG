FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Backend Code
COPY backend/ ./backend/

# Copy the Entrypoint Script & Make it executable
COPY backend/entrypoint.sh .
RUN chmod +x entrypoint.sh

# Create necessary folders
RUN mkdir -p backend/db
RUN mkdir -p data

EXPOSE 8000

# Use the script to start BOTH the watcher and the server
ENTRYPOINT ["./entrypoint.sh"]