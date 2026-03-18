FROM python:3.11-slim
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/

# Copy frontend code
COPY frontend/ ./frontend/

EXPOSE 8001

CMD ["chainlit", "run", "frontend/chainlit_app.py", "--host", "0.0.0.0", "--port", "8001"]
