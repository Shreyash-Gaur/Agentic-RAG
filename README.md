# Agentic RAG

A Retrieval-Augmented Generation (RAG) system with agentic workflows, built with FastAPI, Ollama, and vector stores (FAISS/ChromaDB).

## Features

- **Dual Vector Stores**: Support for both FAISS and ChromaDB
- **Agentic Workflows**: Researcher and Writer agents for intelligent information processing
- **Conversation Memory**: Context-aware conversations with memory management
- **Iterative Refinement**: Multi-iteration query refinement
- **PDF Ingestion**: Document processing and chunking
- **Interactive UI**: Chainlit-based frontend for testing
- **Evaluation Tools**: Scripts for retrieval and RAG evaluation

## Project Structure

```
project-agentic-rag/
├── backend/              # Backend API and core logic
│   ├── main.py          # FastAPI application entry point
│   ├── core/            # Core utilities (config, logging, exceptions)
│   ├── agents/          # Agent implementations
│   ├── tools/           # LLM clients, retrievers, embedders
│   ├── services/        # RAG and memory services
│   ├── workflows/       # Agentic workflow orchestration
│   ├── models/          # Pydantic request/response models
│   └── scripts/         # Utility scripts
├── frontend/            # Chainlit UI
├── experiments/         # Evaluation scripts
└── data/               # Document datasets (gitignored)
```

## Setup

### Prerequisites

- Python 3.11+
- Ollama installed and running (see [Ollama](https://ollama.ai/))
- Required models pulled in Ollama:
  - LLM model (e.g., `llama2`): `ollama pull llama2`
  - Embedding model (e.g., `nomic-embed-text`): `ollama pull nomic-embed-text`

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Agentic-rag
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

4. Create `.env` file from `.env.example`:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### 1. Ingest Documents

Ingest PDF documents into the vector store:

```bash
python backend/scripts/ingest_data.py path/to/document.pdf --vector-store faiss
```

### 2. Start the Backend API

```bash
cd backend
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### 3. Use the Interactive UI (Chainlit)

```bash
cd frontend
chainlit run chainlit_app.py
```

### 4. Run Experiments

Evaluate retrieval performance:
```bash
python experiments/eval_retrieval.py --queries-file data/eval_queries.json
```

Evaluate RAG performance:
```bash
python experiments/eval_rag.py --eval-file data/eval_data.json
```

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health check
- `POST /query` - Query the RAG system (to be implemented)
- `POST /ingest` - Ingest documents (to be implemented)

## Configuration

Edit `.env` file to configure:
- Ollama settings (base URL, models)
- Vector store type and paths
- RAG parameters (chunk size, top-k, etc.)
- Agent settings (temperature, max iterations)

## Development

### Running Tests

```bash
# Add test files and run
pytest backend/tests/
```

### Development Notebook

Use `backend/notebooks/dev_experiments.ipynb` for interactive development and testing.

## Docker

Build and run with Docker:

```bash
docker build -f docker/backend.Dockerfile -t agentic-rag-backend .
docker run -p 8000:8000 agentic-rag-backend
```

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

