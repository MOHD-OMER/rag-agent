# ============================================================
# Multi-Document RAG Chatbot — Docker Image
# ============================================================
FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libmagic1 \
    libpoppler-cpp-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ──
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model to avoid cold-start downloads
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Application code ──
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY .env.example ./.env.example

# Create data directories
RUN mkdir -p /app/chroma_db /app/uploads

# ── Health check ──
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000 8501 7860

# Default: start FastAPI backend
# Override CMD for other services (see docker-compose.yml)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
