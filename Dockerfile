# ── Stage 1: Build React frontend ────────────────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend with built frontend ───────────────────────────────
FROM python:3.11-slim AS backend

WORKDIR /app

# System deps for PyMuPDF (PDF parsing) and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    libfreetype6-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
# Install uv for fast dependency resolution
RUN pip install uv --quiet && \
    uv pip install --system \
    fastapi \
    uvicorn[standard] \
    python-dotenv \
    pydantic-settings \
    pydantic \
    pymupdf \
    langchain \
    langchain-core \
    langchain-openai \
    langchain-groq \
    langchain-google-genai \
    tavily-python \
    sentence-transformers \
    torch --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY backend/ ./backend/
COPY data/ ./data/

# Copy built frontend static files into backend static directory
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create data directories if they don't exist
RUN mkdir -p /app/data/pdf_files

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

# Run the FastAPI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
