# Build stage for python dependencies
FROM python:3.11-slim as builder

# Set env variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for compiling tree-sitter or other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for super fast package installations
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependencies definitions
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies using uv to a virtual environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install --no-cache-dir .

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project files
COPY src/ /app/src/
COPY ingest.py /app/
COPY assistant.py /app/
COPY .env.example /app/

# Expose port for FastAPI
EXPOSE 8000

# Set environment variable for persistence
ENV CHROMA_PERSIST_DIR=/app/chroma_db

# Run FastAPI app by default
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
