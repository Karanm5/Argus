# ── Stage 1: Build dependencies ──────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime image ────────────────────────────────────────────
FROM python:3.11-slim

# Security: run as non-root user
RUN useradd --create-home --shell /bin/bash argus
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY argus/     ./argus/
COPY frontend/  ./frontend/
COPY setup.cfg  .

# Data directory for ChromaDB persistence
RUN mkdir -p /app/data/chroma && chown -R argus:argus /app

USER argus

# Default: run the FastAPI backend
# Override CMD in docker-compose.yml to run Streamlit instead
EXPOSE 8000
CMD ["uvicorn", "argus.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
