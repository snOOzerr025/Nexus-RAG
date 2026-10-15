# Multi-stage Dockerfile for FastAPI RAG backend
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build dependencies (only in builder)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user --no-warn-script-location -r requirements.txt

# Final runtime image
FROM python:3.11-slim

WORKDIR /app

# Install runtime-only dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy Python packages from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Set environment variables
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Create data directories (will be volumes at runtime)
RUN mkdir -p /app/data/raw /app/data/processed /app/data/chroma_db \
    /app/data/indexes /app/data/registry /app/sample_docs && \
    chown -R appuser:appuser /app

# Copy application source code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser pyproject.toml .
COPY --chown=appuser:appuser sample_docs/ ./sample_docs/

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server with graceful shutdown
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
