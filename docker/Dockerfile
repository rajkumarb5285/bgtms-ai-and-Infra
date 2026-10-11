# Multi-stage production Dockerfile for BGTMS AI & Serving Engine
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies required for OpenCV, image decoders, and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with pip wheel caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source tree and models
COPY configs/ ./configs/
COPY bgtms/ ./bgtms/
COPY server/ ./server/
COPY dashboard/ ./dashboard/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/bgtms/models

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --retries=3 --start-period=10s \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
