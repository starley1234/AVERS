# AVERS v0.2 - Dockerfile
# Production-ready image with Web UI + ML capabilities

FROM python:3.11-slim as base

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt pyproject.toml ./
COPY avers ./avers
COPY config.yaml README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn python-multipart && \
    pip install --no-cache-dir -e .

# Optional ML dependencies (uncomment for full ML image)
# RUN pip install --no-cache-dir ultralytics sahi paddleocr transformers torch faiss-cpu

# Create data dirs
RUN mkdir -p /tmp/avers_uploads /tmp/avers_results /tmp/avers_dataset /tmp/avers_rag /tmp/avers_feedback

# Expose Web UI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - Web UI
CMD ["python", "-m", "avers", "web", "--host", "0.0.0.0", "--port", "8000"]

# For CLI usage: docker run avers process input.tif -o output.json
# For Web UI: docker run -p 8000:8000 avers
# For with GPU: docker run --gpus all -p 8000:8000 avers
