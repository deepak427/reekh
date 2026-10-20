FROM python:3.11-slim

# Labels for reproducibility and metadata
LABEL maintainer="Agricultural Surveillance System"
LABEL org.opencontainers.image.title="Wildlife Inference Service"
LABEL org.opencontainers.image.description="OpenCV 5 DNN Wildlife Deterrent Backend on AWS Graviton"

WORKDIR /app

# Install minimal runtime OS dependencies for OpenCV 5 headless and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create non-root system user for secure production operation
RUN useradd -u 1000 -m -s /bin/bash appuser && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app

# Copy application code and model weights
COPY --chown=appuser:appuser app/ ./app
COPY --chown=appuser:appuser models/ ./models

USER appuser

EXPOSE 8000

# Container healthcheck ensuring OpenCV DNN engine is responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
