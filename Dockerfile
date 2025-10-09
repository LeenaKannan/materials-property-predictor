# Multi-stage Dockerfile for Materials Property Predictor

# Base stage with common dependencies
FROM python:3.9-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000 8501

# Default command
CMD ["python", "backend/api/main.py"]

# Production stage
FROM base as production

# Copy only necessary files
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY train_model.py .
COPY .env.example .env

# Create models directory
RUN mkdir -p models

# Run as non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start API server
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]