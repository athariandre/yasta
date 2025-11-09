# Tron PPO Training Dockerfile (CPU-only)
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=42

# Default command: run training script
CMD ["bash", "scripts/run_training.sh"]

