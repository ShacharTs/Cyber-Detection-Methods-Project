# docker/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy ONLY the necessary application directories
# We copy the folder 'app' into a folder named 'app' in the container
COPY app ./app
# Note: We do NOT copy 'docker/' or '.venv/'

# 4. Create necessary directories for runtime
RUN mkdir -p /app/data /app/artifacts /shared/results

# 5. Environment Variables
ENV PYTHONPATH=/app
ENV RESULTS_DIR=/shared/results
ENV ARTIFACT_DIR=/app/artifacts

EXPOSE 8000