FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies (Celery/Redis included)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy only the app logic
COPY app /app/app

# 4. Create empty directories for volume mounting
RUN mkdir -p /app/data /app/artifacts /shared/results

# 5. Environment Variables
ENV RESULTS_DIR=/shared/results
ENV MODEL_PATH=/app/artifacts/xgboost_model.pkl

EXPOSE 8000