FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy app code and ensure directories exist
COPY app /app/app
RUN mkdir -p /app/data /app/artifacts

# 4. Copy the specific training data
COPY data/npm_train.csv /app/data/npm_train.csv

# 5. Set Environment Variables
ENV PORT=8000
ENV MODEL_PATH=/app/artifacts/xgboost_model.pkl

EXPOSE 8000

# 6. SEQUENTIAL STARTUP: Train model, Print Link, THEN start the API
# We use 'echo' to print the clickable localhost link for Windows users.
CMD python -m app.train_models.train && \
    echo "" && \
    echo "====================================================" && \
    echo "  STARTING DOCKER!                                " && \
    echo "  API Ready at: http://localhost:8000                " && \
    echo "====================================================" && \
    echo "" && \
    uvicorn app.api.api:app --host 0.0.0.0 --port 8000