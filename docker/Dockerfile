FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy your whole app package
COPY app /app/app

ENV PORT=8000
EXPOSE 8000

# Default model path: inside container at /app/app/models/model.pkl
ENV MODEL_PATH=/app/app/models/model.pkl

CMD ["uvicorn", "app.api.api:app", "--host", "0.0.0.0", "--port", "8000"]
