# Cyber Detection Methods Project

A specialized security framework for identifying and classifying malicious activity in the **NPM ecosystem**.  
The system applies machine learning techniques to detect **malware-as-a-package** by comparing known malicious packages against legitimate registry data.

---

## 📖 Project Overview

This project focuses on modern cyber detection methods targeting threats within the **Node Package Manager (NPM)** ecosystem.  
It integrates **feature extraction**, **model training**, and a **user interface** to provide an end-to-end security analysis pipeline for software dependencies.

---

## 🔑 Key Components

- **NPM Malware Detection**  
  Identification of threats such as *typosquatting* and *dependency confusion* in Node packages.

- **Ensemble Machine Learning Models**  
  Multiple ML algorithms combined to achieve high-accuracy threat classification.

- **Hybrid Analysis**  
  Combines package metadata analysis with machine-learning-based behavioral and statistical patterns.

---

## 📊 Data Sources

The project is trained and evaluated using both malicious and legitimate NPM package data.

### Malicious Data

- **DataDog Malicious Software Packages Dataset**  
  A verified collection of real-world malicious NPM packages.

### Legitimate NPM Data

Fetched dynamically from official registry sources:

- **ALL_DOCS_URL**  
  `https://replicate.npmjs.com/_all_docs`  
  Used to retrieve the full list of published NPM packages.

- **REG_URL**  
  `https://registry.npmjs.org`  
  Used to fetch detailed metadata for individual packages.

---

## 📂 Project Structure

```plaintext
Cyber-Detection-Methods-Project/
├── app/
│   ├── api/                     # Backend API services
│   ├── features/                # Feature extraction logic
│   ├── models/                  # Trained ML models
│   ├── train_models/            # Model training scripts
│   ├── ui/                      # User interface components
│   ├── predict_and_evaluate.py  # Inference and evaluation
│   └── test.py                  # Unit and integration tests
├── artifacts/                   # Generated outputs (models, logs, reports)
├── data/                        # Training and testing datasets
├── docker/                      # Docker configuration files
├── docker-compose.yml           # Container orchestration
├── requirements.txt             # Python dependencies
└── run.bat                      # Windows execution script
```

## 🏗️ Project Artifacts

The `artifacts/` directory stores all generated outputs and is essential for reproducibility and evaluation tracking.

### Model Checkpoints
Saved model states and weight files.

### Logs & Metrics
Training logs, evaluation results, and performance metrics.

### Static Assets
Generated diagrams and supporting documentation.

## ⚙️ System Architecture

The system runs as **multiple coordinated services** managed by **Docker Compose**.

### Runtime Components

#### API Service
- Receives requests
- Triggers scans, training, and evaluations
- Dispatches tasks to the worker

#### Worker Service
- Executes long-running and CPU-intensive jobs
- Runs feature extraction, dataset processing, and ML training
- Operates asynchronously in the background

#### Message Broker (Redis)
- Connects the API and worker
- Queues tasks for execution

---

## 🔄 How the Worker Runs (Important)

The worker is **not started manually**.

When running:
```bash
docker-compose up --build
```

---

## 🚀 Getting Started

**Important:**  
For the most up-to-date dataset processing scripts and training logic, switch to the branch:

```bash
Dataset-With-Code
```
### Option 1: Run on Windows (Batch Script)

```bash
./run.bat
```
### Option 2: Run with Docker (Recommended)

```bash
docker-compose up --build
```

## 📦 Requirements

This project is implemented in **Python** and relies on the following core libraries and services.

### Python Dependencies

All required Python packages are listed in `requirements.txt` and are installed automatically when building the Docker image.

```txt
fastapi==0.115.6
uvicorn[standard]==0.32.1
pandas==2.2.3
numpy==2.1.3
joblib==1.4.2
python-multipart==0.0.12
xgboost==2.1.4
scikit-learn==1.6.0
celery==5.4.0
redis==5.2.1
docker==7.1.0
```

## 👥 Authors

- **Shachar Tsrafati**
- **Eitan Beriy**

