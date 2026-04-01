# MLOps-Production-Pipeline-E2E
This repository demonstrates a complete, production-grade Machine Learning lifecycle. Inspired by the 'Made With ML' framework, this project goes beyond model training to implement robust data engineering, automated testing, and CI/CD deployment.

# End-to-End MLOps Production Pipeline
[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue)](#)
[![Model Tracking](https://img.shields.io/badge/Tracking-MLflow-orange)](#)
[![Data Validation](https://img.shields.io/badge/Data%20Validation-Great%20Expectations-green)](#)
[![Monitoring](https://img.shields.io/badge/Monitoring-Evidently.ai-red)](#)

## 🎯 Project Overview 
The goal of this project was to build a scalable ML system that manages the entire model lifecycle—from training and versioning to serving and real-world performance monitoring of data drift and automates retraining.

**Key Achievements:**
* **Automated Model Promotion:** Implemented logic to programmatically select the best-performing model from MLflow runs and promote it to the distribution layer.
* **Resilient Serving:** Developed a FastAPI wrapper optimized for Docker that handles complex MLflow artifact paths and `PosixPath` conversions for cross-platform stability.
* **Modern Monitoring:** Integrated **Evidently AI (v0.7.x)** to detect data drift between training baselines and live production data.


## 🏗️ Architecture
* **Experiment Tracking:** MLflow (Local FileStore backend)
* **Data Quality:** Great Expectations for input validation
* **API Layer:** FastAPI with Uvicorn
* **Containerization:** Docker (Python 3.10-slim)
* **Monitoring:** Evidently.ai Data Drift Reports
* **Orchestration:** Ray (Distributed training & serving) - *Planned*

Training (MLflow) → Promotion → Serving (FastAPI/Docker) → Monitoring (Evidently).

## 🚀 Live Demo
#TODO update with demo when available 
(link to my Hugging Face Space or Render API endpoint or somehting similiar)

## 🚀 Quickstart: Reproduce with Docker
The most reliable way to run this project is via Docker, which ensures environment parity and bypasses local dependency conflicts.

1. **Generate the Model Artifacts**:
   ```bash
    python src/train.py
    python src/promote_model.py   
   ```

2. **Build the Image**:
   ```bash
    docker build -t iris-predictor:v1 -f Dockerfile.serve .
   ```

3. **Run the Container**:
   ```bash
    docker run -p 8001:8000 iris-predictor:v1
   ```

4. **Test the Endpoint**:
   ```bash
    curl -X 'POST' '[http://127.0.0.1:8001/predict](http://127.0.0.1:8001/predict)' \
         -H 'Content-Type: application/json' \
         -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
   ```

## 📊 Monitoring & Drift Detection

The pipeline includes a dedicated monitoring suite (```src/monitor.py```) utilizing Evidently AI to compare production data against training baselines.

* **Drift Report**: Generates an interactive HTML dashboard in docs/reports/data_drift_report.html.

* **Simulated Drift**: The current suite verifies system health by simulating a 50% increase in petal length to ensure drift is detected and visualized.

## 🛠️ Performance Metrics
#TODO update this template later for P&M
| Metric | Baseline (v1.0) | Status |
| :--- | :--- | :--- |
| Accuracy | 1.00 | 1.00 | ✅ Promoted
| Inference Latency | < 20 ms> | ✅ Optimized |
| Data Drift | 0% (Baseline) | ⚠️ Detected (Simulated) |

> **Note on Model Drift:** #TODO update this later on --> I utilize a daily data stream from [API Name] to monitor for concept drift. View the latest monitoring report [here](#).

## 📖 Roadmap
- [x] Week 1: Baseline Pipeline & Data Validation
- [x] Week 2: Testing & Containerization
- [x] Week 3: CI/CD & Cloud Deployment
- [x] Week 4: Drift Detection & Performance Tuning



## 📖 TODO

- Bridge Conda and Pip - Developed using Conda for environment stability; compatible with Pip for lightweight container deployment. - before deploying to containers export to pip
-Integrate Ray for distributed orchestration.

## 🖼️ Sample Screenshots

### Local Run Experiment Results

![MLFlow Dashboard Local Run](docs/images/MlflowDashLocalRun.png)
