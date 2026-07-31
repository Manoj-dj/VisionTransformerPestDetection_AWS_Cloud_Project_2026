# VisionTransformerPestDetection_AWS_Cloud_Project_2026

A Vision Transformer–based cloud platform on AWS for **early insect pest detection** in crops using leaf and canopy images captured via mobile devices and UAVs. The goal is to provide farmers and agronomists with a scalable, explainable, and low-latency pest detection service backed by modern MLOps on AWS.[web:44]

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Project Objectives](#project-objectives)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Early detection of insect pests is critical to reducing crop losses and avoiding excessive pesticide usage. Traditional manual scouting is time‑consuming, subjective, and difficult to scale across large farms.

This project implements a **Vision Transformer (ViT)–based model** for insect pest recognition and integrates it into an **AWS cloud platform** that provides:

- REST APIs for inference
- Secure data storage and user management
- Monitoring, logging, and alerting for pest risk

The repository contains the Phase‑I work for the BITE412L Cloud Computing course, including literature survey, research gap analysis, architecture design, and initial implementation plan.

---

## Features

- **ViT-based pest detection model** trained on the IP102 insect pest dataset and field-like images.
- **Early-stage detection focus** (small pests, subtle leaf damage, canopy anomalies).
- **AWS-native MLOps stack** using Amazon SageMaker, S3, API Gateway, Lambda, Cognito, CloudWatch, and SNS.
- **Explainable AI** (Grad‑CAM/heatmaps) for selected pest classes to highlight image regions influencing predictions.
- **Multi-tenant support** for different farms/teams with role-based access control.
- **GitHub workflow** with feature branches, PR reviews, and tagged releases (e.g., `v1.0-Phase1`).

---

## Architecture

### High-Level AWS Cloud Architecture

Core AWS components:

- **Amazon S3** – Stores raw images, labeled datasets, model artifacts, and explanation outputs.
- **Amazon SageMaker** – Trains and hosts ViT/hybrid models for pest classification and detection.
- **Amazon API Gateway** – Exposes REST endpoints (`/predict`, `/upload`) for web/mobile clients.
- **AWS Lambda** – Orchestrates requests, preprocessing, and communication with SageMaker endpoints.
- **Amazon Cognito** – Manages authentication and user pools (students, farmers, instructor).
- **Amazon DynamoDB / Amazon RDS** – Persists metadata: images, labels, predictions, and farm configuration.
- **Amazon CloudWatch** – Collects logs and metrics (latency, error rate, alert counts).
- **Amazon SNS** – Sends pest alerts and system notifications.
- **AWS IoT Core** (optional) – Ingests UAV or edge camera streams for real-time canopy monitoring.

### System Workflow

1. **Image Capture & Upload**
   - Farmer/mobile app or UAV captures crop images.
   - Images are uploaded to S3; metadata stored in DynamoDB/RDS.

2. **Preprocessing & Labeling**
   - Lambda/SageMaker Processing jobs resize and normalize images.
   - Experts/students annotate pest species and infestation severity.

3. **Model Training**
   - SageMaker training jobs load IP102 + field data from S3.
   - Vision Transformer/hybrid models are trained and evaluated.
   - Best models are registered in SageMaker Model Registry.

4. **Deployment & Inference**
   - Real-time endpoints are deployed via SageMaker.
   - API Gateway + Lambda route inference requests to endpoints.
   - Responses include pest class, confidence score, severity level, and optional explanation heatmap.

5. **Monitoring & Alerts**
   - CloudWatch tracks performance metrics and logs.
   - SNS sends alerts when pest risk exceeds thresholds.

6. **Continuous Improvement**
   - Misclassified or “hard” examples are collected for retraining.
   - New model versions are deployed and tagged in GitHub (e.g., `v1.1-Phase2`).

---

## Datasets

### IP102 – Insect Pest Benchmark

- **Name:** IP102 – A Large-Scale Benchmark Dataset for Insect Pest Recognition
- **Source:** CVPR 2019, GitHub repository `xpwu95/IP102`
- **Size:** >75,000 images, 102 pest categories
- **Annotations:** ~19,000 images with bounding boxes
- **Usage in this project:**
  - Main training and evaluation dataset for pest classification/detection.
  - Used to benchmark ViT-based models and evaluate early detection performance.

### Supplementary Datasets

- **PlantVillage:** For auxiliary pretraining (general plant visual features).
- **Field-like datasets (e.g., PlantDoc, PlantDiseaseNet):** For testing robustness under complex backgrounds.

---

## Getting Started

> Note: These steps are indicative for Phase‑I. Adjust paths and commands as the implementation evolves.

### Prerequisites

- Git
- Python 3.10+ (for model training and backend scripts)
- Node.js or another frontend stack (optional, for dashboards)
- AWS account with appropriate permissions for SageMaker, S3, Lambda, API Gateway, Cognito, CloudWatch

### Clone the Repository

```bash
git clone https://github.com/<your-username>/VisionTransformerPestDetection_AWS_Cloud_Project_2026.git
cd VisionTransformerPestDetection_AWS_Cloud_Project_2026
```

### Branch Workflow (Team of Two)

```bash
# Student A
git checkout -b feature/studentA

# Student B
git checkout -b feature/studentB
```

### Basic Commands

- **Install Python dependencies (example):**
  ```bash
  pip install -r src/ai_model/requirements.txt
  ```

- **Run local training script (example):**
  ```bash
  python src/ai_model/train_vit_ip102.py
  ```

- **Run local API backend (example):**
  ```bash
  uvicorn src/backend.main:app --reload
  ```

(You can update these commands as your actual scripts are implemented.)

---

## Repository Structure

```text
VisionTransformerPestDetection_AWS_Cloud_Project_2026/
├── README.md
├── docs/
│   ├── phase1_report/
│   └── architecture_diagrams/
├── literature_survey/
│   ├── studentA_papers_1_8.md
│   └── studentB_papers_9_15.md
├── architecture/
│   ├── aws_cloud_architecture.png
│   └── system_workflow.png
├── dataset/
│   ├── ip102_sample/
│   └── preprocessing_scripts/
├── src/
│   ├── frontend/        # Web/mobile UI
│   ├── backend/         # API service (FastAPI/Node.js)
│   ├── ai_model/        # ViT model code and training notebooks
│   └── aws/             # CloudFormation/Terraform, Lambda handlers
├── results/
│   ├── evaluation_metrics/
│   └── explanation_examples/
├── presentation/
│   └── phase1_slides/
└── references/
    └── bibtex_and_citations/
```

You can create these directories gradually as you implement each part.

---

## Project Objectives

Phase‑I objectives (aligned with course guidelines):

1. Perform a **literature survey of 15 papers** and document methods, advantages, limitations, and research gaps for pest/disease detection.
2. Define **4–6 measurable project objectives** (accuracy, latency, early detection, multi-tenant support, explainability, monitoring).
3. Design **two architecture diagrams**:
   - AWS cloud architecture (services and interactions).
   - Complete system workflow (data → model → alerts).
4. Document **dataset details** for IP102 and supplementary sets.
5. Plan an initial **GitHub workflow** with feature branches, PRs, and tagged releases.

---

## Contributing

- Use feature branches (`feature/studentA`, `feature/studentB`) for development.
- Open Pull Requests into `develop` for code review.
- After integration and testing, merge `develop` into `main` and tag releases (e.g., `v1.0-Phase1`).
- Write clear commit messages (e.g., `"Added ViT training script for IP102"`).

For external contributors (if any), please open an issue first to discuss proposed changes.

---

## License

Add a suitable license (e.g., MIT, Apache 2.0) in a separate `LICENSE` file and reference it here once decided.

```text
This project is currently for academic use (BITE412L Cloud Computing course).
A formal open-source license will be added if the project is published publicly.
```