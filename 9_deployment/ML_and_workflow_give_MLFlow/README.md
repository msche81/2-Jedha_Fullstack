---
title: "MLflow Deployment"
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# MLflow Deployment on Hugging Face 🚀

This Space deploys an MLflow tracking server using Docker.

## 🔧 Configuration

- **Port:** 7860 (default Hugging Face port)
- **Backend Storage:** `sqlite:///mlflow.db`
- **Artifact Store:** `s3://bucketdeployment/mlflow-artifacts/`

## 🚀 How to Use

1. Clone the repository:
git clone https://huggingface.co/spaces/MSInTech/MLFlow_test.git

2. Run MLflow locally (optional):
docker run -p 7860:7860 deployment-mlflow-server

3. View the MLflow UI:
http://localhost:7860

## 🛠 Deployment Status

Check logs in Hugging Face to see if the build is successful.# MLflow Deployment on Hugging Face 🚀

This Space deploys an MLflow tracking server using Docker.

## 🔧 Configuration

- **Port:** 7860 (default Hugging Face port)
- **Backend Storage:** `sqlite:///mlflow.db`
- **Artifact Store:** `s3://bucketdeployment/mlflow-artifacts/`

## 🚀 How to Use

1. Clone the repository:
git clone https://huggingface.co/spaces/MSInTech/MLFlow_test.git

2. Run MLflow locally (optional):
docker run -p 7860:7860 deployment-mlflow-server

3. View the MLflow UI:
http://localhost:7860

## 🛠 Deployment Status

Check logs in Hugging Face to see if the build is successful.
