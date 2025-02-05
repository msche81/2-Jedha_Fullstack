#!/bin/bash

echo "===== Début du script de débogage ====="

# Vérifier si MLflow est bien configuré
echo "🌍 Vérification de l'accès MLflow"
echo "MLflow Tracking URI: $MLFLOW_TRACKING_URI"
echo "MLflow Artifact Root: $MLFLOW_ARTIFACT_URI"

# Vérification de la connexion à AWS S3
echo "🌍 Vérification de l'accès AWS S3"
aws s3 ls s3://skin-dataset-project/deployment/

# Afficher les fichiers présents dans l’environnement
echo "📂 Contenu du dossier de travail"
ls -lah

echo "===== Fin du script de débogage ====="

# Lancer l'application FastAPI avec Uvicorn
uvicorn main:app --host 0.0.0.0 --port 7860