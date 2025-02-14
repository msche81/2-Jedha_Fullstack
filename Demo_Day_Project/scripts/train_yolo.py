import os
import zipfile
import numpy as np
import mlflow
from dotenv import load_dotenv
from sklearn.utils.class_weight import compute_class_weight
import collections
from ultralytics import YOLO

# --------------------
# 📌 Chargement des variables d'environnement
# --------------------
load_dotenv("../.env")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
ARTIFACT_ROOT = os.getenv("ARTIFACT_ROOT")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Configuration MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Yolo-Skin-Type-Classification")

# --------------------
# 📌 Définition du modèle YOLO
# --------------------
# Load a model
base_model = YOLO("yolo11n-cls.pt")

# Train the model
data_source = "Oily-Dry-Skin-Types"
results = base_model.train(data=data_source, epochs=100, imgsz=244, device="mps")