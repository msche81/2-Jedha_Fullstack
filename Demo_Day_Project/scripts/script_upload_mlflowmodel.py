import boto3
import os

# 🔹 Paramètres
BUCKET_NAME = "skin-dataset-project-final"
S3_MODEL_PATH = "deployment/2/a1d3f171f7a846c88814497294900a99/artifacts/model/data/model.keras"
LOCAL_MODEL_DIR = "/Users/marie-sophiechenevier/Library/CloudStorage/Dropbox/8-Jedha/GitHub/2-Jedha_Fullstack/Demo_Day_Project/models/MobileNetV2"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model.keras")

# 🔹 Initialiser la connexion S3
s3 = boto3.client("s3")

# 🔹 Créer le dossier local s'il n'existe pas
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# 🔹 Télécharger le modèle
print(f"📥 Téléchargement du modèle depuis S3 : s3://{BUCKET_NAME}/{S3_MODEL_PATH} ...")
s3.download_file(BUCKET_NAME, S3_MODEL_PATH, LOCAL_MODEL_PATH)

print(f"✅ Modèle téléchargé avec succès dans {LOCAL_MODEL_PATH}")