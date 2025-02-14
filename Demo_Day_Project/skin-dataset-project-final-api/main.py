import os
import boto3
import mlflow
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from PIL import Image
import numpy as np
import uvicorn
from tensorflow.keras.metrics import Metric
import tensorflow.keras.backend as K
from predict import predict_skin_type

# ✅ Définition de la métrique F1Score pour éviter l'erreur de chargement
class F1Score(Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass  # Ne pas modifier ici, juste pour la compatibilité du modèle

    def result(self):
        return K.variable(0.0)

# ✅ Enregistrement de la métrique pour TensorFlow/Keras
tf.keras.utils.get_custom_objects()["F1Score"] = F1Score

# ✅ API Metadata
description = """
🚀 **Skin Type Classification API**

This API allows users to:
- 🩺 **Upload an image** for skin type classification.
- 🎭 **Get a skin type prediction using a trained MLflow model**.
- ✅ Improve model inference with TensorFlow
- ✅ Enhance image preprocessing pipeline
- ✅ Hyperparameter tuning via MLflow

**Developed as part of the Jedha Full-Stack Data Science Bootcamp.**
"""

tags_metadata = [
    {"name": "Health Check", "description": "API status check."},
    {"name": "Prediction", "description": "Skin type classification using MLflow & TensorFlow."},
    {"name": "Upload", "description": "Upload an image for prediction."}
]

# ✅ Initialize FastAPI app
app = FastAPI(
    title="Skin Type Classification API",
    description=description,
    version="1.0.0",
    contact={"name": "Marie-Sophie Chenevier", "email": "mschenevier@gmail.com"},
    openapi_tags=tags_metadata
)

# ✅ Configuration de MLflow & S3
S3_BUCKET = "skin-dataset-project-final"
S3_MODEL_PATH = "deployment/2/a1d3f171f7a846c88814497294900a99/artifacts/model/data/model.keras"
LOCAL_MODEL_PATH = "./model.keras"

# 🔥 Télécharger le modèle depuis S3
s3_client = boto3.client("s3")

def download_model_from_s3():
    """
    Télécharge le modèle MLflow depuis S3 et le sauvegarde localement.
    """
    try:
        print(f"📥 Téléchargement du modèle depuis S3: s3://{S3_BUCKET}/{S3_MODEL_PATH}")
        s3_client.download_file(S3_BUCKET, S3_MODEL_PATH, LOCAL_MODEL_PATH)
        print("✅ Modèle téléchargé avec succès !")
        return True
    except Exception as e:
        print(f"❌ Erreur de téléchargement du modèle depuis S3: {e}")
        return False

# 🔥 Charger le modèle TensorFlow uniquement si le téléchargement a réussi
model = None
if download_model_from_s3():
    try:
        model = tf.keras.models.load_model(LOCAL_MODEL_PATH, custom_objects={"F1Score": F1Score})
        print("✅ Modèle chargé avec succès depuis le fichier local !")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")

# ✅ Health Check Endpoint
@app.get("/health", tags=["Health Check"])
def health_check():
    return {
        "status": "API is running!",
        "model_loaded": model is not None
    }

# ✅ Endpoint pour la prédiction avec upload S3
@app.post("/predict", tags=["Prediction"])
async def predict_and_store(file: UploadFile = File(...)):
    """
    Upload an image, predict skin type and store the image in S3.
    """
    image_bytes = await file.read()  # 📸 Lire l'image

    # 🧠 Faire la prédiction avec le modèle externe
    predicted_label = predict_skin_type(image_bytes)

    # 📤 Envoyer l'image vers S3 après la prédiction
    try:
        s3_filename = f"images/{file.filename}"
        s3_client.upload_fileobj(BytesIO(image_bytes), S3_BUCKET, s3_filename)
        print(f"✅ Image {file.filename} envoyée sur S3 !")
    except Exception as e:
        print(f"❌ Erreur lors de l'upload sur S3 : {e}")

    return {"filename": file.filename, "skin_type": predicted_label}

# ✅ Image Upload
@app.post("/upload-to-s3", tags=["Upload"])
async def upload_image_to_s3(file: UploadFile = File(...)):
    """
    Upload an image and save it to S3 in the 'images/' folder.
    """
    try:
        s3_filename = f"images/{file.filename}"
        file.file.seek(0)
        s3_client.upload_fileobj(file.file, S3_BUCKET, s3_filename)
        
        s3_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_filename}"
        
        return {"filename": file.filename, "s3_url": s3_url, "message": "Upload vers S3 réussi!"}

    except Exception as e:
        return {"error": f"Erreur lors de l'upload sur S3 : {str(e)}"}

# ✅ Lancer l'API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))