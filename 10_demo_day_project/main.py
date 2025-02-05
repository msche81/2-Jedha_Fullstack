import os
import boto3
import mlflow
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from PIL import Image
import numpy as np
import uvicorn

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
S3_BUCKET = "skin-dataset-project"
S3_MODEL_PATH = "deployment/mobilenetv2_skin_classifier/latest_model/model.keras"  # Utiliser 'latest_model' en attendant la structure exacte
LOCAL_MODEL_PATH = "/tmp/model.keras"  # 📌 Fichier temporaire local

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
    except Exception as e:
        print(f"❌ Erreur de téléchargement du modèle depuis S3: {e}")

# 🔥 Charger le modèle TensorFlow après téléchargement
model = None
try:
    download_model_from_s3()
    model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
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

# ✅ Endpoint pour la prédiction
@app.post("/predict", tags=["Prediction"])
async def predict_skin_type(file: UploadFile = File(...)):
    """
    Upload an image and get a skin type prediction using MLflow.
    """
    if model is None:
        return {"error": "Modèle non chargé, impossible de prédire."}

    # 📸 Lire et prétraiter l'image
    image = Image.open(BytesIO(await file.read()))
    image = image.resize((224, 224))  # 📏 Adapter à la taille du modèle
    image_array = np.array(image) / 255.0  # 📌 Normalisation
    image_array = np.expand_dims(image_array, axis=0)  # 📌 Adapter la dimension pour le modèle

    # 🧠 Faire la prédiction avec le modèle TensorFlow
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    skin_types = {0: "Oily", 1: "Dry", 2: "Normal"}
    predicted_label = skin_types.get(predicted_class, "Unknown")

    return {"filename": file.filename, "skin_type": predicted_label}

# ✅ Lancer l'API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))