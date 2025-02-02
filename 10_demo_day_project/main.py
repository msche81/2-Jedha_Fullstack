from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Dict

# API Metadata
description = """
🚀 **Skin Type Classification API**

This API allows users to:
- 🩺 **Upload an image** for skin type classification (coming soon).
- 🎭 **Get a dummy prediction** (for now).

Future updates will include:
- ✅ Model inference with TensorFlow
- ✅ Image preprocessing pipeline
- ✅ Connection to MLflow for optimized models

**Developed as part of the Jedha Full-Stack Data Science Bootcamp.**
"""

tags_metadata = [
    {"name": "Health Check", "description": "API status check."},
    {"name": "Prediction", "description": "Skin type classification (dummy for now)."},
    {"name": "Upload", "description": "Upload an image for future processing."}
]

# Initialize FastAPI app
app = FastAPI(
    title="Skin Type Classification API",
    description=description,
    version="0.1.0",
    contact={
        "name": "Marie-Sophie Chenevier",
        "email": "mschenevier@gmail.com"
    },
    openapi_tags=tags_metadata
)

# Health Check Endpoint
@app.get("/health", tags=["Health Check"])
def health_check():
    return {"status": "API is running!"}

# Dummy Prediction Endpoint
@app.get("/predict", tags=["Prediction"])
def predict_dummy():
    """
    Returns a dummy skin type classification.
    """
    return {"skin_type": "Oily"}  # Placeholder result

# Upload Image Endpoint (Future Implementation)
@app.post("/upload", tags=["Upload"])
def upload_image(file: UploadFile = File(...)):
    """
    Upload an image for future processing (currently not implemented).
    """
    return {"filename": file.filename, "message": "Image received but not processed yet."}

# ✅ Ensure FastAPI runs on port 7860
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
