import mlflow
from huggingface_hub import HfApi, Repository
import os
import subprocess
import shutil

# Spécifie l'ID de ton modèle Hugging Face (ton espace)
model_name = "MSInTech/mlflow-doctolib"  # Remplace par le nom de ton espace Hugging Face

# Spécifie l'ID de ton modèle MLflow (ID de l'exécution)
run_id = "aca239ac49854907864e4e7ac3613b6b"  # Remplace par ton run ID MLflow

# Télécharge le modèle depuis MLflow
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

# Sauvegarde ton modèle localement avec un nom dynamique
model_local_path = f"/tmp/{model_name}_model"
os.makedirs(model_local_path, exist_ok=True)
mlflow.sklearn.save_model(model, model_local_path)

# Connexion à Hugging Face
hf_api = HfApi()

# Prépare le dépôt Hugging Face
repo = Repository(local_dir=model_local_path, clone_from=model_name)

# Configure l'identité Git (si nécessaire)
subprocess.run(["git", "config", "--global", "user.name", "MSInTech"])
subprocess.run(["git", "config", "--global", "user.email", "mschenevier@gmail.com"])

# Ajoute et commit les fichiers
subprocess.run(["git", "add", "."], cwd=model_local_path)
subprocess.run(["git", "commit", "-m", "Upload model after training"], cwd=model_local_path)

# Pousse le modèle vers Hugging Face
subprocess.run(["git", "push", "origin", "main"], cwd=model_local_path)

# Affiche un message de confirmation
print(f"Model {model_name} uploaded to Hugging Face.")