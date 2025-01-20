from huggingface_hub import create_repo

# Créer un Space sur Hugging Face
create_repo(repo_id="mlflow-doctolib", repo_type="space", space_sdk="docker")

print("✅ Space 'mlflow-doctolib' créé avec succès sur Hugging Face!")
