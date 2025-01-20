from huggingface_hub import HfApi, Repository
import mlflow
from mlflow.tracking import MlflowClient
import os
import subprocess
import shutil

# Remplace par ton nom d'utilisateur Hugging Face
hf_username = "MSInTech"  # Ton nom d'utilisateur Hugging Face
hf_token = os.getenv("HF_TOKEN")  # Ton token Hugging Face (assure-toi qu'il est stocké dans l'environnement)

# Connexion à Hugging Face
hf_api = HfApi()

# Connexion à MLflow
mlflow.set_tracking_uri("http://host.docker.internal:7860")
client = MlflowClient()

# Nom de l'espace Hugging Face écrit en dur
latest_space_name = "MSInTech/mlflow-doctolib"  # Remplace par ton espace Hugging Face

# Récupérer les informations sur l'expérience la plus récente dans MLflow
def get_latest_mlflow_experiment():
    try:
        experiments = client.list_experiments()
    except AttributeError:
        experiments = client.search_experiments(order_by=["creation_time desc"])
    
    latest_experiment = max(experiments, key=lambda x: x.creation_time)
    return latest_experiment

# Récupérer l'expérience la plus récente
latest_experiment = get_latest_mlflow_experiment()

print(f"Most recent MLflow experiment: {latest_experiment.name}")
print(f"Most recent Hugging Face space: {latest_space_name}")

# Trouver le dernier run de l'expérience la plus récente
runs = client.search_runs(
    experiment_ids=[latest_experiment.experiment_id],  # Passer correctement l'ID de l'expérience
    order_by=["start_time desc"]
)

# Vérifier que nous avons bien des runs disponibles
if not runs:
    print("No runs found for this experiment!")
else:
    latest_run_id = runs[0].info.run_id  # Dernier run

    # Télécharger le modèle depuis MLflow
    model_uri = f"runs:/{latest_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    # Sauvegarder ton modèle localement avec un nom dynamique basé sur l'ID du run ou l'expérience
    model_path = f"{latest_experiment.name}_{latest_run_id}_model"
    model_local_path = f"/tmp/{model_path}"  # Temp location for model
    mlflow.sklearn.save_model(model, model_local_path)

    # Créer le fichier .netrc pour l'authentification automatique
    netrc_path = "/home/user/.netrc"  # ou "/tmp/.netrc" si tu es en mode temp
    with open(netrc_path, "w") as f:
        f.write(f"machine huggingface.co\nlogin {hf_username}\npassword {hf_token}\n")

    # Configurer l'identité Git dans Docker (évite l'erreur liée à l'identité inconnue)
    subprocess.run(["git", "config", "--global", "user.name", "MSInTech"])
    subprocess.run(["git", "config", "--global", "user.email", "mschenevier@gmail.com"])

    # Créer un répertoire temporaire pour cloner Hugging Face (sans interférer avec GitHub)
    temp_clone_dir = f"/tmp/{latest_space_name}_clone"
    os.makedirs(temp_clone_dir, exist_ok=True)

    # Initialiser le dépôt Git dans le répertoire temporaire
    subprocess.run(["git", "init"], cwd=temp_clone_dir)

    # Utiliser le token Hugging Face pour l'authentification via .git-credentials
    credentials_path = "/home/user/.git-credentials"  # Le fichier doit être dans /home/user/ ou /tmp/
    with open(credentials_path, 'w') as credentials_file:
        credentials_file.write(f"https://{hf_token}@huggingface.co\n")

    subprocess.run(["git", "config", "--global", "credential.helper", "store"], cwd=temp_clone_dir)

    # Ajouter le remote Hugging Face
    subprocess.run(["git", "remote", "add", "origin", f"https://huggingface.co/{latest_space_name}"], cwd=temp_clone_dir)

    # Vérifier si la branche 'main' existe déjà, sinon créer
    branch_check = subprocess.run(
        ["git", "branch", "--list", "main"], cwd=temp_clone_dir, capture_output=True, text=True
    )

    # Si la branche 'main' n'existe pas, la créer
    if "main" not in branch_check.stdout:
        subprocess.run(["git", "checkout", "-b", "main"], cwd=temp_clone_dir)
    else:
        subprocess.run(["git", "checkout", "main"], cwd=temp_clone_dir)  # Assurer que nous sommes sur la branche 'main'

    # Copier le modèle dans le répertoire cloné
    subprocess.run(["cp", "-r", model_local_path, temp_clone_dir], cwd=temp_clone_dir)

    # Ajouter, commettre et pousser les fichiers vers Hugging Face
    subprocess.run(["git", "pull", "origin", "main"], cwd=temp_clone_dir)
    subprocess.run(["git", "add", "."], cwd=temp_clone_dir)
    subprocess.run(["git", "commit", "-m", "Upload model after training"], cwd=temp_clone_dir)
    subprocess.run(["git", "push", "origin", "main"], cwd=temp_clone_dir)

    print(f"Model uploaded to Hugging Face Space: {latest_space_name}")

    # Optionnel: Supprimer le dossier temporaire après la pousse
    shutil.rmtree(temp_clone_dir)