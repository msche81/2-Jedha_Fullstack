import mlflow
from mlflow.tracking import MlflowClient

# 🔗 Connexion au tracking server MLflow distant
MLFLOW_TRACKING_URI = "https://msintech-skin-dataset-project-final.hf.space/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

# 📌 ID de l'expérience à restaurer
experiment_to_restore = 3

# 📌 IDs des expériences à supprimer définitivement
experiments_to_delete = [4, 5]

# 🔄 Restauration de l'expérience ID 3
try:
    client.restore_experiment(experiment_to_restore)
    print(f"✅ Expérience avec ID {experiment_to_restore} restaurée avec succès.")
except Exception as e:
    print(f"❌ Erreur lors de la restauration de l'expérience {experiment_to_restore} : {e}")

# 🚮 Suppression définitive des expériences IDs 4 et 5
for exp_id in experiments_to_delete:
    try:
        client.delete_experiment(exp_id)
        print(f"✅ Expérience avec ID {exp_id} supprimée définitivement.")
    except Exception as e:
        print(f"❌ Erreur lors de la suppression de l'expérience {exp_id} : {e}")

print("🚀 Restauration et nettoyage terminés !")