from dotenv import load_dotenv
import os
import boto3
import json

# Charger les variables d’environnement depuis secrets.sh
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)

BUCKET_NAME = "skin-dataset-project-final"
S3_FOLDER = "dataset/"
LOCAL_FILE = os.path.join(os.path.dirname(__file__), "..", "src", "oily-dry-and-normal-skin-types-dataset.zip")

# Vérifier que les variables sont bien chargées
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region_name = os.getenv("AWS_DEFAULT_REGION")

if not aws_access_key or not aws_secret_key or not region_name:
    raise ValueError("❌ Erreur: Les variables AWS ne sont pas chargées correctement depuis secrets.sh")

# Initialiser le client S3 avec les credentials chargés
s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region_name
)

# Désactiver les restrictions publiques immédiatement (évite les erreurs si déjà désactivé)
try:
    s3.delete_public_access_block(Bucket=BUCKET_NAME)
    print("✅ BlockPublicPolicy désactivé !")
except Exception as e:
    print(f"⚠️ Impossible de supprimer le blocage d'accès public : {e}")

# Définir une politique bucket publique en lecture
bucket_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": "*",
            "Action": ["s3:ListBucket"],
            "Resource": f"arn:aws:s3:::{BUCKET_NAME}"
        },
        {
            "Effect": "Allow",
            "Principal": "*",
            "Action": ["s3:GetObject"],
            "Resource": f"arn:aws:s3:::{BUCKET_NAME}/{S3_FOLDER}*"
        }
    ]
}

# Appliquer la politique (gérer les erreurs si elle est déjà en place)
try:
    s3.put_bucket_policy(Bucket=BUCKET_NAME, Policy=json.dumps(bucket_policy))
    print("✅ Politique appliquée avec succès !")
except Exception as e:
    print(f"⚠️ Impossible d'appliquer la politique du bucket : {e}")

# Fonction pour uploader un fichier spécifique
def upload_file(file_path, s3_folder):
    """Upload un fichier unique dans S3"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Erreur: Le fichier {file_path} n'existe pas.")

    file_name = os.path.basename(file_path)
    s3_key = f"{s3_folder}{file_name}"  # Stockage dans dataset/
    
    try:
        s3.upload_file(file_path, BUCKET_NAME, s3_key)
        print(f"✅ Fichier uploadé : {file_path} → s3://{BUCKET_NAME}/{s3_key}")
    except Exception as e:
        print(f"❌ Erreur lors de l'upload du fichier {file_name}: {e}")

# Upload du fichier ZIP
upload_file(LOCAL_FILE, S3_FOLDER)

print("🚀 Upload terminé avec succès !")