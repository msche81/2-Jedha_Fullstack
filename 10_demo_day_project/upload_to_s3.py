from dotenv import load_dotenv
import os
import boto3

# Charger toutes les variables d’environnement
load_dotenv(dotenv_path="credentials.env")

# Vérifier que les variables sont bien chargées
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
artifacts_uri = os.getenv("ARTIFACTS_URI")

if not aws_access_key or not aws_secret_key:
    raise ValueError("❌ AWS credentials are missing! Check credentials.env")
if not artifacts_uri:
    raise ValueError("❌ ARTIFACTS_URI is not set! Check credentials.env")

# Initialiser le client S3 avec les credentials chargées
s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name="eu-west-3"  # Remplace par ta région AWS
)

# Configurer le bucket S3
BUCKET_NAME = "skin-dataset-project"
DATASET_DIR = "dataset_skin_types/Oily-Dry-Skin-Types"

# Fonction pour uploader un dossier sur S3
def upload_directory(local_directory, s3_prefix):
    for root, _, files in os.walk(local_directory):
        for file_name in files:
            local_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = f"{s3_prefix}/{relative_path}"
            
            # Upload vers S3
            s3.upload_file(local_path, BUCKET_NAME, s3_path)
            print(f"✅ Uploadé : s3://{BUCKET_NAME}/{s3_path}")

# Upload des datasets dans un dossier "skin-dataset"
upload_directory(f"{DATASET_DIR}/train", "skin-dataset/train")
upload_directory(f"{DATASET_DIR}/valid", "skin-dataset/valid")

print("🚀 Upload terminé avec succès !")