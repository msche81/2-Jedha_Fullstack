import boto3
import os

# Configurer le bucket S3
BUCKET_NAME = "skin-dataset-project"  # Ton bucket dédié
DATASET_DIR = "dataset_skin_types/Oily-Dry-Skin-Types"

# Initialiser le client S3
s3 = boto3.client("s3")

# Fonction pour uploader un dossier et ses sous-dossiers sur S3
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
