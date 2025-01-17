# Utilisation d'une image plus récente et légère
FROM python:3.9-slim 

# Mise à jour des paquets essentiels et installation des outils nécessaires
RUN apt-get update && apt-get install -y \
    nano unzip curl git && \
    rm -rf /var/lib/apt/lists/*  # Nettoyage pour réduire la taille de l'image

# Installation d'AWS CLI (dernière version stable)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws  # Nettoyage des fichiers inutiles

# Ajout d'un utilisateur dédié à Hugging Face pour éviter d'exécuter en root
RUN useradd -m -u 1000 app
USER app

# Définition des variables d'environnement pour éviter les problèmes de permissions
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Définition du dossier de travail
WORKDIR $HOME/app

# Copie des fichiers locaux avec les bonnes permissions
COPY --chown=user . $HOME/app

# Installation de pip et des dépendances
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Spécifie le port 7860 (utilisé par Hugging Face)
ENV PORT=7860

# Configurer les variables pour Hugging Face
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV BACKEND_STORE_URI=${BACKEND_STORE_URI}
ENV ARTIFACT_STORE_URI=${ARTIFACT_STORE_URI}

# Lance le serveur MLflow avec le bon port

###CMD mlflow server -p $PORT \
   ### --host 0.0.0.0 \
   ### --backend-store-uri "sqlite:///mlflow.db" \
   ### --default-artifact-root "s3://bucketdeployment/mlflow-artifacts/"

CMD ["mlflow", "server", "-p", "7860", "--host", "0.0.0.0", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "s3://bucketdeployment/mlflow-artifacts/"]
