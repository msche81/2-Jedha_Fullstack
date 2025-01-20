# Utilisation d'une image plus récente et légère
FROM python:3.9-slim 

# Mise à jour des paquets essentiels et installation des outils nécessaires
RUN apt-get update && apt-get install -y \
    nano unzip curl git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Installation d'AWS CLI (dernière version stable)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws

# Ajout d'un utilisateur dédié à Hugging Face pour éviter d'exécuter en root
RUN useradd -m -u 1000 app
USER app

# Définition des variables d'environnement pour éviter les problèmes de permissions
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Définition du dossier de travail
WORKDIR $HOME/app

# Copier tous les fichiers locaux dans le dossier de l'application avec les bonnes permissions
COPY --chown=app:app . /home/user/app/

# Installation de pip et des dépendances
RUN pip install --upgrade pip && \
    pip install -r /home/user/app/requirements.txt

# Spécifie le port 7860 (utilisé par Hugging Face)
ENV PORT=7860

# Configurer les variables AWS et MLflow
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV BACKEND_STORE_URI=${BACKEND_STORE_URI}
ENV ARTIFACT_STORE_URI=${ARTIFACT_STORE_URI}

# Définir l'URL locale
ENV APP_URI="http://0.0.0.0:7860"

# Lance le serveur MLflow avec le bon port"
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "7860", "--backend-store-uri", "sqlite:///mlflow.db"]
