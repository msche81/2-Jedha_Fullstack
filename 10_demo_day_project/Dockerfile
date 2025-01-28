# Utilisation d'une image Python légère
FROM python:3.10-slim

# Définition des variables d'environnement
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

# Correctement pointer vers le fichier credentials AWS dans /home/user/app/
ENV AWS_SHARED_CREDENTIALS_FILE=/home/user/app/.aws/credentials

# Définition du répertoire de travail
WORKDIR $HOME/app

# Installation des outils de base
RUN apt-get update -y && \
    apt-get install -y nano curl && \
    apt-get clean

# Copier tous les fichiers du projet
COPY . /home/user/app

# Installation des dépendances depuis requirements.txt
RUN pip install --no-cache-dir -r /home/user/app/requirements.txt

# S'assurer que run.sh est exécutable
RUN chmod +x /home/user/app/run.sh

# Sécuriser les credentials AWS
RUN chmod 600 /home/user/app/.aws/credentials

# Accorder les bonnes permissions pour éviter les problèmes d'accès
RUN chmod -R 777 /home/user/app  

# Vérifier où est installé uvicorn (si besoin pour FastAPI)
RUN which uvicorn
RUN python -m uvicorn --version  

# Exposer le bon port pour Hugging Face
EXPOSE 7860                                              

# Commande par défaut pour démarrer l’application
CMD ["sh", "-c", "/usr/local/bin/python -m uvicorn app:app --host 0.0.0.0 --port $PORT"]