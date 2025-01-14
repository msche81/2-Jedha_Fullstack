FROM python:3.9-slim

# Créer le dossier pour les fichiers d'application
WORKDIR /app

# Copier les fichiers locaux dans le conteneur
COPY . /app

# Installer les dépendances, y compris Streamlit
RUN pip install --no-cache-dir streamlit

# Spécifier la commande pour démarrer l'application
CMD ["streamlit", "run", "/app/app.py", "--server.port=8501", "--server.enableCORS=false"]