FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier tous les fichiers du projet
COPY . /app

# Installer les dépendances nécessaires
RUN pip install --no-cache-dir -r requirements.txt

# Lancer l'application Streamlit
CMD ["streamlit", "run", "covid_tracker.py", "--server.port=8501"]