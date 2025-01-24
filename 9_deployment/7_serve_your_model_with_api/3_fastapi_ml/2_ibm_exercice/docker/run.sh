#!/bin/bash

# Définir le port par défaut si non spécifié
PORT=${PORT:-7860}  # Par défaut, Hugging Face Spaces utilise 7860

echo "🔄 Vérification de l'installation de Uvicorn..."
# Vérifier si Uvicorn est installé
if ! command -v uvicorn &> /dev/null
then
    echo "🚨 Erreur : Uvicorn n'est pas installé ou pas dans le PATH."
    exit 1
fi

echo "🔒 Vérification des permissions sur le fichier credentials..."
# Donner les bonnes permissions sur le fichier AWS credentials
chmod 600 /home/user/.aws/credentials

echo "🚀 Démarrage de l'application FastAPI sur le port $PORT..."

# Lancer l'application FastAPI avec Uvicorn
/usr/local/bin/uvicorn app:app --host 0.0.0.0 --port "$PORT"
