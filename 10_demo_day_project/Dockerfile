# Utilisation d'une image de base avec Python et Pip
FROM python:3.9

# Définition du répertoire de travail
WORKDIR /app

# Copie des fichiers nécessaires
COPY requirements.txt .
COPY . .

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Commande par défaut (peut être surchargée au run)
CMD ["python", "app.py"]
