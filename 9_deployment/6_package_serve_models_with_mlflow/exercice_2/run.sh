#!/bin/bash

# Vérifier si des paramètres sont fournis
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run.sh <epochs> <learning_rate>"
    exit 1
fi

# Donner les permissions d’exécution à secrets.sh (au cas où)
chmod +x secrets.sh

# Charger les variables d'environnement
source secrets.sh

# Exécuter MLflow avec les paramètres donnés en argument
mlflow run . -P epochs=$1 -P lr=$2
