#!/bin/bash

# Nom de l'image Docker
IMAGE_NAME="skin_type_classifier"

# Build de l'image Docker
docker build -t $IMAGE_NAME .

# Lancement du conteneur en mode interactif
docker run -it --rm -p 7860:7860 $IMAGE_NAME