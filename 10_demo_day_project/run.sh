#!/bin/bash

# Nom de l'image Docker
IMAGE_NAME="skin_type_classifier"

# Port fixe pour Hugging Face
PORT=7860

# Lancement du conteneur en mode interactif
docker run -it --rm -p $PORT:$PORT \
    -v ~/.aws:/home/user/.aws:ro \
    $IMAGE_NAME