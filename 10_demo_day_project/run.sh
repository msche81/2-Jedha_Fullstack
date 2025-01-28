#!/bin/bash

# Nom de l'image Docker
IMAGE_NAME="skin_type_classifier"

# Lancement du conteneur en mode interactif
docker run -it --rm -p 7860:7860 \
    -v ~/.aws:/home/user/.aws:ro \
    $IMAGE_NAME
