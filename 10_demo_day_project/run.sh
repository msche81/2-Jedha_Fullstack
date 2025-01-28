#!/bin/bash

# Build de l'image Docker
docker build -t demo_day_project .

# Lancement du conteneur en mode interactif
docker run -it --rm -p 8501:8501 demo_day_project
