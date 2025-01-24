FROM python:3.10-slim

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

# Correctement pointer vers le répertoire .aws dans /home/user/app/
ENV AWS_SHARED_CREDENTIALS_FILE=/home/user/app/.aws/credentials

WORKDIR $HOME/app

RUN apt-get update -y && \
    apt-get install -y nano curl && \
    apt-get clean

# Copier tout le dossier en préservant la structure
COPY . /home/user/app

# Installer les dépendances depuis le fichier requirements.txt situé dans le dossier docker
RUN pip install --no-cache-dir -r /home/user/app/docker/requirements.txt

# S'assurer que run.sh est exécutable
RUN chmod +x /home/user/app/docker/run.sh

# Sécuriser les credentials en rendant le fichier uniquement lisible par l'utilisateur
RUN chmod 600 /home/user/app/.aws/credentials

# Accorder les bonnes permissions
RUN chmod -R 777 /home/user/app

# Vérifier où uvicorn est installé
RUN which uvicorn
RUN python -m uvicorn --version  # Vérifier que uvicorn peut être exécuté

EXPOSE 7860

CMD ["sh", "-c", "/usr/local/bin/python -m uvicorn app:app --host 0.0.0.0 --port $PORT"]
