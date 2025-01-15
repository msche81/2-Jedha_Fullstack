FROM python:3.11-slim

WORKDIR /app

COPY . /app

# Installer les dépendances système nécessaires pour numpy et pandas
RUN apt-get update && apt-get install -y \
    gcc g++ python3-dev libffi-dev \
    --no-install-recommends \
 && rm -rf /var/lib/apt/lists/*

# Mettre à jour pip et installer les dépendances Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "/app/app.py"]