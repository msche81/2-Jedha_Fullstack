from fastapi import FastAPI, Query
import pandas as pd
import boto3
from typing import List
from enum import Enum
import os

# 📌 Description détaillée de l'API (en Markdown)
description = """
🚀 **IBM HR Attrition API**  
Cette API permet d'explorer et d'analyser les données des employés d'IBM afin de comprendre les facteurs de départ.  

### 📌 **Fonctionnalités :**
- 🔍 **Prévisualisation** du dataset (`/preview`)
- 🎭 **Exploration** des valeurs uniques (`/unique-values`)
- 📊 **Analyse groupée** avec agrégation (`/groupby`)
- 🕵️‍♂️ **Filtrage des données** (`/filter-by`)
- 📈 **Analyse des extrêmes** (`/quantile`)

👩‍💻 **Développé avec FastAPI & hébergé sur AWS S3**  
"""

# 📌 Tags pour la documentation interactive
tags_metadata = [
    {
        "name": "Exploration",
        "description": "Endpoints permettant d'explorer le dataset.",
    },
    {
        "name": "Analyse",
        "description": "Endpoints permettant d'effectuer des statistiques et des analyses sur les données.",
    }
]

class AggregationMetric(str, Enum):
    mean = "mean"
    sum = "sum"
    max = "max"
    min = "min"

# Initialiser l'application FastAPI
app = FastAPI(
    title="IBM Employee Attrition API",
    description=description,
    contact={
        "name": "Marie-Sophie",
        "email": "contact@example.com"
    },  # <--- La virgule était peut-être manquante ici !
    version="1.0.0"
)

# Charger les données depuis S3
def load_data_from_s3():
    bucket_name = "ibm-employee-data"
    file_key = "ibm_hr_attrition.xlsx"
    local_file = "local_ibm_hr_attrition.xlsx"

    s3 = boto3.client("s3")

    try:
        print(f"🔄 Téléchargement du fichier {file_key} depuis S3...")
        s3.download_file(bucket_name, file_key, local_file)
        
        if os.path.exists(local_file):
            print(f"✅ Fichier téléchargé avec succès : {local_file}")
        else:
            print(f"❌ Erreur : fichier {local_file} introuvable après téléchargement !")
            return None

        return pd.read_excel(local_file)

    except Exception as e:
        print(f"🚨 Erreur lors du téléchargement depuis S3 : {e}")
        return None

df = load_data_from_s3()
if df is None:
    print("🚨 Impossible de charger les données ! Vérifiez l'accès S3.")

# Charger les données
df = load_data_from_s3()

@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API IBM HR Attrition!"}
    
# Endpoint `/preview`
@app.get("/preview", tags=["Exploration"])
def preview_data(rows: int = Query(5, description="Nombre de lignes à afficher")):
    """
    Affiche les premières lignes du dataset.
    """
    return df.head(rows).to_dict(orient="records")

# Endpoint `/unique-values`
@app.get("/unique-values", tags=["Exploration"])
def unique_values(column: str):
    """
    Renvoie les valeurs uniques d'une colonne donnée.
    """
    if column not in df.columns:
        return {"error": f"Colonne '{column}' non trouvée dans le dataset."}
    return df[column].dropna().unique().tolist()

# Endpoint `/groupby`
@app.get("/groupby", tags=["Analyse"])
def group_by(column: str, metric: AggregationMetric):
    """
    Applique une agrégation sur une colonne catégorielle.
    """
    if column not in df.columns:
        return {"error": f"Colonne '{column}' non trouvée."}

    grouped = getattr(df.groupby(column), metric.value)()
    return grouped.reset_index().to_dict(orient="records")

# Endpoint `/filter-by`
@app.get("/filter-by", tags=["Exploration"])
def filter_by(column: str, values: List[str]):
    """
    Filtre les lignes en fonction des valeurs dans une colonne donnée.
    """
    if column not in df.columns:
        return {"error": f"Colonne '{column}' non trouvée."}
    filtered_df = df[df[column].isin(values)]
    return filtered_df.to_dict(orient="records")

# Endpoint `/quantile`
@app.get("/quantile", tags=["Analyse"])
def quantile(column: str, percent: float, top: bool = True):
    """
    Renvoie les top ou bottom x% d'une colonne numérique.
    """
    if column not in df.columns:
        return {"error": f"Colonne '{column}' non trouvée."}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"La colonne '{column}' n'est pas numérique."}
    
    threshold = df[column].quantile(percent / 100)
    if top:
        result = df[df[column] >= threshold]
    else:
        result = df[df[column] <= threshold]
    return result.to_dict(orient="records")
