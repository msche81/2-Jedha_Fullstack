import pandas as pd
import boto3
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# 🏥 Définir les infos du bucket S3
BUCKET_NAME = "bucketdeployment"
FILE_KEY = "mlflow-doctolib/doctolib_simplified_dataset_01.csv"

# 📥 Télécharger le dataset depuis S3
def load_dataset():
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=BUCKET_NAME, Key=FILE_KEY)
    dataset = pd.read_csv(StringIO(response["Body"].read().decode("utf-8")))
    return dataset

# 🛠️ Feature Engineering
def preprocess_data(dataset):
    # Supprimer les trois premières colonnes inutiles
    dataset_filtered = dataset.iloc[:, 3:].copy()

    # Convertir la target en numérique
    dataset_filtered['No-show'] = dataset_filtered['No-show'].map({'Yes': 1, 'No': 0})

    # 🔹 Convertir les dates en format datetime
    dataset_filtered['ScheduledDay'] = pd.to_datetime(dataset_filtered['ScheduledDay'], utc=True)
    dataset_filtered['AppointmentDay'] = pd.to_datetime(dataset_filtered['AppointmentDay'], utc=True)

    # 🔹 Ajouter une nouvelle feature "DaysUntilAppointment"
    dataset_filtered["DaysUntilAppointment"] = (dataset_filtered["AppointmentDay"] - dataset_filtered["ScheduledDay"]).dt.days

    # 🔹 Rendre toutes les valeurs de délai positives (éviter les valeurs négatives)
    dataset_filtered["DaysUntilAppointment"] = dataset_filtered["DaysUntilAppointment"].abs()

    # 🔹 Supprimer les valeurs aberrantes d'âge (plage réaliste : 0 à 100 ans)
    dataset_filtered = dataset_filtered[(dataset_filtered["Age"] >= 0) & (dataset_filtered["Age"] <= 100)]

    # Supprimer les colonnes de dates après avoir créé 'DaysUntilAppointment'
    dataset_filtered_nodate = dataset_filtered.drop(columns=['ScheduledDay', 'AppointmentDay'])
    
    # Vérification des types de données avant transformation
    print("📌 Types de données avant transformation :")
    print(dataset_filtered_nodate.dtypes)

    # Vérification des colonnes avant transformation
    print("📌 Colonnes avant transformation :", dataset_filtered_nodate.columns)

    # Définir la target et les features
    X = dataset_filtered_nodate.drop(columns=["No-show"])
    Y = dataset_filtered_nodate["No-show"]

    # Forcer la conversion en catégorie (category) avant l'encodage
    dataset_filtered_nodate["Gender"] = dataset_filtered_nodate["Gender"].astype("category")
    dataset_filtered_nodate["Neighbourhood"] = dataset_filtered_nodate["Neighbourhood"].astype("category")

    # Identifier les colonnes numériques et catégorielles
    numeric_features = [col for col in X.columns if X[col].dtype in ['float64', 'int64', 'int32']]
    categorical_features = [col for col in X.columns if X[col].dtype.name in ['category', 'object']]

    # Vérification des colonnes catégorielles
    print("📌 Colonnes catégorielles avant transformation :", categorical_features)

    # Créer un pipeline de preprocessing
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return X, Y, preprocessor, dataset_filtered_nodate

# 🔧 Exécuter le preprocessing
if __name__ == "__main__":
    print("📥 Loading dataset...")
    dataset = load_dataset()

    print("🛠️ Preprocessing data...")
    X, Y, preprocessor, dataset_filtered_nodate = preprocess_data(dataset)

    # Diviser en train/test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Transformer les données
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print(f"✅ Preprocessing done! Train size: {X_train.shape}, Test size: {X_test.shape}")

    # Vérification des données transformées
    print("Sample of transformed X_train:")
    print(X_train_transformed[:5])  # Affiche un échantillon pour vérifier

    # Forcer la conversion en catégorie (category) avant l'encodage
    dataset_filtered_nodate["Gender"] = dataset_filtered_nodate["Gender"].astype("category")
    dataset_filtered_nodate["Neighbourhood"] = dataset_filtered_nodate["Neighbourhood"].astype("category")

    # Ensuite, tu peux appliquer .cat.codes
    dataset_filtered_nodate["Gender"] = dataset_filtered_nodate["Gender"].cat.codes
    dataset_filtered_nodate["Neighbourhood"] = dataset_filtered_nodate["Neighbourhood"].cat.codes

    # Sauvegarder les fichiers transformés pour `train.py`
    pd.DataFrame(X_train_transformed).to_csv("X_train.csv", index=False)
    pd.DataFrame(X_test_transformed).to_csv("X_test.csv", index=False)

    # Exporter en Excel
    dataset_filtered_nodate.to_excel("/home/user/app/preprocessed_data.xlsx", index=False)
    print("✅ Preprocessed dataset saved as Excel")