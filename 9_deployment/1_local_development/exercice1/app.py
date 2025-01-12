import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Charger les données
def load_data():
    url = "https://julie-2-next-resources.s3.eu-west-3.amazonaws.com/full-stack-full-time/linear-regression-ft/californian-housing-market-ft/california_housing_market.csv"
    print("📂 Chargement des données...")
    data = pd.read_csv(url)
    print("✅ Données chargées avec succès!")
    return data

# 2. Analyse des données
def analyze_data(data):
    print("\n🔍 Aperçu des données :")
    print(data.head())
    print("\n📊 Statistiques descriptives :")
    print(data.describe())
    print("\n🧹 Vérification des valeurs manquantes :")
    print(data.isnull().sum())

# 3. Préparer des données
def prepare_data(data):
    print("\n⚙️ Préparation des données...")
    X = data.drop('MedHouseVal', axis=1)
    y = data['MedHouseVal']
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("✅ Données prêtes pour l'entraînement!")
    return X_train, X_test, y_train, y_test

# 4. Entraîner le modèle
def train_model(X_train, y_train):
    print("\n🚀 Entraînement du modèle...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✅ Modèle entraîné!")
    return model

# 5. Evaluer le modèle
def evaluate_model(model, X_test, y_test):
    print("\n📈 Évaluation du modèle...")
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"📊 RMSE : {rmse:.2f}")

# 6. Sauvegarder le modèle
def save_model(model, filename='model.pkl'):
    print("\n💾 Sauvegarde du modèle...")
    import joblib
    joblib.dump(model, filename)
    print(f"✅ Modèle sauvegardé sous {filename}")

if __name__ == "__main__":
    # Charger et analyser les données
    data = load_data()
    analyze_data(data)

    # Préparer les données
    X_train, y_train, X_test, y_test = prepare_data(data)

    # Entraîner  et évaluer le modèle
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Sauvegarder le modèle
    save_model(model)
    