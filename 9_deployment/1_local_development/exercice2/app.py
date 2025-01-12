import pandas as pd
import requests
import joblib
import json
from io import BytesIO

# Charger le modèle
def load_model(model_url):
    print("📥 Téléchargement du modèle...")
    response = requests.get(model_url)
    if response.status_code == 200:
        print("✅ Modèle téléchargé avec succès!")
        return joblib.load(BytesIO(response.content))
    else:
        raise Exception(f"❌ Échec du téléchargement du modèle. Code : {response.status_code}")

# Interroger l'API pour obtenir des données
def fetch_data(api_url):
    print("\n📡 Requête à l'API en cours...")
    response = requests.get(api_url, headers={"Accept": "application/json"})
    if response.status_code == 200:
        print("✅ Données récupérées avec succès!")
        print("Contenu brut (premiers caractères) :", response.text[:500])  # Debug : Inspecter le contenu

        try:
            # Charger le contenu JSON encodé
            data_json = json.loads(response.text)  # Transformation explicite en dictionnaire
            if isinstance(data_json, str):
                print("🔄 Double parsing requis : contenu JSON encodé détecté.")
                data_json = json.loads(data_json)  # Décoder à nouveau si nécessaire

            print("✅ Réponse convertie en JSON avec succès!")
            # Transformer les données en DataFrame
            df = pd.DataFrame(data=data_json["data"], columns=data_json["columns"])
            return df
        except json.JSONDecodeError:
            print("❌ Erreur : Impossible de convertir le texte en JSON.")
            return None
        except KeyError as e:
            print(f"❌ Erreur : Clé manquante dans la réponse JSON. {e}")
            return None
    else:
        raise Exception(f"❌ Échec de la requête API. Code : {response.status_code}")

# Faire une prédiction avec le modèle
def predict(model, data_df):
    print("\n📊 Prédiction en cours...")
    X = data_df.drop(columns=["MedHouseVal"])  # Supprimer la colonne cible
    prediction = model.predict(X)  # Faire une prédiction
    return prediction[0]

# Main
if __name__ == "__main__":
    # URLs
    MODEL_URL = "https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/house_prices_model.joblib"
    API_URL = "https://charlestng-house-prices-simple-api.hf.space/data"

    # Charger le modèle
    model = load_model(MODEL_URL)

    # Récupérer les données depuis l'API
    data_df = fetch_data(API_URL)

    # Faire une prédiction
    predicted_price = predict(model, data_df)

    # Simuler la vraie valeur pour comparaison
    actual_price = data_df["MedHouseVal"].iloc[0]  # Prendre la vraie valeur de la réponse API
    print(f"\nSelon notre modèle, cette maison devrait coûter : {predicted_price:.4f}")
    print(f"La vraie valeur est : {actual_price:.4f}")
    print(f"Notre modèle est à {abs(predicted_price - actual_price):.4f} près de la vérité.")