import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# 📌 Envoyer les logs au conteneur Docker
mlflow.set_tracking_uri("http://localhost:7860")

# 📌 Nom de l'expérience MLflow
EXPERIMENT_NAME = "doctolib-no-show-prediction"
mlflow.set_experiment(EXPERIMENT_NAME)

# 📌 Charger les données prétraitées
print("📥 Loading preprocessed dataset...")
dataset = pd.read_excel("/home/user/app/preprocessed_data.xlsx")
print("Dataset loaded successfully.")

# 📌 Séparer features (X) et target (Y)
target_variable = "No-show"
X = dataset.drop(columns=[target_variable])
Y = dataset[target_variable]

# 📌 Séparer en train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 📌 Activer l'autolog pour scikit-learn dans MLflow
mlflow.sklearn.autolog()

# 📌 Lancer un run MLflow
with mlflow.start_run():
    print("Training model...")
    
    # 📌 Instancier et entraîner le modèle
    logistic_model = LogisticRegression(solver="saga", class_weight="balanced", C=0.1, random_state=42)
    logistic_model.fit(X_train, Y_train)

    # 📌 Prédictions et métriques
    Y_pred = logistic_model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)

    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ F1-score: {f1:.4f}")

    # 📌 Enregistrer le modèle comme artefact dans MLflow
    mlflow.sklearn.log_model(logistic_model, "model")

print("Training complete and logged in MLflow! 🚀")