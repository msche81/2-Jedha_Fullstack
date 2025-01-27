import os
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from data_pipeline import get_data  # Chargement des datasets

# 📈 Paramètres généraux
batch_size = 32
img_size = (224, 224)
num_classes = 3  # Dry, Normal, Oily
learning_rate = 0.0005  # Plus petit pour éviter de casser les poids pré-entraînés
num_epochs = 30  # Moins d'epochs car le modèle est déjà partiellement entraîné

# 👉 Initialisation de MLflow
mlflow.set_experiment("transfer_learning_skin_type")

# 👉 Chargement des données
train_generator, valid_generator, test_generator, _ = get_data(batch_size, img_size)

with mlflow.start_run():
    # 👉 Enregistrement des hyperparamètres
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("img_size", img_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_epochs", num_epochs)

    # 👉 Charger MobileNetV2 sans la couche de classification finale
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # ❌ On ne fine-tune pas les couches pré-entraînées (pour l'instant)

    # 👉 Construction du modèle
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    # 👉 Compilation du modèle
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 👉 Enregistrement du modèle dans MLflow avant entraînement
    mlflow.tensorflow.autolog()

    # 👉 Entraînement du modèle
    history = model.fit(
        train_generator,
        epochs=num_epochs,
        validation_data=valid_generator,
	    verbose=1  # 🔍 Afficher les logs à chaque epoch
    )

    # 👉 Enregistrement des métriques finales
    final_loss, final_acc = model.evaluate(valid_generator)
    mlflow.log_metric("final_loss", final_loss)
    mlflow.log_metric("final_accuracy", final_acc)

    # 👉 Enregistrement du modèle dans MLflow
    mlflow.tensorflow.log_model(model, "transfer_learning_model")
    
    print(f"✅ Modèle MobileNetV2 enregistré dans MLflow avec une accuracy de {final_acc:.4f}")

# 👉 Affichage du résumé du modèle
model.summary()

