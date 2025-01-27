import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, 
    BatchNormalization, Input
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from data_pipeline import get_data
import mlflow
import mlflow.tensorflow
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

# 📌 Définition des paramètres
batch_size = 32
img_size = (224, 224)
num_classes = 3  # Dry, Normal, Oily
learning_rate = 0.005
num_epochs = 2

# 📌 Initialisation de MLflow
mlflow.set_experiment("cnn_skin_type_classification")

with mlflow.start_run():
    print("✅ MLflow Run démarré")

    # 📌 Log des hyperparamètres
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("img_size", img_size)
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_epochs", num_epochs)

    print("✅ Hyperparamètres loggés dans MLflow")

    # 📌 Chargement des données
    train_generator, valid_generator, test_generator, _ = get_data(batch_size, img_size)
    print("✅ Infos des datasets loggées dans MLflow")

    # 📌 Définition du modèle CNN
    model = Sequential([
        Input(shape=(224, 224, 3)),  
        Conv2D(32, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),

        GlobalAveragePooling2D(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.6),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(num_classes, activation='softmax')
    ])

    # 📌 Compilation du modèle
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 📌 EarlyStopping pour éviter l'overfitting
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True, verbose=1
    )

    # 📌 Désactivation de l'autolog car il empêche la signature manuelle
    mlflow.tensorflow.autolog(disable=True)

    # 📌 Exemple d'entrée pour la signature MLflow (PRÉLEVÉ DU DATASET)
    example_batch = next(iter(train_generator))  # Prend un batch de l'iterator
    example_input = example_batch[0][:1]  # On prend la première image

    # 📌 Définition manuelle de la signature du modèle
    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, 224, 224, 3), name="input_image")]),
        outputs=Schema([TensorSpec(np.dtype(np.float32), (-1, num_classes), name="output_probabilities")])
    )

    # 📌 Entraînement du modèle
    history = model.fit(
        train_generator,
        epochs=num_epochs,
        validation_data=valid_generator,
        callbacks=[early_stopping],
        verbose=1
    )

    # 📌 Enregistrement des métriques finales
    final_loss, final_acc = model.evaluate(valid_generator)
    mlflow.log_metric("final_loss", final_loss)
    mlflow.log_metric("final_accuracy", final_acc)

    # 📌 Enregistrement du modèle avec signature et input_example
    mlflow.tensorflow.log_model(
        model, 
        "cnn_model",
        signature=signature, 
        input_example=example_input
    )

    print(f"✅ Modèle enregistré dans MLflow avec une accuracy de {final_acc:.4f}")

# 📌 Affichage du résumé du modèle
model.summary()