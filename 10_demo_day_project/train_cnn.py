import os
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from data_pipeline import get_data  # Importation des datasets

# 📌 Définition des paramètres
batch_size = 32
img_size = (224, 224)
num_classes = 3  # Dry, Normal, Oily
learning_rate = 0.005
num_epochs = 50

# 📌 Initialisation de MLflow
mlflow.set_experiment("cnn_skin_type_classification")

# 📌 Chargement des données
train_generator, valid_generator, test_generator, _ = get_data(batch_size, img_size)

with mlflow.start_run():
    # 📌 Enregistrement des hyperparamètres
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("img_size", img_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_epochs", num_epochs)
    
    # 📌 Définition du modèle CNN
    model = Sequential([
        Input(shape=(224, 224, 3)),  # Ajout explicite d'une couche Input
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
    
    # 📌 Enregistrement du modèle dans MLflow avant entraînement
    mlflow.tensorflow.autolog()
    
    # 📌 Entraînement du modèle
    history = model.fit(
        train_generator,
        epochs=num_epochs,
        validation_data=valid_generator,
        verbose=1  # Forcer l'affichage en Terminal
    )
    
    # 📌 Enregistrement des métriques finales
    final_loss, final_acc = model.evaluate(valid_generator)
    mlflow.log_metric("final_loss", final_loss)
    mlflow.log_metric("final_accuracy", final_acc)
    
    # 📌 Enregistrement du modèle dans MLflow
    mlflow.tensorflow.log_model(model, "cnn_model")

    print(f"✅ Modèle enregistré dans MLflow avec une accuracy de {final_acc:.4f}")

# 📌 Affichage du résumé du modèle
model.summary()
