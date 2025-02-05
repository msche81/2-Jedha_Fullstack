import os
import zipfile
import tensorflow as tf
import numpy as np
import pandas as pd
import mlflow
from mlflow import tensorflow as mlflow_tf
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall
import tensorflow.keras.backend as K

# --------------------
# 📌 Chargement des variables d'environnement
# --------------------
load_dotenv("../.env")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
ARTIFACT_ROOT = os.getenv("ARTIFACT_ROOT")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Configuration MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Custom_cnn-Skin-Type-Classification")

# --------------------
# 📌 Chargement et extraction du dataset
# --------------------
dataset_url = "https://skin-dataset-project-final.s3.amazonaws.com/dataset/oily-dry-and-normal-skin-types-dataset.zip"
dataset_path = tf.keras.utils.get_file("dataset.zip", origin=dataset_url, extract=False)
extract_path = os.path.join(os.path.dirname(dataset_path), "Oily-Dry-Skin-Types")

os.makedirs(extract_path, exist_ok=True)
with zipfile.ZipFile(dataset_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

dataset_root = os.path.join(extract_path, "Oily-Dry-Skin-Types")
if not os.path.exists(dataset_root):
    raise FileNotFoundError(f"⚠️ Le dossier {dataset_root} n'existe pas. Vérifiez le téléchargement.")

print(f"✅ Dataset extrait avec succès dans : {dataset_root}")

# --------------------
# 📌 Définition des hyperparamètres dynamiques
# --------------------
learning_rate = 0.0002
num_epochs = 30
batch_size = 64
img_size = (224, 224)

# --------------------
# 📌 Définition de la fonction F1-score custom
# --------------------
def f1_score_custom(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    return 2 * (precision * recall) / (precision + recall + K.epsilon())

# --------------------
# 📌 Chargement et Préparation des Données
# --------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)  # Seulement la normalisation

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_root, "train"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

valid_generator = valid_datagen.flow_from_directory(
    os.path.join(dataset_root, "valid"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

# --------------------
# 📌 Définition du Modèle CNN
# --------------------
model = Sequential([
    Input(shape=(224, 224, 3)),
    
    Conv2D(64, (3,3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(256, (3,3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(512, (3,3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.005)),
    Dropout(0.4),
    Dense(64, activation='relu', kernel_regularizer=l2(0.005)),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy', AUC(name="auc"), Precision(name="precision"), Recall(name="recall"), f1_score_custom]
)

model.summary()

# --------------------
# 📌 Entraînement du Modèle
# --------------------
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6)

with mlflow.start_run():
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", num_epochs)

    history = model.fit(
        train_generator,
        epochs=num_epochs,
        validation_data=valid_generator,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # --------------------
    # 📌 Log des métriques dans MLflow
    # --------------------
    mlflow.log_metric("final_train_accuracy", history.history['accuracy'][-1])
    mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])
    mlflow.log_metric("final_train_auc", history.history['auc'][-1])
    mlflow.log_metric("final_val_auc", history.history['val_auc'][-1])
    mlflow.log_metric("final_train_precision", history.history['precision'][-1])
    mlflow.log_metric("final_val_precision", history.history['val_precision'][-1])
    mlflow.log_metric("final_train_recall", history.history['recall'][-1])
    mlflow.log_metric("final_val_recall", history.history['val_recall'][-1])
    mlflow.log_metric("final_train_f1_score", history.history['f1_score_custom'][-1])
    mlflow.log_metric("final_val_f1_score", history.history['val_f1_score_custom'][-1])

    # --------------------
    # 📌 Sauvegarde du Modèle
    # --------------------
    model.save("skin_type_classifier_model.h5")
    mlflow.log_artifact("skin_type_classifier_model.h5")

print("✅ Modèle entraîné et sauvegardé avec succès !")