import os
import zipfile
import tensorflow as tf
import numpy as np
import pandas as pd
import mlflow
from mlflow import tensorflow as mlflow_tf
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.applications import MobileNetV2
import tensorflow.keras.backend as K
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt

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
mlflow.set_experiment("MobileNetV2-Skin-Type-Classification")

# --------------------
# 📥 Téléchargement et extraction du dataset
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
learning_rate = 0.0004
num_epochs = 50
batch_size = 32
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
# 📌 Scheduler natif
# --------------------
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-4 * (0.9 ** (epoch // 3))  # On diminue moins vite
)

# --------------------
# 📌 Préparation des générateurs de données
# --------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

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
# 📌 Poids des classes 
# --------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weights_dict = dict(enumerate(class_weights))
class_weights_dict[1] *= 0.9
class_weights_dict[2] *= 1.2

print("Poids des classes ajustés :", class_weights_dict)

# --------------------
# 📌 Définition du modèle MobileNetV2
# --------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

for layer in base_model.layers[-80:]:
    layer.trainable = True

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu", kernel_regularizer=l2(0.005))(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(64, activation="relu", kernel_regularizer=l2(0.005))(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
outputs = Dense(3, activation="softmax")(x)
model = Model(inputs=inputs, outputs=outputs)

# --------------------
# 📌 Compilation du modèle
# --------------------
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=['accuracy', AUC(name="auc"), Precision(name="precision"), Recall(name="recall"), f1_score_custom]
)

model.summary()

# --------------------
# 📌 Entraînement du modèle avec MLflow
# --------------------
early_stopping = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

with mlflow.start_run():
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", num_epochs)

    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=num_epochs,
        class_weight=class_weights_dict,  # ⬅ Ajout de la pondération
        callbacks=[early_stopping, reduce_lr, lr_scheduler],
        verbose=1
    )

    # --------------------
    # 📌 Log des métriques dans MLflow
    # --------------------
    for metric in history.history.keys():
        mlflow.log_metric(f"final_{metric}", history.history[metric][-1])

    # --------------------
    # 📌 Sauvegarde du modèle
    # --------------------
    model.save("mobilenetv2_skin_classifier.h5")
    mlflow.log_artifact("mobilenetv2_skin_classifier.h5")

print("✅ Modèle entraîné et sauvegardé avec succès dans MLflow!")

# --------------------
# 📌 Matrice de confusion
# --------------------

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# 📌 Récupération des labels de validation
y_true = valid_generator.classes
y_pred_probs = model.predict(valid_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# 📌 Matrice de confusion
conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title("Matrice de confusion")
plt.show()

# 📌 Rapport de classification
print(classification_report(y_true, y_pred))