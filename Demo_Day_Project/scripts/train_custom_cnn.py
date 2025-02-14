import os
import zipfile
import tensorflow as tf
import numpy as np
import mlflow
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from sklearn.utils.class_weight import compute_class_weight
import collections
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

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
learning_rate = 0.0003
num_epochs = 25
batch_size = 16
img_size = (224, 224)

# --------------------
# 📌 Définition du F1-score custom
# --------------------
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + K.epsilon())
        recall = self.tp / (self.tp + self.fn + K.epsilon())
        return 2 * (precision * recall) / (precision + recall + K.epsilon())

# --------------------
# 📌 Préparation des générateurs de données
# --------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,  
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_root, "train"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
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

# 🔥 Ajustement manuel pour donner plus d'importance aux classes sous-représentées
for class_id in class_weights_dict:
    if class_weights_dict[class_id] > 2:
        class_weights_dict[class_id] *= 1.5

print("🔄 Nouveaux poids des classes :", class_weights_dict)

# --------------------
# 📌 Définition du modèle CNN custom
# --------------------
model = Sequential([
    Input(shape=(224, 224, 3)),

    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0001)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Dropout(0.2),

    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0001)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0001)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Dropout(0.4),

    GlobalAveragePooling2D(),

    Flatten(),

    Dense(3, activation='sigmoid')
])

# --------------------
# 📌 Compilation du modèle
# --------------------
model.compile(
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0),  
    loss="categorical_crossentropy",
    metrics=[
        'accuracy',
        AUC(name="auc"),
        Precision(name="precision"),
        Recall(name="recall"),
        F1Score(name="f1_score")
    ]
)

model.summary()

# --------------------
# 📌 Callbacks
# --------------------
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# --------------------
# 📌 Entraînement avec MLflow
# --------------------
with mlflow.start_run():
    mlflow.log_params({
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size
    })
    
    # 📌 Entraînement du modèle avec MLflow
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=num_epochs,
        class_weight=class_weights_dict,
        callbacks=[reduce_lr],
        verbose=1
    )

    # 📊 Logging des métriques à chaque époque pour visualiser les graphes dans MLflow
    for epoch in range(num_epochs):
        for metric, values in history.history.items():
            mlflow.log_metric(metric, values[epoch], step=epoch)

    # 📊 Logging des métriques finales pour qu'elles apparaissent sur la page principale des expériences
    final_metrics = {f"final_{metric}": values[-1] for metric, values in history.history.items()}
    mlflow.log_metrics(final_metrics)

    # 📊 Logging des métriques finales sous le bon nom pour MLflow
    mlflow.log_metric("final_f1_score_custom", history.history["f1_score"][-1])
    mlflow.log_metric("final_val_f1_score_custom", history.history["val_f1_score"][-1])

    # --------------------
    # 📌 Sauvegarde du modèle
    # --------------------
    mlflow.keras.log_model(model, "model")
    print("✅ Modèle CNN custom entraîné et sauvegardé avec succès dans MLflow!")

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

print("Distribution des prédictions :", collections.Counter(y_pred))
print("Distribution des classes dans le train :", collections.Counter(train_generator.classes))
print("Distribution des classes dans le valid :", collections.Counter(valid_generator.classes))