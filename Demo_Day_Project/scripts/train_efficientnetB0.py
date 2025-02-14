import os
import zipfile
import tensorflow as tf
import numpy as np
import mlflow
import random
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.applications import EfficientNetB0
import tensorflow.keras.backend as K
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import collections
import cv2

# --------------------
# 📌 Chargement des variables d'environnement
# --------------------
load_dotenv("../.env")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("EfficientNetB0-Skin-Type-Classification")

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

# --------------------
# 📌 Hyperparamètres
# --------------------
initial_learning_rate = 1e-4
num_epochs = 70
batch_size = 64
img_size = (224, 224)


# --------------------
# 📌 Suppression des bordures noires des images
# --------------------
def remove_black_borders(image):
    try:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.uint8)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(np.vstack(contours))
            cropped_image = image[y:y+h, x:x+w]
            if w > 20 and h > 20:
                return cv2.resize(cropped_image, (224, 224))
        return cv2.resize(image, (224, 224))
    except:
        return cv2.resize(image, (224, 224))

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
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    preprocessing_function=remove_black_borders,
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_root, "train"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=seed_value
)

valid_generator = valid_datagen.flow_from_directory(
    os.path.join(dataset_root, "valid"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
    seed=seed_value
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
print("🔄 Poids des classes ajustés :", class_weights_dict)

# 🔍 Vérification des poids des classes
print("💡 Classes dans le dataset :", np.unique(train_generator.classes))
print("⚖️ Poids des classes :", class_weights_dict)

# Vérification de la distribution des classes
import collections
print("📊 Distribution des classes d'entraînement :", collections.Counter(train_generator.classes))
print("📊 Distribution des classes de validation :", collections.Counter(valid_generator.classes))

# --------------------
# 📌 Définition du modèle EfficientNetB0
# --------------------
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(64, activation="relu", kernel_regularizer=l2(0.002))(x)
x = Dropout(0.30)(x)
x = BatchNormalization()(x)

outputs = Dense(3, activation="softmax", kernel_regularizer=l2(0.002))(x)
model = Model(inputs=inputs, outputs=outputs)


# --------------------
# 📌 Compilation du modèle
# --------------------
model.compile(
    optimizer=Adam(learning_rate=initial_learning_rate),
    loss="categorical_crossentropy",
    metrics=['accuracy', AUC(name="auc"), Precision(name="precision"), Recall(name="recall"), F1Score(name="f1_score")]
)

model.summary()

# --------------------
# 📌 Callbacks
# --------------------
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    elif epoch < 15:
        return lr * 0.9
    else:
        return lr * 0.8

lr_scheduler = LearningRateScheduler(scheduler)

class UnfreezeCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 5:
            for layer in base_model.layers[-5:]:
                layer.trainable = True
        if epoch == 15:
            for layer in base_model.layers[-15:]:
                layer.trainable = True
        if epoch == 25:
            base_model.trainable = True

# --------------------
# 📌 Entraînement avec MLflow
# --------------------
class MonitorPredictionsCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch == 0:  # Vérification après la première epoch
                y_pred_probs = self.model.predict(valid_generator)
                y_pred_classes = np.argmax(y_pred_probs, axis=1)
                print(f"\n🔍 Distribution des prédictions après Epoch {epoch+1}: {collections.Counter(y_pred_classes)}")

with mlflow.start_run():
    mlflow.log_params({
        "learning_rate": initial_learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "class_weights": class_weights_dict
    })

    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=num_epochs,
        class_weight=class_weights_dict,
        callbacks=[UnfreezeCallback(), lr_scheduler, MonitorPredictionsCallback()],
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
    print("✅ Modèle EfficientNetB0 custom entraîné et sauvegardé avec succès dans MLflow!")

    # Sauvegarde la matrice de confusion dans MLflow
    mlflow.log_figure(plt.gcf(), "confusion_matrix.png")

# --------------------
# 📌 Matrice de confusion
# --------------------
y_true = valid_generator.classes
y_pred_probs = model.predict(valid_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title("Matrice de confusion")
plt.show()

print(classification_report(y_true, y_pred))
print("Distribution des prédictions :", collections.Counter(y_pred))