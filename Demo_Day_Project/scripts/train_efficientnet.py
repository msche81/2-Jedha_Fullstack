import os
import zipfile
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.applications import EfficientNetB0
import mlflow
import mlflow.tensorflow

# 📌 Configuration MLflow
mlflow.set_experiment("EfficientNetB0-Skin-Type-Classification")

# 📌 Définition du dataset
DATASET_URL = "https://skin-dataset-project-final.s3.amazonaws.com/dataset/oily-dry-and-normal-skin-types-dataset.zip"
DATASET_PATH = tf.keras.utils.get_file("skin_dataset.zip", origin=DATASET_URL, extract=False)
EXTRACT_PATH = os.path.join(os.path.dirname(DATASET_PATH), "Oily-Dry-Skin-Types")
os.makedirs(EXTRACT_PATH, exist_ok=True)

with zipfile.ZipFile(DATASET_PATH, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_PATH)

dataset_root = os.path.join(EXTRACT_PATH, "Oily-Dry-Skin-Types")

# 📌 Préparation des générateurs de données
img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_root, "train"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

valid_generator = valid_test_datagen.flow_from_directory(
    os.path.join(dataset_root, "valid"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_generator = valid_test_datagen.flow_from_directory(
    os.path.join(dataset_root, "test"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

# 📌 Définition du modèle EfficientNetB0
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu", kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu", kernel_regularizer=l2(0.05))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu", kernel_regularizer=l2(0.05))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(3, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

class FineTuningCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 10:
            print("\n🔓 Déverrouillage de 20 couches et réduction du learning rate 🔽")
            base_model.trainable = True
            new_lr = 0.0001
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
            print(f"✅ Nouveau learning rate : {new_lr:.6f}")

fine_tuning_callback = FineTuningCallback()

def f1_score_custom(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
    tp = tf.reduce_sum(tf.cast(tf.equal(y_true * y_pred, 1), tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.equal(y_true * (1 - y_pred), 0), tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.equal((1 - y_true) * y_pred, 0), tf.float32))
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

optimizer = Adam(learning_rate=0.0003)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy', AUC(), Precision(), Recall(), f1_score_custom])

# 📌 Entraînement du modèle avec MLflow
with mlflow.start_run():
    mlflow.tensorflow.autolog()
    history = model.fit(
        train_generator,
        epochs=30,
        validation_data=valid_generator,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                   ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
                   fine_tuning_callback],
        verbose=1
    )
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", 0.0003)

print("✅ Modèle entraîné et loggé avec MLflow.")
