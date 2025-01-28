import os
import zipfile
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------
# 📌 Téléchargement et Extraction du Dataset
# ----------------------
dataset_url = "https://skin-dataset-project.s3.amazonaws.com/oily-dry-and-normal-skin-types-dataset.zip"
dataset_path = tf.keras.utils.get_file(fname="oily-dry-and-normal-skin-types-dataset.zip", origin=dataset_url, extract=True)
dataset_root = os.path.join(os.path.dirname(dataset_path), "Oily-Dry-Skin-Types")

# ----------------------
# 📌 Vérification des datasets
# ----------------------
categories = ["dry", "normal", "oily"]
folders = ["train", "valid", "test"]
datasets = [os.path.join(dataset_root, folder) for folder in folders]

image_counts = {dataset: {cat: 0 for cat in categories} for dataset in datasets}
for dataset in datasets:
    for category in categories:
        category_path = os.path.join(dataset, category)
        if os.path.exists(category_path):
            image_counts[dataset][category] = len(os.listdir(category_path))

# ----------------------
# 📌 Préparation des Données
# ----------------------
img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
valid_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(os.path.join(dataset_root, "train"), target_size=img_size, batch_size=batch_size, class_mode="categorical")
valid_generator = valid_test_datagen.flow_from_directory(os.path.join(dataset_root, "valid"), target_size=img_size, batch_size=batch_size, class_mode="categorical")
test_generator = valid_test_datagen.flow_from_directory(os.path.join(dataset_root, "test"), target_size=img_size, batch_size=batch_size, class_mode="categorical")

# ----------------------
# 📌 Construction du Modèle CNN
# ----------------------
input_shape = (224, 224, 3)
num_classes = 3
learning_rate = 0.001

model = Sequential([
    Input(shape=input_shape),
    Conv2D(32, (3,3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.005)),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=l2(0.005)),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ----------------------
# 📌 Entraînement du Modèle
# ----------------------
num_epochs = 20
early_stopping = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1)

history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=valid_generator,
    callbacks=[early_stopping],
    verbose=1
)
