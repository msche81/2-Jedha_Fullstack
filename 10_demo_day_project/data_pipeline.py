import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 📌 Détection de l'environnement
def get_dataset_path():
    if "COLAB_GPU" in os.environ:
        print("🔍 Détection : Environnement Google Colab")
        return "/content/dataset_skin_types/Oily-Dry-Skin-Types"
    else:
        print("🔍 Détection : Environnement Local (Mac / Docker)")
        return os.path.join(os.getcwd(), "Oily-Dry-Skin-Types")  # ✅ Path dynamique

# 📌 Vérification de l'existence du dataset
dataset_path = get_dataset_path()
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ ERREUR : Le dataset {dataset_path} est introuvable !")

# 📌 Création des générateurs d'images
def get_image_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # 📌 Normalisation entre 0 et 1
        rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, brightness_range=[0.7, 1.3],
        channel_shift_range=50.0, horizontal_flip=True, fill_mode='nearest'
    )
    valid_test_datagen = ImageDataGenerator(rescale=1./255)  # Pas d'augmentation sur validation et test

    return train_datagen, valid_test_datagen

# 📌 Chargement des datasets
def get_data_generators(dataset_path, img_size=(224, 224), batch_size=32):
    train_datagen, valid_test_datagen = get_image_generators()

    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_path, "train"), target_size=img_size,
        batch_size=batch_size, class_mode="categorical"
    )
    valid_generator = valid_test_datagen.flow_from_directory(
        os.path.join(dataset_path, "valid"), target_size=img_size,
        batch_size=batch_size, class_mode="categorical"
    )
    test_generator = valid_test_datagen.flow_from_directory(
        os.path.join(dataset_path, "test"), target_size=img_size,
        batch_size=batch_size, class_mode="categorical"
    )

    return train_generator, valid_generator, test_generator

# 📌 Fonction principale
def get_data(batch_size=32, img_size=(224, 224)):
    dataset_path = get_dataset_path()
    train_generator, valid_generator, test_generator = get_data_generators(dataset_path, img_size, batch_size)
    return train_generator, valid_generator, test_generator, dataset_path

if __name__ == "__main__":
    train_generator, valid_generator, test_generator, _ = get_data()
    print(f"✅ Train : {train_generator.samples} images")
    print(f"✅ Validation : {valid_generator.samples} images")
    print(f"✅ Test : {test_generator.samples} images")