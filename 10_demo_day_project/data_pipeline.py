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
        return "Oily-Dry-Skin-Types"  # ✅ Chemin générique pour Mac ET Docker

# 📌 Chargement des datasets
def load_datasets(dataset_path, img_size=(224, 224), batch_size=32):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path + "/train", image_size=img_size, batch_size=batch_size, label_mode="categorical"
    )
    valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path + "/valid", image_size=img_size, batch_size=batch_size, label_mode="categorical"
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path + "/test", image_size=img_size, batch_size=batch_size, label_mode="categorical"
    )
    return train_ds, valid_ds, test_ds

# 📌 Normalisation
def normalize_datasets(train_ds, valid_ds, test_ds):
    train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
    valid_ds = valid_ds.map(lambda x, y: (x / 255.0, y))
    test_ds = test_ds.map(lambda x, y: (x / 255.0, y))
    return train_ds, valid_ds, test_ds

# 📌 Augmentation des images
def get_image_generators():
    train_datagen = ImageDataGenerator(
        rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, brightness_range=[0.7, 1.3],
        channel_shift_range=50.0, horizontal_flip=True, fill_mode='nearest'
    )
    valid_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()
    return train_datagen, valid_datagen, test_datagen

# 📌 Création des générateurs
def get_data_generators(dataset_path, img_size=(224, 224), batch_size=32):
    train_datagen, valid_datagen, test_datagen = get_image_generators()
    
    train_generator = train_datagen.flow_from_directory(
        dataset_path + "/train", target_size=img_size, batch_size=batch_size, class_mode="categorical"
    )
    valid_generator = valid_datagen.flow_from_directory(
        dataset_path + "/valid", target_size=img_size, batch_size=batch_size, class_mode="categorical"
    )
    test_generator = test_datagen.flow_from_directory(
        dataset_path + "/test", target_size=img_size, batch_size=batch_size, class_mode="categorical"
    )
    
    return train_generator, valid_generator, test_generator

# 📌 Fonction principale pour récupérer les datasets
def get_data(batch_size=32, img_size=(224, 224)):
    dataset_path = get_dataset_path()
    train_ds, valid_ds, test_ds = load_datasets(dataset_path, img_size, batch_size)
    train_ds, valid_ds, test_ds = normalize_datasets(train_ds, valid_ds, test_ds)
    train_generator, valid_generator, test_generator = get_data_generators(dataset_path, img_size, batch_size)
    
    return train_generator, valid_generator, test_generator, dataset_path

if __name__ == "__main__":
    train_generator, valid_generator, test_generator, _ = get_data()
    print(f"✅ Train : {train_generator.samples} images")
    print(f"✅ Validation : {valid_generator.samples} images")
    print(f"✅ Test : {test_generator.samples} images")
