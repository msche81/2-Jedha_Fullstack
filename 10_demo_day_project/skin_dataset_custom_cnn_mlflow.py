import os
import zipfile
import tensorflow as tf
import numpy as np
import boto3
import mlflow
import mlflow.tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.metrics import f1_score
from dotenv import load_dotenv

# --------------------
# 📌 Charger les variables d'environnement
# --------------------
load_dotenv(os.path.join(os.path.dirname(__file__), 'credentials.env'))
print("AWS_ACCESS_KEY_ID:", os.getenv("AWS_ACCESS_KEY_ID"))
print("AWS_SECRET_ACCESS_KEY:", os.getenv("AWS_SECRET_ACCESS_KEY"))
print("AWS_DEFAULT_REGION:", os.getenv("AWS_DEFAULT_REGION"))
bucket_name = "skin-dataset-project"

# --------------------
# 📌 Configurer MLflow en local avec tracking URI
# --------------------
mlflow.set_tracking_uri("http://localhost:5001")  # URI de suivi MLflow local
mlflow.set_experiment("custom_cnn_skin_type_classification")  # Nom de l'expérience

# --------------------
# 📌 Initialisation de S3 pour MLflow (en local, on n'a plus besoin de S3)
# --------------------
s3 = boto3.client("s3")
bucket_name = "skin-dataset-project"

# Vérification de la connexion au bucket S3
try:
    s3.head_bucket(Bucket=bucket_name)
    print(f"✅ Successfully connected to S3 bucket: {bucket_name}")
except Exception as e:
    print(f"❌ Failed to connect to S3 bucket {bucket_name}: {e}")
    raise

# 📌 Téléchargement du dataset depuis S3
dataset_s3_path = "oily-dry-and-normal-skin-types-dataset.zip"
local_dataset_path = os.path.join(os.getcwd(), dataset_s3_path)

s3.download_file(bucket_name, dataset_s3_path, local_dataset_path)

# 📦 Extraction si nécessaire
dataset_root = os.path.join(os.path.dirname(local_dataset_path), "Oily-Dry-Skin-Types")
if not os.path.exists(dataset_root):
    with zipfile.ZipFile(local_dataset_path, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(local_dataset_path))
        print("✅ Extraction réussie !")
else:
    print("Dataset déjà extrait.")

# --------------------
# 📌 Préparation des données
# --------------------
img_size = (224, 224)
batch_size = 32

# Data augmentation améliorée
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  # Rotation plus importante
    width_shift_range=0.3,  # Décalage horizontal plus grand
    height_shift_range=0.3, # Décalage vertical plus grand
    shear_range=0.3,  # Distorsion diagonale plus importante
    zoom_range=0.3,  # Zoom supplémentaire
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],  # Changer la luminosité de manière aléatoire
    channel_shift_range=30.0  # Changement aléatoire de couleur
)

valid_test_datagen = ImageDataGenerator(rescale=1./255)  # Seulement la normalisation

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

# ✅ Vérification que les datasets contiennent bien des images
if train_generator.samples == 0 or valid_generator.samples == 0:
    raise ValueError("❌ Dataset vide ! Vérifie tes chemins.")

# --------------------
# 📌 Définir la fonction f1_score_custom
# --------------------
def f1_score_custom(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)  # Seuil de 0.5

    tp = tf.reduce_sum(tf.cast(tf.equal(y_true * y_pred, 1), tf.float32))  # True Positives
    fp = tf.reduce_sum(tf.cast(tf.equal(y_true * (1 - y_pred), 0), tf.float32))  # False Positives
    fn = tf.reduce_sum(tf.cast(tf.equal((1 - y_true) * y_pred, 0), tf.float32))  # False Negatives

    precision = tp / (tp + fp + tf.keras.backend.epsilon())  # Precision
    recall = tp / (tp + fn + tf.keras.backend.epsilon())  # Recall

    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())  # F1-scor

# --------------------
# 📌 Construction du modèle CNN
# --------------------
model = Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(32, (3,3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(256, (3,3), activation='relu', padding="same"),  # Ajout de nouvelles couches convolutives
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.005)),  # Augmenter la taille de la couche dense
    Dropout(0.6),  # Augmenter le taux de dropout
    Dense(64, activation='relu', kernel_regularizer=l2(0.005)),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Trois classes à prédire
])

# --------------------
# 📌 Optimisation et Compilation
# --------------------
learning_rate = 0.0005  # Réduire légèrement le taux d'apprentissage pour une meilleure convergence
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy', 
              metrics=['accuracy', AUC(), Precision(), Recall(), f1_score_custom])  # F1-score ajouté

model.summary()

# --------------------
# 📌 Entraînement du modèle
# --------------------
num_epochs = 30
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)  # Augmenter la patience
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=valid_generator,
    callbacks=[early_stopping, reduce_lr],
)

# --------------------
# 📌 Log des métriques dans MLflow
# --------------------
mlflow.log_param("learning_rate", learning_rate)
mlflow.log_param("epochs", num_epochs)
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
# 📌 Enregistrer et loguer le modèle
# --------------------
model_name = "cnn_skin_classifier"
with mlflow.start_run():
    # Log du modèle dans un répertoire spécifique au run
    mlflow.tensorflow.log_model(model, artifact_path=f"models/{model_name}")
    
    # Enregistrer le modèle dans le registre global
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/models/{model_name}"
    mlflow.register_model(model_uri, model_name)
    
    # Copier le modèle vers latest_model
    latest_model_path = f"s3://skin-dataset-project/deployment/custom_cnn_skin_classifier/latest_model/model.keras"
    s3.copy_object(
        Bucket="skin-dataset-project",
        CopySource=f"skin-dataset-project/deployment/{mlflow.active_run().info.experiment_id}/{mlflow.active_run().info.run_id}/models/{model_name}/model.keras",
        Key=latest_model_path
    )
    
    # Enregistrer les indices des classes
    mlflow.log_dict(train_generator.class_indices, "class_indices.json")

print("✅ Model training and tracking completed successfully!")