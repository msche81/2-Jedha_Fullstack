import os
import zipfile
import tensorflow as tf
import numpy as np
import mlflow
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall
import tensorflow.keras.backend as K
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import collections

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
mlflow.set_experiment("ResNet-Skin-Type-Classification")

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
learning_rate = 0.0001
num_epochs = 50
batch_size = 32
img_size = (224, 224)

# --------------------
# 📌 Définition du F1-score custom
# --------------------
class MultiClassF1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", num_classes=3, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)

        tp = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.not_equal(y_true, y_pred), tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.not_equal(y_pred, y_true), tf.float32))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + K.epsilon())
        recall = self.tp / (self.tp + self.fn + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1
    
# --------------------
# 📌 Générateurs de données
# --------------------
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_root, "train"), 
    target_size=img_size, 
    batch_size=batch_size, 
    class_mode="categorical",
    shuffle=True,
    interpolation="lanczos"  # Améliore la qualité du redimensionnement
)

valid_generator = valid_datagen.flow_from_directory(
    os.path.join(dataset_root, "valid"), target_size=img_size,
    batch_size=batch_size, class_mode="categorical"
)

# --------------------
# 📌 Poids des classes 
# --------------------
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(train_generator.classes),
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

#  --------------------
# 📌 Définition du modèle ResNet
# --------------------
# 📌 Chargement du modèle pré-entraîné (sans la dernière couche)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # On gèle le modèle de base au début

# 🔥 Ajout de la tête de classification
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(3, activation="softmax")(x)  # 3 classes

model = Model(inputs=inputs, outputs=outputs)

# --------------------
# 📌 Compilation du modèle
# --------------------
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=[
        'accuracy',
        AUC(name="auc"),
        Precision(name="precision"),
        Recall(name="recall"),
        MultiClassF1Score(name="f1_score", num_classes=3)
    ]
)

model.summary()

# --------------------
# 📌 Callbacks
# --------------------
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=8, min_lr=1e-6, verbose=1)

# 🔓 Déblocage des poids après 10 epochs
class ProgressiveUnfreeze(tf.keras.callbacks.Callback):
    def __init__(self, start_epoch=10, step=5):
        super().__init__()
        self.start_epoch = start_epoch
        self.step = step  # On dégèle progressivement

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.start_epoch and (epoch - self.start_epoch) % self.step == 0:
            num_layers_to_unfreeze = 10  # On dégèle 10 couches toutes les `step` epochs
            print(f"🔓 Dégel progressif de {num_layers_to_unfreeze} couches à l'epoch {epoch}")
            
            trainable_layers = [layer for layer in self.model.layers if not layer.trainable]
            for layer in trainable_layers[:num_layers_to_unfreeze]:
                layer.trainable = True

            self.model.compile(optimizer=Adam(learning_rate=learning_rate / 10), loss="categorical_crossentropy")

# --------------------
# 📌 Vérification des prédictions initiales
# --------------------
sample_images, sample_labels = next(valid_generator)
y_pred_probs = model.predict(sample_images)  # Probas softmax
y_pred_classes = np.argmax(y_pred_probs, axis=1)  # Conversion en classes

print("🔍 Probas des prédictions :", y_pred_probs[:5])
print("🔍 Classes prédites :", y_pred_classes[:5])
print("🔍 Labels réels (argmax) :", np.argmax(sample_labels[:5], axis=1))

# Aperçu des images de TRAINING
sample_images, sample_labels = next(train_generator)
fig, axes = plt.subplots(1, 5, figsize=(15, 5))

for i in range(5):
    ax = axes[i]
    ax.imshow(sample_images[i])
    ax.set_title(f"Classe: {np.argmax(sample_labels[i])}")
    ax.axis("off")

plt.show()

# Aperçu des images de VALIDATION
sample_images, sample_labels = next(valid_generator)
fig, axes = plt.subplots(1, 5, figsize=(15, 5))

for i in range(5):
    ax = axes[i]
    ax.imshow(sample_images[i])
    ax.set_title(f"Classe: {np.argmax(sample_labels[i])}")
    ax.axis("off")

plt.show()

# --------------------
# 📌 Entraînement avec MLflow
# --------------------
with mlflow.start_run():
    mlflow.log_params({
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size
    })

    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=num_epochs,
        class_weight=class_weights_dict,
        callbacks=[reduce_lr, ProgressiveUnfreeze(start_epoch=10, step=5)],
        verbose=1
    )

    # --------------------
    # 📌 Vérification des prédictions après entraînement
    # --------------------
    y_pred_probs = model.predict(valid_generator)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    print("📊 Distribution des prédictions :", collections.Counter(y_pred_classes))

    # 📊 Logging des métriques sur chaque epoch
    for epoch in range(num_epochs):
        for metric, values in history.history.items():
            try:
                mlflow.log_metric(metric, values[epoch], step=epoch)
            except Exception as e:
                print(f"⚠️ Erreur lors du logging de {metric} à l'epoch {epoch} : {e}")

    # 📊 Logging des métriques finales
    final_metrics = {}
    for metric, values in history.history.items():
        try:
            final_metrics[f"final_{metric}"] = values[-1]  # Dernière valeur de chaque métrique
        except Exception as e:
            print(f"⚠️ Impossible d'enregistrer {metric} en métrique finale : {e}")

    mlflow.log_metrics(final_metrics)

    # 🔥 Ajout des métriques spécifiques en cas d'absence
    if "f1_score" in history.history:
        mlflow.log_metric("final_f1_score", history.history["f1_score"][-1])
    if "val_f1_score" in history.history:
        mlflow.log_metric("final_val_f1_score", history.history["val_f1_score"][-1])
    if "precision" in history.history:
        mlflow.log_metric("final_precision", history.history["precision"][-1])
    if "val_precision" in history.history:
        mlflow.log_metric("final_val_precision", history.history["val_precision"][-1])
    if "recall" in history.history:
        mlflow.log_metric("final_recall", history.history["recall"][-1])
    if "val_recall" in history.history:
        mlflow.log_metric("final_val_recall", history.history["val_recall"][-1])

    # 📌 Sauvegarde du modèle
    mlflow.keras.log_model(model, "model")
    print("✅ Modèle ResNet entraîné et sauvegardé avec succès !")

# --------------------
# 📌 Matrice de confusion
# --------------------
y_pred_probs = model.predict(valid_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = valid_generator.classes

conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title("Matrice de confusion")
plt.show()

print(classification_report(y_true, y_pred))
print("📊 Distribution des prédictions :", collections.Counter(y_pred))