import argparse
import os

# 📌 Mapping des modèles vers leurs scripts respectifs
MODEL_SCRIPTS = {
    "custom_cnn": "skin_dataset_custom_cnn_mlflow.py",
    "mobilenetv2": "skin_dataset_mobilenetv2_mlflow.py",
    "efficientnet": "skin_dataset_efficientnet_mlflow.py"
}

def run_model(model_name):
    """ Exécute le script correspondant au modèle choisi """
    script_path = MODEL_SCRIPTS.get(model_name)

    if script_path is None:
        print(f"❌ Modèle '{model_name}' non reconnu. Choisis parmi : {list(MODEL_SCRIPTS.keys())}")
        return

    print(f"🚀 Lancement de {script_path}...")
    os.system(f"python /content/10_skin-dataset-project/{script_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exécute un modèle de classification de peau.")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_SCRIPTS.keys(),
                        help="Choisis un modèle: 'custom_cnn', 'mobilenetv2', 'efficientnet'")

    args = parser.parse_args()
    run_model(args.model)