import json
import joblib
import argparse
import os
import shutil

def select_best_model(results_path: str, output_model_path: str):
    """Sélectionne le meilleur modèle en fonction de l'accuracy et le sauvegarde"""
    try:
        # Lecture des résultats du GridSearch
        with open(results_path, 'r') as f:
            results = json.load(f)

        # Trouver le modèle avec la meilleure accuracy
        best_model_name = None
        best_accuracy = -1
        best_model_info = None

        for model_name, info in results.items():
            if info['accuracy'] > best_accuracy:
                best_accuracy = info['accuracy']
                best_model_name = model_name
                best_model_info = info

        if best_model_name is None:
            raise ValueError("Aucun modèle trouvé dans les résultats.")

        # Copier le modèle sélectionné vers le chemin de sortie
        model_source_path = best_model_info['model_path']
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        shutil.copy(model_source_path, output_model_path)

        # Afficher les informations du meilleur modèle
        print(f"Meilleur modèle : {best_model_name}")
        print(f"Accuracy : {best_accuracy:.2%}")
        print(f"Modèle sauvegardé dans : {os.path.abspath(output_model_path)}")

    except Exception as e:
        print(f"Erreur : {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sélectionner le meilleur modèle basé sur GridSearch")
    parser.add_argument('--results_path', required=True, help="Chemin vers gridsearch_results.json")
    parser.add_argument('--output_model_path', required=True, help="Chemin pour sauvegarder le meilleur modèle")
    args = parser.parse_args()

    select_best_model(args.results_path, args.output_model_path)
