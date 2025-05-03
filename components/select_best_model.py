import json
import argparse
import os
import shutil

def select_best_model(results_path: str, output_model_path: str, output_json_path: str):
    """Sélectionne le meilleur modèle en fonction de l'accuracy et le sauvegarde"""
    try:
        # Vérification que le fichier de résultats existe
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Le fichier de résultats {results_path} n'a pas été trouvé.")

        # Lecture des résultats du GridSearch
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Vérification du contenu du fichier results
        if not results:
            raise ValueError(f"Aucun résultat trouvé dans le fichier {results_path}.")

        # Trouver le modèle avec la meilleure accuracy
        best_model_name = None
        best_accuracy = -1
        best_model_info = None

        for model_name, info in results.items():
            print(f"Évaluation du modèle : {model_name} - Accuracy : {info.get('accuracy')}")
            if info.get('accuracy', 0) > best_accuracy:
                best_accuracy = info['accuracy']
                best_model_name = model_name
                best_model_info = info

        if best_model_name is None:
            raise ValueError("Aucun modèle n'a été trouvé avec les résultats d'accuracy.")

        # Vérifier que le chemin du modèle existe et le copier
        model_source_path = best_model_info['model_path']
        if not os.path.exists(model_source_path):
            raise FileNotFoundError(f"Le modèle source {model_source_path} n'a pas été trouvé.")

        # Créer le répertoire de destination si nécessaire
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        shutil.copy(model_source_path, output_model_path)

        # Sauvegarder les informations du meilleur modèle dans un fichier JSON
        best_model_info['best_model_name'] = best_model_name
        best_model_info['best_accuracy'] = best_accuracy
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as json_file:
            json.dump(best_model_info, json_file, indent=2)

        # Afficher les informations du meilleur modèle
        print(f"Meilleur modèle : {best_model_name}")
        print(f"Accuracy : {best_accuracy:.2%}")
        print(f"Modèle sauvegardé dans : {os.path.abspath(output_model_path)}")
        print(f"Résultats sauvegardés dans : {os.path.abspath(output_json_path)}")

    except Exception as e:
        print(f"Erreur : {str(e)}")
        raise

if __name__ == "__main__":
    # Configuration des arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Sélectionner le meilleur modèle basé sur GridSearch")
    parser.add_argument('--results_path', required=True, help="Chemin vers gridsearch_results.json")
    parser.add_argument('--output_model_path', required=True, help="Chemin pour sauvegarder le meilleur modèle (ex: ./output_model/best_model.joblib)")
    parser.add_argument('--output_json_path', required=True, help="Chemin pour sauvegarder les résultats du meilleur modèle (ex: ./output_results/best_model_results.json)")
    args = parser.parse_args()

    # Appel de la fonction de sélection du meilleur modèle
    select_best_model(args.results_path, args.output_model_path, args.output_json_path)
