import pandas as pd
import json
import argparse
import os
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def train(X_train_path: str, y_train_path: str, output_dir: str):
    """Effectue l'entraînement et le GridSearch sur 6 modèles essentiels"""
    try:
        # Chargement des données
        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path).values.ravel()

        # Définir les modèles + hyperparamètres
        models = {
            'DecisionTree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {'n_estimators': [50, 100], 'max_depth': [None, 10]}
            },
            'SVM': {
                'model': SVC(),
                'params': {'kernel': ['linear', 'rbf'], 'C': [0.1, 1]}
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
            },
            'AdaBoost': {
                'model': AdaBoostClassifier(random_state=42),
                'params': {'n_estimators': [50, 100], 'learning_rate': [0.1, 1]}
            },
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=1000, solver='liblinear'),
                'params': {'C': [0.1, 1, 10], 'penalty': ['l2']}
            }
        }

        # Créer le dossier de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}

        # GridSearch et entraînement
        for name, config in models.items():
            print(f"\nEntraînement {name}...")

            # Initialisation de GridSearchCV
            grid = GridSearchCV(
                config['model'],
                config['params'],
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            grid.fit(X_train, y_train)
            
            # Enregistrer le modèle sous forme de dictionnaire
            model_info = {
                'model': str(grid.best_estimator_),  # Conversion en chaîne de caractères
                'accuracy': grid.best_score_,
                'best_params': grid.best_params_
            }
            results[name] = model_info
            print(f"{name} terminé \n Accuracy: {grid.best_score_:.2%}")

        # Sauvegarde des résultats dans un fichier JSON
        results_path = os.path.join(output_dir, "gridsearch_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nEntraînement terminé. Résultats sauvegardés dans : {os.path.abspath(output_dir)}")

    except Exception as e:
        print(f"\nErreur lors de l'entraînement : {str(e)}")
        raise

if __name__ == "__main__":
    # Utilisation d'argparse pour récupérer les chemins d'entrée et de sortie
    parser = argparse.ArgumentParser(description="Entraîne 6 modèles avec GridSearch")
    parser.add_argument('--X_train_path', required=True, help="Chemin vers X_train.csv")
    parser.add_argument('--y_train_path', required=True, help="Chemin vers y_train.csv")
    parser.add_argument('--output_dir', default="./models", help="Dossier de sortie pour les résultats")
    args = parser.parse_args()

    # Appel de la fonction d'entraînement
    train(args.X_train_path, args.y_train_path, args.output_dir)
