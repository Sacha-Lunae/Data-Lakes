import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import tqdm
import joblib
from collections import OrderedDict
from sklearn.utils.class_weight import compute_class_weight
import json
from collections import defaultdict


def preprocess_data(data_file, output_dir):
    """
    Exercice : Fonction pour prétraiter les données brutes et les préparer pour l'entraînement de modèles.

    Objectifs :
    1. Charger les données brutes à partir d’un fichier CSV.
    2. Nettoyer les données (par ex. : supprimer les valeurs manquantes).
    3. Encoder les labels catégoriels (colonne `family_accession`) en entiers.
    4. Diviser les données en ensembles d’entraînement, de validation et de test selon une logique définie.
    5. Sauvegarder les ensembles prétraités et des métadonnées utiles.

    Étapes :
    - Charger les données avec `pd.read_csv`.
    - Supprimer les valeurs manquantes avec `dropna`.
    - Encoder les valeurs de `family_accession` en utilisant `LabelEncoder`.
    - Diviser les données en ensembles d’entraînement, de validation et de test.
    - Sauvegarder les données prétraitées en fichiers CSV (train.csv, dev.csv, test.csv).
    - Calculer et sauvegarder les poids de classes pour équilibrer les classes.

    Paramètres :
    - data_file (str) : Chemin vers le fichier CSV contenant les données brutes.
    - output_dir (str) : Répertoire où les fichiers prétraités et les métadonnées seront sauvegardés.

    Indices :
    - Utilisez `LabelEncoder` pour encoder les catégories.
    - Utilisez `train_test_split` pour diviser les indices des données.
    - Utilisez `to_csv` pour sauvegarder les fichiers prétraités.
    - Calculez les poids de classes en utilisant les comptes des classes.

    Défis bonus :
    - Assurez que les données très déséquilibrées sont bien réparties dans les ensembles.
    - Générez des fichiers supplémentaires comme un mapping des classes et les poids de classes.
    """

    # Étape 1 : Chargez les données brutes avec pd.read_csv
    df = pd.read_csv(data_file)

    # Étape 2 : Supprimez les valeurs manquantes avec dropna
    df = df.dropna(how="all", axis=1)
    df = df.dropna(how="all", axis=0)

    train = []
    val = []
    test = []

    # Étape 3 : Grouper les données par famille et appliquer les règles
    le = LabelEncoder()
    df['encoded_family'] = le.fit_transform(df["family_accession"])

    family_groups = df.groupby('encoded_family')

    for family, group in family_groups:
        group = group.sample(frac=1, random_state=42)  # Mélanger les données pour éviter les biais
        n = len(group)
        
        if n == 1:
            # Une seule observation : dans test
            test.append(group)
        elif n == 2:
            # Deux observations : une dans test, une dans val
            test.append(group.iloc[:1])
            val.append(group.iloc[1:])
        else:
            # Trois ou plus : une dans chaque
            test.append(group.iloc[:1])
            val.append(group.iloc[1:2])
            train.append(group.iloc[2:])

    # Étape 4 : Concaténer les données pour chaque ensemble
    train_df = pd.concat(train).reset_index(drop=True)
    val_df = pd.concat(val).reset_index(drop=True)
    test_df = pd.concat(test).reset_index(drop=True)

    # Étape 5 : Sauvegarder les ensembles
    train_df.to_csv(f"{output_dir}/train_data.csv", index=False)
    val_df.to_csv(f"{output_dir}/val_data.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_data.csv", index=False)

    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

    # Étape 6 : Calculez les poids de classes pour équilibrer les données
    # Sauvegardez les poids de classes dans un fichier texte
    classes = np.unique(df['encoded_family'])
    weights = compute_class_weight(
        class_weight='balanced', 
        classes=classes, 
        y=df['encoded_family']
    )

    # Convertir les poids en dictionnaire pour faciliter la sauvegarde
    class_weights = dict(zip(classes, weights))

    # Sauvegarder les poids dans un fichier texte
    output_file = f"{output_dir}/class_weights.json"
    with open(output_file, "w") as f:
        json.dump(class_weights, f)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess protein data")
    parser.add_argument("--data_file", type=str, required=True, help="Path to train CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the preprocessed files")
    args = parser.parse_args()

    preprocess_data(args.data_file, args.output_dir)
