import os
import pandas as pd


def unpack_data(input_dir, output_file="data/bronze/combined_data.csv"):
    """
    Exercice : Fonction pour décompresser et combiner plusieurs fichiers CSV à partir d'un répertoire en un seul fichier CSV.

    Objectifs :
    1. Lire tous les fichiers CSV dans un répertoire donné.
    2. Combiner les fichiers dans un seul DataFrame.
    3. Sauvegarder le DataFrame combiné dans un fichier CSV final.

    Étapes :
    - Parcourez tous les fichiers dans `input_dir`.
    - Vérifiez que les fichiers sont au format CSV ou contiennent "data-" dans leur nom.
    - Chargez chaque fichier dans un DataFrame Pandas.
    - Combinez tous les DataFrames dans un seul DataFrame.
    - Enregistrez le DataFrame combiné dans `output_file`.

    Paramètres :
    - input_dir (str) : Chemin vers le répertoire contenant les fichiers CSV.
    - output_file (str) : Chemin vers le fichier CSV combiné de sortie.

    Indices :
    - Utilisez `os.listdir` pour parcourir les fichiers.
    - Utilisez `os.path.join` pour construire le chemin complet des fichiers.
    - Utilisez `pd.read_csv` pour lire un fichier CSV en DataFrame.
    - Combinez les DataFrames avec `pd.concat`.
    - Sauvegardez le résultat avec `to_csv`.
    """

    all_dataframes = []
    
    # On parcourt tous les dossiers (on part du principe qu'ils sont tous ok pour être chargés)
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            try:
                # On sait que c'est des CSV malgré les extensions
                df = pd.read_csv(file_path, header=0)
                all_dataframes.append(df)
                print(f"Loaded file: {file_path}")
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
    

    if all_dataframes:
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        merged_file_path = os.path.join(output_file)
        
        merged_df.to_csv(merged_file_path, index=False)
        print(f"Merged file saved at: {merged_file_path}")
    else:
        print("No files to merge.")




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unpack and combine protein data")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output combined CSV file")
    args = parser.parse_args()

    unpack_data(args.input_dir, args.output_file)
