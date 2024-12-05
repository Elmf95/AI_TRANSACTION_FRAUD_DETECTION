import pandas as pd
import numpy as np

# Chemins des fichiers
input_path = r"C:\data\PS_20174392719_1491204439457_log.csv"
output_path = r"C:\data\transaction_fraudAI\cleaned_transaction_data.csv"


# Détection des outliers avec le z-score
def detect_outliers_zscore(df, threshold=3):
    z_scores = (df - df.mean()) / df.std()
    return z_scores.abs() > threshold


# Suppression des outliers
def remove_outliers(df, threshold=3):
    mask = detect_outliers_zscore(df.select_dtypes(include=[np.number]), threshold)
    filtered_df = df[~mask.any(axis=1)]
    return filtered_df


# Chargement des données
print("Chargement des données...")
data = pd.read_csv(input_path)

# Nettoyage des données
print("Détection et suppression des outliers...")
threshold = 3  # Seuil pour le z-score (modifiez si nécessaire)
filtered_data = remove_outliers(data, threshold)

# Sauvegarde des données nettoyées
print(f"Sauvegarde des données nettoyées dans {output_path}...")
filtered_data.to_csv(output_path, index=False)

print("Processus terminé.")
