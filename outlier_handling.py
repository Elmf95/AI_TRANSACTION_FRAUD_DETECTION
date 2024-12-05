import pandas as pd
import numpy as np

# Chemins des fichiers
input_path = r"C:\data\PS_20174392719_1491204439457_log.csv"
output_path = r"C:\data\transaction_fraudAI\cleaned_transaction_data.csv"


# Détection des outliers avec le z-score
def detect_outliers_zscore(df, threshold=3):
    z_scores = (df - df.mean()) / df.std()
    return z_scores.abs() > threshold


# Suppression des outliers sauf pour les fraudes
def remove_outliers(df, threshold=3):
    # Séparer les fraudes
    fraud_data = df[df["isFraud"] == 1]
    non_fraud_data = df[df["isFraud"] == 0]

    # Identifier et supprimer les outliers uniquement sur les données non frauduleuses
    mask = detect_outliers_zscore(
        non_fraud_data.select_dtypes(include=[np.number]), threshold
    )
    filtered_non_fraud_data = non_fraud_data[~mask.any(axis=1)]

    # Réassembler le dataset avec les fraudes intactes
    cleaned_data = pd.concat([filtered_non_fraud_data, fraud_data], ignore_index=True)
    return cleaned_data


# Chargement des données
print("Chargement des données...")
data = pd.read_csv(input_path)

# Vérification initiale de la distribution des classes
print("\nDistribution initiale des classes :")
print(data["isFraud"].value_counts())

# Nettoyage des données
print("\nDétection et suppression des outliers...")
threshold = 3  # Seuil pour le z-score (modifiez si nécessaire)
filtered_data = remove_outliers(data, threshold)

# Vérification finale de la distribution des classes
print("\nDistribution des classes après nettoyage :")
print(filtered_data["isFraud"].value_counts())

# Sauvegarde des données nettoyées
print(f"\nSauvegarde des données nettoyées dans {output_path}...")
filtered_data.to_csv(output_path, index=False)

print("Processus terminé.")
