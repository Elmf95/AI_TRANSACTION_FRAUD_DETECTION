import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Chemins des fichiers
input_path = r"C:\data\transaction_fraudAI\cleaned_transaction_data.csv"
output_path = r"C:\data\transaction_fraudAI\transaction_data_processed.csv"

print("Chargement des données...")
data = pd.read_csv(input_path)

# Vérification initiale de la distribution des classes
print("\nDistribution initiale des classes :")
print(data["isFraud"].value_counts())

# Sélection des colonnes numériques pour la normalisation et la standardisation
numeric_columns = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

# Création des scalers
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Normalisation des données
print("\nNormalisation des données...")
data_normalized = data.copy()
data_normalized[numeric_columns] = minmax_scaler.fit_transform(data[numeric_columns])

# Standardisation des données
print("\nStandardisation des données...")
data_standardized = data.copy()
data_standardized[numeric_columns] = standard_scaler.fit_transform(
    data[numeric_columns]
)

# Ajout des colonnes normalisées et standardisées dans le dataset original
print("\nAjout des données normalisées et standardisées au dataset...")
for col in numeric_columns:
    data[f"{col}_normalized"] = data_normalized[col]
    data[f"{col}_standardized"] = data_standardized[col]

# Vérification de la distribution des classes après le traitement
print("\nDistribution des classes après prétraitement (vérification) :")
print(data["isFraud"].value_counts())

# Enregistrement du fichier traité
print(f"\nEnregistrement des données traitées dans : {output_path}")
data.to_csv(output_path, index=False)

print("\nPrétraitement terminé avec succès.")
