import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chemin du fichier
file_path = r"C:\data\PS_20174392719_1491204439457_log.csv"

# Charger les données
data = pd.read_csv(file_path)

# Aperçu des données
print("Aperçu des données :")
print(data.head())
print("\nRésumé des colonnes :")
print(data.info())

# Statistiques descriptives
print("\nStatistiques descriptives :")
print(data.describe())

# Vérifier les valeurs uniques de la colonne de fraude
fraud_column = "isFraud"  # Remplacez par le vrai nom de la colonne si différent
if fraud_column in data.columns:
    print("\nRépartition des fraudes :")
    print(data[fraud_column].value_counts())

    # Visualisation de la répartition
    sns.countplot(x=fraud_column, data=data)
    plt.title("Distribution des transactions (Fraudes vs Non-Fraudes)")
    plt.show()
else:
    print("La colonne de fraude n'a pas été trouvée.")
