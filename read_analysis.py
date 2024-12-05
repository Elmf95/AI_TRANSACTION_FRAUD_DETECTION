import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = r"C:\data\PS_20174392719_1491204439457_log.csv"

data = pd.read_csv(file_path)


print("Aperçu des données :")
print(data.head())
print("\nRésumé des colonnes :")
print(data.info())

# Statistiques descriptives
print("\nStatistiques descriptives :")
print(data.describe())

# Vérifier les valeurs uniques de la colonne de fraude
fraud_column = "isFraud"
if fraud_column in data.columns:
    print("\nRépartition des fraudes :")
    print(data[fraud_column].value_counts())

    # Créer un graphique en ajustant l'échelle pour mieux visualiser
    plt.figure(figsize=(10, 6))

    # Utilisation d'une échelle logarithmique
    sns.countplot(x=fraud_column, data=data)
    plt.yscale("log")
    plt.title("Distribution des transactions (Fraudes vs Non-Fraudes)")
    plt.ylabel("Nombre de transactions (échelle log)")
    plt.xlabel("Type de transaction (0 = Non-Fraude, 1 = Fraude)")

    # Ajouter des annotations pour les valeurs exactes
    counts = data[fraud_column].value_counts()
    for i, count in enumerate(counts):
        plt.text(i, count + 1000, f"{count:,}", ha="center", fontsize=10, color="black")

    plt.show()

    # Calculer le pourcentage de fraudes
    total_transactions = len(data)
    fraud_transactions = counts.get(1, 0)
    non_fraud_transactions = counts.get(0, 0)
    fraud_percentage = (fraud_transactions / total_transactions) * 100
    print(f"Pourcentage de fraudes : {fraud_percentage:.2f}%")

    # Zoom sur les transactions frauduleuses uniquement
    fraud_data = data[data[fraud_column] == 1]
    print("\nAperçu des transactions frauduleuses :")
    print(fraud_data.describe())

    # Afficher les transactions frauduleuses par type
    type_column = "type"
    if type_column in fraud_data.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(
            x=type_column,
            data=fraud_data,
            order=fraud_data[type_column].value_counts().index,
        )
        plt.title("Répartition des types de transactions frauduleuses")
        plt.xlabel("Type de transaction")
        plt.ylabel("Nombre de fraudes")
        plt.xticks(rotation=45)
        plt.show()
else:
    print("La colonne de fraude n'a pas été trouvée.")
