# Importer les bibliothèques nécessaires
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données
file_path = r"C:\data\PS_20174392719_1491204439457_log.csv"
data = pd.read_csv(file_path)

# Variables clés pour l'analyse
key_features = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

# Visualisation des distributions
plt.figure(figsize=(15, 10))
for i, feature in enumerate(key_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[feature], bins=50, kde=True, color="skyblue")
    plt.title(f"Distribution de {feature}")
    plt.xlabel(feature)
    plt.ylabel("Fréquence")
plt.tight_layout()
plt.show()

# Boxplots pour détecter les valeurs aberrantes
plt.figure(figsize=(15, 10))
for i, feature in enumerate(key_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=data[feature], color="orange")
    plt.title(f"Boxplot de {feature}")
    plt.xlabel(feature)
plt.tight_layout()
plt.show()
