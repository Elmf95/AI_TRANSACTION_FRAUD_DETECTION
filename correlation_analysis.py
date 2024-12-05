# Importer les bibliothèques nécessaires
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file_path = r"C:\data\PS_20174392719_1491204439457_log.csv"
data = pd.read_csv(file_path)

# Filtrer les colonnes numériques pour la corrélation
numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns

# Calculer la matrice de corrélation
correlation_matrix = data[numeric_cols].corr()

# Afficher une heatmap des corrélations
plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True
)
plt.title("Heatmap des corrélations entre les variables numériques")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Identifier les corrélations les plus fortes avec la variable cible "isFraud"
fraud_corr = correlation_matrix["isFraud"].sort_values(ascending=False)
print("\nCorrélations avec la variable cible 'isFraud' :")
print(fraud_corr)
