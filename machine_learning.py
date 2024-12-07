import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

file_path = r"C:\data\transaction_fraudAI\transaction_data_processed.csv"
data = pd.read_csv(file_path)

# Afficher la distribution initiale des classes
print("Distribution initiale des classes :")
print(data["isFraud"].value_counts().to_frame("count"))

# Séparer les caractéristiques et les étiquettes
X = data.drop(columns=["isFraud"])
y = data["isFraud"]

print(f"Dimensions de X : {X.shape}, y : {y.shape}")

# Identifier les colonnes catégorielles
categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Remplacer les NaN dans les colonnes catégorielles par "unknown" et convertir en chaînes
X[categorical_columns] = X[categorical_columns].fillna("unknown").astype(str)

# Encoder les colonnes catégorielles avec LabelEncoder
encoders = {col: LabelEncoder() for col in categorical_columns}
for col in categorical_columns:
    X[col] = encoders[col].fit_transform(X[col])

# Split stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Classes dans y_train après stratification :")
print(y_train.value_counts().to_frame("count"))

# Appliquer SMOTE pour équilibrer les classes
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nDistribution des classes après SMOTE :")
print(y_train.value_counts().to_frame("count"))

# Mettre à jour les encodeurs pour inclure "unknown"
for col in categorical_columns:
    # Ajouter 'unknown' comme une classe valide
    encoder_classes = encoders[col].classes_.tolist()
    if "unknown" not in encoder_classes:
        encoder_classes.append("unknown")
    encoders[col].classes_ = np.array(encoder_classes)  # Convertir en tableau NumPy

# Traiter les colonnes catégoriques dans X_test
for col in categorical_columns:
    X_test[col] = X_test[col].apply(
        lambda x: x if x in encoders[col].classes_ else "unknown"
    )
    X_test[col] = encoders[col].transform(X_test[col])

# Création des objets Pool pour CatBoost
train_data = Pool(X_train, label=y_train, cat_features=categorical_columns)
test_data = Pool(X_test, label=y_test, cat_features=categorical_columns)

# Définir et entraîner le modèle CatBoost
model = CatBoostClassifier(
    iterations=1000,
    depth=10,
    learning_rate=0.05,
    random_seed=42,
    task_type="GPU",  # Remplace par 'CPU' si GPU indisponible
    devices="0",  # Indique le GPU à utiliser
    loss_function="Logloss",
    eval_metric="AUC",
    early_stopping_rounds=500,
)

print("Entraînement du modèle CatBoost...")
model.fit(train_data, eval_set=test_data, verbose=100)

# Sauvegarder le modèle entraîné
model.save_model("catboost_fraud_detection.cbm")
print("Modèle sauvegardé dans 'catboost_fraud_detection.cbm'")

# Évaluation du modèle
print("\nÉvaluation sur les données de test :")
print(f"AUC : {model.get_best_score()['validation']['AUC']}")
