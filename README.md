# **AI Transaction Fraud Detection**

## **Description**

Ce projet utilise le machine learning pour détecter les transactions frauduleuses à partir de données financières. Le modèle a été entraîné avec **CatBoost**, un algorithme performant pour les données tabulaires, et offre une interface interactive basée sur **Gradio**. Il permet aux utilisateurs de tester le modèle avec leurs propres fichiers CSV ou avec un fichier d'exemple fourni dans le repository.

---

## **Fonctionnalités**

- **Détection de fraude** : Analyse des transactions et prédiction des cas frauduleux.
- **Interface interactive** : Chargement de fichiers CSV via une interface simple.
- **Modèle performant** : Utilise CatBoost avec un score AUC supérieur à 0.99.
- **Fichier d'exemple** : Un fichier CSV issu de Kaggle est inclus pour tester rapidement l'application.

---

## **Fichiers principaux**

- `catboost_fraud_detection.cbm` : Modèle pré-entraîné.
- `UI.py` : Script pour lancer l'interface interactive.
- `transaction_data_processed.csv` : Fichier d'exemple contenant des transactions issues de Kaggle.
- `requirements.txt` : Liste des dépendances nécessaires pour exécuter le projet.

---

## **Comment utiliser ce projet**

### **1. Cloner le dépôt**

```bash
git clone https://github.com/ton_nom_utilisateur/AI_TRANSACTION_FRAUD_DETECTION.git
````

## **Installation et utilisation**

### **1. Installer les dépendances**
Assurez-vous d'avoir Python 3.9+ installé. Installez les bibliothèques nécessaires :

```bash
pip install -r requirements.txt
````

### **2. Lancer l'application**

Exécutez le script UI.py pour ouvrir l'interface Gradio :

```bash
python UI.py
````
### **3. Utiliser l'interface**

Chargez un fichier CSV contenant les transactions à analyser, le fichier transaction_data_processed.csv est fourni dans le repo pour tester facilement le script.
Si vous n’avez pas de fichier, utilisez transaction_data_processed.csv inclus dans le projet.
Le modèle prédira quelles transactions sont frauduleuses (isFraud=1).

## Exemple de fichier d'entrée

Le fichier CSV doit inclure les colonnes suivantes (ou similaires si adaptées au domaine) :

- `TransactionID`, `Amount`, `Time`, `Category`, etc.
- **Colonne cible obligatoire** : `isFraud` (pour les tests).

**Note** : Le fichier d'exemple (`transaction_data_processed.csv`) est fourni pour vos tests.

---

## Technologies utilisées

- **Python 3.9** : Langage principal.
- **CatBoost** : Entraînement du modèle.
- **Gradio** : Interface utilisateur interactive.
- **Pandas, Numpy** : Manipulation et préparation des données.
- **SMOTE** : Équilibrage des classes lors de l’entraînement.

---

## Dataset

Le dataset d'entraînement provient de **Kaggle**, où il a été nettoyé et préparé pour le machine learning. Il est disponible en version transformée dans le fichier `transaction_data_processed.csv`.

---

## Hébergement

Vous pouvez héberger ce projet :

1. **Localement** : En exécutant le script `UI.py`.
2. **En ligne** :
   - Utilisez **Streamlit Cloud** ou **Hugging Face Spaces** pour le rendre public.
   - Fournissez l’URL générée aux utilisateurs ou recruteurs.

---

## Démo pour recruteurs

Pour tester le projet :

1. Téléchargez un fichier de transactions ou utilisez le fichier inclus (`transaction_data_processed.csv`).
2. Lancez le script `UI.py` pour prédire les fraudes.
3. Si hébergé, utilisez simplement le lien fourni par l’auteur pour accéder à l’application en ligne.

---

## Contributeur

Ce projet a été développé pour démontrer la capacité à :

- Préparer et nettoyer des données pour le machine learning.
- Entraîner un modèle performant.
- Concevoir une interface utilisateur intuitive.
- Rendre le modèle accessible via une solution d'hébergement.

Contactez-moi pour toute question ou collaboration.


