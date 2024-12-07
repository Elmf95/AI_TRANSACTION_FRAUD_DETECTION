import gradio as gr
import pandas as pd
from catboost import CatBoostClassifier

# Charger le modèle entraîné
model = CatBoostClassifier()
model.load_model("catboost_fraud_detection.cbm")


# Fonction pour prédire
def predict_fraud(file):
    data = pd.read_csv(file)
    predictions = model.predict(data)
    return pd.DataFrame(predictions, columns=["isFraud"])


# Interface utilisateur
with gr.Blocks() as demo:
    gr.Markdown("# Détection de Fraude - Interface Utilisateur")
    with gr.Row():
        with gr.Column():
            input_file = gr.File(label="Uploader un fichier CSV")
        with gr.Column():
            dropdown = gr.Dropdown(
                choices=["Transaction A", "Transaction B"], label="Choix Exemple"
            )
    submit_button = gr.Button("Prédire")

    # Zone de sortie
    output = gr.Dataframe(label="Prédictions")

    # Actions au clic
    submit_button.click(predict_fraud, inputs=[input_file], outputs=[output])

# Lancer l'application
demo.launch()
