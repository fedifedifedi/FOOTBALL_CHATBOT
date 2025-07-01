import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# --- Configuration du modèle ---
# Assurez-vous que ce nom de modèle correspond à celui utilisé dans train.py
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_path = "./football_chatbot_final" # Chemin où le modèle fine-tuné est sauvegardé

print(f"Chargement du tokenizer et du modèle de base : {base_model_name}")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Charger le modèle LoRA fine-tuné
print(f"Chargement du modèle LoRA depuis : {model_path}")
model = PeftModel.from_pretrained(base_model, model_path)

# Définir le périphérique à utiliser (MPS pour Apple Silicon, sinon CPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval() # Mettre le modèle en mode évaluation
print(f"Modèle chargé sur {device}")

# Pour TinyLlama, le pad_token est généralement déjà défini ou géré.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Fonction de réponse du chatbot ---
def respond(message, chat_history):
    # Le message de l'utilisateur est déjà ajouté à chat_history par Gradio
    # Nous devons seulement générer la réponse de l'assistant et l'ajouter.

    # Format de prompt pour TinyLlama-Chat
    prompt = f"<s>[INST] {message} [/INST]"

    inputs = tokenizer(prompt, return_tensors="pt")

    # --- Gestion du périphérique pour la génération (contournement MPS) ---
    # Déplacer temporairement le modèle et les inputs vers le CPU pour la génération
    original_device = model.device
    model.to("cpu")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    # -------------------------------------------------------------------

    # Paramètres de génération
    outputs = model.generate(
        **inputs,
        max_new_tokens=50, # Essayez une valeur plus petite
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

    # --- Remettre le modèle sur son périphérique original après la génération ---
    model.to(original_device)
    # --------------------------------------------------------------------------

    # Décodage de la réponse
    response = tokenizer.decode(outputs[0].to("cpu"), skip_special_tokens=True)

    # Nettoyer la réponse pour ne garder que la partie générée par le modèle
    # Le format est "<s>[INST] message [/INST]\nréponse"
    # On cherche la fin du prompt pour extraire la réponse
    response_start_index = response.find("[/INST]")
    if response_start_index != -1:
        response = response[response_start_index + len("[/INST]"):]
        response = response.strip() # Supprimer les espaces et retours à la ligne inutiles

    # --- CORRECTION ICI : Ajouter la réponse de l'assistant au format dictionnaire ---
    chat_history.append({"role": "assistant", "content": response})
    # --------------------------------------------------------------------------------

    return "", chat_history

# --- Interface Gradio ---
with gr.Blocks() as demo:
    gr.Markdown("# Chatbot Football")
    gr.Markdown("Posez-moi des questions sur le football !")

    # Utilisation de type='messages' pour le chatbot
    chatbot = gr.Chatbot(type='messages', height=400)
    msg = gr.Textbox(label="Votre question")
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()
#app.py : Le "Corps" du Robot (Phase d'Utilisation)
