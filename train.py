from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch # Importation de torch pour la gestion des périphériques

# --- Configuration du modèle ---
# Modèle de base : TinyLlama/TinyLlama-1.1B-Chat-v1.0
# C'est le modèle le plus adapté pour un chatbot en français.
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"Chargement du modèle : {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Définir le périphérique à utiliser (MPS pour Apple Silicon, sinon CPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device) # Déplacer le modèle vers le périphérique choisi
print(f"Le modèle est déplacé vers le périphérique : {device}")

# Pour TinyLlama, le pad_token est généralement déjà défini ou géré par le tokenizer.
# Cette section est commentée car elle n'est pas toujours nécessaire pour TinyLlama.
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
#     print("tokenizer.pad_token défini sur tokenizer.eos_token pour la compatibilité.")

# --- Chargement et préparation des données ---
print("Chargement des données depuis data.json...")
dataset = load_dataset("json", data_files="data.json")

# Fonction pour formater les prompts en fonction du modèle utilisé.
def generate_prompt(examples):
    formatted_texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        # Format spécifique à TinyLlama-Chat : <s>[INST] instruction [/INST]\noutput<eos_token>
        formatted_texts.append(f"""<s>[INST] {instruction} [/INST]
{output}{tokenizer.eos_token}""")
    return formatted_texts

# Fonction pour tokeniser et préparer les données pour l'entraînement
def tokenize_function(examples):
    formatted_texts = generate_prompt(examples)
    tokenized_output = tokenizer(
        formatted_texts,
        padding="max_length",
        truncation=True,
        max_length=128, # Longueur maximale réduite pour accélérer et économiser la mémoire
    )
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output

print("Tokenisation et formatage du dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True # Traite les exemples par lots pour plus d'efficacité
)

# --- Configuration de LoRA (Low-Rank Adaptation) ---
print("Configuration de LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    # Modules ciblés pour TinyLlama : q_proj, k_proj, v_proj, o_proj
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # Affiche le nombre de paramètres entraînés par LoRA

# --- Paramètres d'entraînement ---
print("Configuration des paramètres d'entraînement...")
training_args = TrainingArguments(
    output_dir="./football_chatbot_results", # Dossier pour sauvegarder les checkpoints
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1, # Réduit l'accumulation de gradients pour accélérer
    learning_rate=3e-4,
    num_train_epochs=1, # Réduit à 1 époque pour un test rapide et une accélération
    save_strategy="epoch", # Sauvegarde à la fin de chaque époque
    logging_steps=10, # Log toutes les 10 étapes
    save_total_limit=2, # Ne garde que les 2 derniers checkpoints
    report_to="none", # Désactive les rapports externes
    optim="adamw_torch",
    label_names=["labels"],
    # fp16=True, # COMMENTÉ : fp16 n'est pas compatible avec MPS pour l'entraînement dans cette configuration
)

# --- Création et démarrage du Trainer ---
print("Démarrage de l'entraînement...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

trainer.train()

# --- Sauvegarde du modèle final ---
print("Sauvegarde du modèle et du tokenizer finaux...")
model.save_pretrained("./football_chatbot_final")
tokenizer.save_pretrained("./football_chatbot_final")

# --- Fonction de test du chatbot ---
def ask_football_question(question):
    # Le prompt pour l'inférence doit correspondre au format TinyLlama
    prompt = f"<s>[INST] {question} [/INST]"

    inputs = tokenizer(prompt, return_tensors="pt")

    # --- Gestion du périphérique pour la génération (contournement MPS) ---
    original_device = model.device
    model.to("cpu") # Déplacer temporairement le modèle vers le CPU pour la génération
    inputs = {k: v.to("cpu") for k, v in inputs.items()} # Déplacer les inputs vers le CPU
    # -------------------------------------------------------------------

    # Paramètres de génération pour des réponses fluides et pertinentes
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )

    # --- Remettre le modèle sur son périphérique original après la génération ---
    model.to(original_device)
    # --------------------------------------------------------------------------

    print("\n--- Réponse du Chatbot ---")
    # Décodage de la réponse et nettoyage des espaces/retours à la ligne
    print(tokenizer.decode(outputs[0].to("cpu"), skip_special_tokens=True).strip())
    print("--------------------------\n")

# --- Test du chatbot avec des questions exemples ---
ask_football_question("Qui a gagné la Champions League 2023 ?")
ask_football_question("Quel est le meilleur buteur de l'histoire de la Premier League ?")
ask_football_question("Quelle est la capitale de la France ?") # Teste la réponse "Je ne peux pas t'aider"
#train.py : Le "Cerveau" du Robot (Phase d'Apprentissage)
