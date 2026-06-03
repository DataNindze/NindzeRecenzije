import os
import re
import pandas as pd
import torch
import torch.nn.functional as F

# Stabilizacija okruženja
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

from datasets import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback

# --- POSTAVKA UREĐAJA ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- DIREKTORIJI I DATOTEKE ---
BASE_DIR = "."
TRAIN_FILE = os.path.join(BASE_DIR, "TRAIN.csv")
VALIDATION_FILE = os.path.join(BASE_DIR, "VALIDATION.csv")

# Definicija svih testnih datoteka
TEST_FILES = {
    "Test 1": os.path.join(BASE_DIR, "test_1.xlsx"),
    "Test 2": os.path.join(BASE_DIR, "test_2.xlsx"),
    "Test 3": os.path.join(BASE_DIR, "test_3.csv"),
    "Test 4": os.path.join(BASE_DIR, "test_4.tsv")
}

MODEL_NAME = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
MODEL_SHORT_NAME = "gemma_3_1b_fixed_scoring"

VALID_LABELS = ["positive", "negative", "neutral", "mixed", "sarcasm"]

# --- HIPERPARAMETRI ---
MAX_SEQ_LENGTH = 256           
EPOCHS = 5                     # Early stopping kontrolira stvarni kraj (strpljenje = 3)
BATCH_SIZE = 16                
GRADIENT_ACCUMULATION_STEPS = 8 # Efektivni batch = 128 (16 * 8)
LEARNING_RATE = 2e-4           
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42

MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "ift_models", "gemma_3_1b_fixed_scoring_all_tests")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "ift_predictions_gemma3_fixed_scoring_all_tests")
CONFUSION_DIR = os.path.join(BASE_DIR, "ift_confusion_matrices_gemma3_fixed_scoring_all_tests")

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(CONFUSION_DIR, exist_ok=True)


def read_dataset(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t", encoding="utf-8-sig", engine="python")
    elif ext == ".csv":
        try:
            # Primarno čitamo s točka-zarezom (;) jer je to tvoj standard
            df = pd.read_csv(path, sep=";", encoding="utf-8-sig", engine="python")
        except Exception:
            # Ako ne uspije, probaj standardni zarez (,)
            df = pd.read_csv(path, sep=",", encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported file type: {path}")
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


def clean_dataset(df):
    df = df.copy()
    df["text"] = (
        df["text"]
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].replace("sarcastic", "sarcasm")
    df = df[df["label"].isin(VALID_LABELS)]
    df = df[df["text"].str.len() > 0]
    return df.reset_index(drop=True)


def make_id(row):
    review_id = str(row.get("review_id", "")).replace(".0", "")
    sentence_id = str(row.get("sentence_id", "")).replace(".0", "")
    if review_id and sentence_id:
        return review_id + "_" + sentence_id
    return str(row.name)


# --- NAPREDNA INSTRUKCIJA IZ EUROLLM KODA ---
INSTRUCTION = (
    "Odredi sentiment sljedeće hrvatske rečenice. "
    "Odgovori isključivo jednom od oznaka: positive, negative, neutral, mixed, sarcasm."
)

def make_prompt(text):
    messages = [
        {"role": "user", "content": f"{INSTRUCTION}\n\nRečenica: {text}"},
    ]
    return messages


def build_sft_dataset(df, tokenizer):
    texts = []
    for _, row in df.iterrows():
        messages = make_prompt(row["text"])
        messages.append({"role": "assistant", "content": row["label"]})
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(full_text)
    return Dataset.from_dict({"text": texts})


def compute_scores(gold_labels, pred_labels):
    return {
        "precision": precision_score(gold_labels, pred_labels, average="weighted", zero_division=0),
        "recall": recall_score(gold_labels, pred_labels, average="weighted", zero_division=0),
        "f1": f1_score(gold_labels, pred_labels, average="weighted", zero_division=0),
        "accuracy": accuracy_score(gold_labels, pred_labels),
    }


def format_scores(scores):
    return (
        f"P: {scores['precision']:.3f}, "
        f"R: {scores['recall']:.3f}, "
        f"F1: {scores['f1']:.3f}, "
        f"Acc: {scores['accuracy']:.3f}"
    )


# Učitavanje podataka za trening i validaciju
print("Učitavanje osnovnih skupova podataka...")
train_df = clean_dataset(read_dataset(TRAIN_FILE))
validation_df = clean_dataset(read_dataset(VALIDATION_FILE))

# Učitavanje modela i tokenizatora preko Unslotha
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Konfiguracija LoRA adaptera
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=RANDOM_SEED,
)

train_dataset = build_sft_dataset(train_df, tokenizer)
validation_dataset = build_sft_dataset(validation_df, tokenizer)

# --- KONFIGURACIJA TRENINGA ---
sft_args = SFTConfig(
    output_dir=MODEL_OUTPUT_DIR,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.03,              
    lr_scheduler_type="cosine",
    weight_decay=WEIGHT_DECAY,
    logging_strategy="steps",
    logging_steps=10,
    eval_strategy="steps",           
    eval_steps=20,                    
    save_strategy="steps",           
    save_steps=20,
    save_total_limit=1,              
    load_best_model_at_end=True,      
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    report_to="none",
    seed=RANDOM_SEED,
)

try:
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        args=sft_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
except TypeError:
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        args=sft_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

print("\nStarting Instruction Fine-Tuning...")
trainer.train()

print("\nSaving best model...")
trainer.save_model(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

FastLanguageModel.for_inference(model)


# --- OPTIMIZIRANI ROBUSTAN SCORING PROLAZ KROZ CHAT TEMPLATE ---
def score_all_labels_batched(text):
    prompt_messages = make_prompt(text)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    prompt_len = prompt_inputs["input_ids"].shape[-1]

    seq_losses = []

    for label in VALID_LABELS:
        messages_with_label = make_prompt(text)
        messages_with_label.append({"role": "assistant", "content": label})
        
        full_text = tokenizer.apply_chat_template(messages_with_label, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        input_ids = inputs["input_ids"]
        
        labels_tokens = input_ids.clone()
        labels_tokens[:, :prompt_len] = -100  # Maskiranje instrukcije i teksta rečenice

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels_tokens[:, 1:].contiguous()
        
        vocab_size = shift_logits.size(-1)
        token_losses = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )
        
        valid_loss_tokens = token_losses[shift_labels.view(-1) != -100]
        if len(valid_loss_tokens) > 0:
            loss = valid_loss_tokens.mean().item()
        else:
            loss = 999.0
            
        seq_losses.append(loss)

    del inputs, input_ids, outputs, logits, token_losses
    return {label: float(loss) for label, loss in zip(VALID_LABELS, seq_losses)}


def predict_one_fixed(text):
    scores = score_all_labels_batched(text)
    predicted_label = min(scores, key=scores.get)
    return predicted_label, str(scores)


# --- EVALUACIJSKA PETLJA ZA SVE TESTNE SKUPOVE ---
results_for_markdown = []

for test_name, test_path in TEST_FILES.items():
    if not os.path.exists(test_path):
        print(f"\nUpozorenje: {test_path} ne postoji. Preskačem {test_name}.")
        continue

    print(f"\nEvaluating: {test_name} ({os.path.basename(test_path)})")
    test_df = clean_dataset(read_dataset(test_path))
    
    ids, texts, gold_labels, pred_labels, raw_outputs = [], [], [], [], []

    for idx, row in test_df.iterrows():
        if idx % 50 == 0:
            print(f"Prediction {idx}/{len(test_df)}")

        predicted_label, raw_answer = predict_one_fixed(row["text"])

        ids.append(make_id(row))
        texts.append(row["text"])
        gold_labels.append(row["label"])
        pred_labels.append(predicted_label)
        raw_outputs.append(raw_answer)

    # Izračun metriki
    scores = compute_scores(gold_labels, pred_labels)
    formatted_score_str = format_scores(scores)
    print(f"{test_name} -> {formatted_score_str}")
    
    # Dodavanje u listu za konačnu markdown tablicu
    results_for_markdown.append({
        "Test Set": test_name,
        "File": os.path.basename(test_path),
        "Scores": formatted_score_str
    })

    # Spremanje datoteka za ovaj konkretni test
    safe_name = test_name.lower().replace(" ", "_")
    
    prediction_df = pd.DataFrame({
        "model": f"Gemma 3 1B IT - Advanced Fixed Scoring ({test_name})",
        "id": ids,
        "text": texts,
        "predicted_label": pred_labels,
        "original_label": gold_labels,
        "a==b": [p == g for p, g in zip(pred_labels, gold_labels)],
        "raw_output": raw_outputs,
    })

    # 1. Sve predikcije (Spremanje s separatorom ;)
    prediction_path = os.path.join(PREDICTIONS_DIR, f"gemma_3_1b_fixed_scoring_{safe_name}_predictions.csv")
    prediction_df.to_csv(prediction_path, index=False, encoding="utf-8-sig", sep=";")

    # 2. Samo greške (Spremanje s separatorom ;)
    errors_df = prediction_df[prediction_df["a==b"] == False]
    errors_path = os.path.join(PREDICTIONS_DIR, f"gemma_3_1b_fixed_scoring_{safe_name}_errors_only.csv")
    errors_df.to_csv(errors_path, index=False, encoding="utf-8-sig", sep=";")

    # 3. Matrica zabune (Spremanje s separatorom ;)
    cm = confusion_matrix(gold_labels, pred_labels, labels=VALID_LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in VALID_LABELS],
        columns=[f"pred_{label}" for label in VALID_LABELS],
    )
    cm_path = os.path.join(CONFUSION_DIR, f"gemma_3_1b_fixed_scoring_{safe_name}_confusion_matrix.csv")
    cm_df.to_csv(cm_path, encoding="utf-8-sig", sep=";")


# --- GENERIRANJE ZAJEDNIČKOG MARKDOWN IZVJEŠTAJA ---
markdown_rows = []
for idx, res in enumerate(results_for_markdown, start=1):
    markdown_rows.append({
        "#": f"3.{idx}",
        "method": "IFT (Chat Template)",
        "algorithm": "Gemma 3 1B IT",
        "train": "TRAIN",
        f"Evaluated Dataset ({res['Test Set']})": res["Scores"]
    })
markdown_table = pd.DataFrame(markdown_rows).to_markdown(index=False)

content = f"""# IFT Results - Gemma 3 1B Advanced Fixed-Label Scoring (All Tests)

## Task: Multi-Dataset Evaluation with Instruction Fine-Tuning

Model:
- `{MODEL_NAME}`

Training set:
- TRAIN

Validation set:
- VALIDATION

Prompt format (Chat Template):
```text
<|im_start|>user
{INSTRUCTION}

Rečenica: ...<|im_end|>
<|im_start|>assistant
...
Training setup:
Training was run for {EPOCHS} epochs s uključenim Early Stopping-om (patience=3).
max_seq_length: {MAX_SEQ_LENGTH}
batch_size: {BATCH_SIZE}
gradient_accumulation_steps: {GRADIENT_ACCUMULATION_STEPS} (Efektivni batch: 128)
learning_rate: {LEARNING_RATE}
cosine scheduler and warmup (0.03) were used.

Prediction method:
Batched fixed-label scoring unutar službenih chat oznaka tokenizatora (Gemma 3 format).
Svih 5 kandidata evaluirano je kroz kauzalni log-likelihood maskirani prolaz.
Label s najnižim loss-om odabran je kao konačno predviđanje.

Labels:
positive, negative, neutral, mixed, sarcasm

Evaluation metrics:
weighted precision, weighted recall, weighted F1-score, accuracy

Summary Results Table
{markdown_table}

Hyperparameters:
max_seq_length: {MAX_SEQ_LENGTH}

batch_size: {BATCH_SIZE}

gradient_accumulation_steps: {GRADIENT_ACCUMULATION_STEPS}

epochs: {EPOCHS}

learning_rate: {LEARNING_RATE}

warmup_ratio: 0.03

lr_scheduler_type: cosine

weight_decay: {WEIGHT_DECAY}

LoRA rank: 16

LoRA alpha: 16

load_in_4bit: True

random_seed: {RANDOM_SEED}

Outputs Directories:
Models saved to: {MODEL_OUTPUT_DIR}

Predictions saved to: {PREDICTIONS_DIR}

Confusion Matrices saved to: {CONFUSION_DIR}
"""

results_path = os.path.join(BASE_DIR, "results_ift_gemma3_1b_fixed_scoring_all_tests.md")
with open(results_path, "w", encoding="utf-8") as f:
    f.write(content)  # POPRAVLJENO: Indentacija je sada ispravna

print("\n[ZAVRŠENO] Sve datoteke su uspješno procesuirane, evaluirane i spremljene!")
print("Zajednički Markdown izvještaj:", results_path)
