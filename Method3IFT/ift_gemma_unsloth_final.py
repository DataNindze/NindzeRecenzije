import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datasets import Dataset
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig


# =====================================================
# SETTINGS
# =====================================================

BASE_DIR = "/content"

TRAIN_FILE = os.path.join(BASE_DIR, "TRAIN.csv")
VALIDATION_FILE = os.path.join(BASE_DIR, "VALIDATION.csv")

TEST_FILES = {
    "Test 1: group 1": os.path.join(BASE_DIR, "test_1.xlsx"),
    "Test 2: group 2": os.path.join(BASE_DIR, "test_2.xlsx"),
    "Test 3: group 3 (OURS)": os.path.join(BASE_DIR, "test_3.csv"),
    "Test 4: group 4": os.path.join(BASE_DIR, "test_4.tsv"),
}

MODEL_NAME = "unsloth/gemma-2-2b-it"
MODEL_SHORT_NAME = "gemma_2_2b_it_chat_ift"

VALID_LABELS = [
    "positive",
    "negative",
    "neutral",
    "mixed",
    "sarcasm"
]

MAX_SEQ_LENGTH = 512

EPOCHS = 2
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42

MODEL_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ift_models",
    "gemma_2_2b_sentiment_chat"
)

CONFUSION_DIR = os.path.join(
    BASE_DIR,
    "ift_confusion_matrices_chat"
)

PREDICTIONS_DIR = os.path.join(
    BASE_DIR,
    "ift_predictions_chat"
)

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFUSION_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


# =====================================================
# DATA LOADING
# =====================================================

def read_dataset(path):
    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)

    elif ext == ".tsv":
        df = pd.read_csv(
            path,
            sep="\t",
            encoding="utf-8-sig",
            engine="python"
        )

    elif ext == ".csv":
        try:
            df = pd.read_csv(
                path,
                sep=";",
                encoding="utf-8-sig",
                engine="python"
            )
        except Exception:
            df = pd.read_csv(
                path,
                sep=",",
                encoding="utf-8-sig",
                engine="python"
            )

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

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[df["label"].isin(VALID_LABELS)]
    df = df[df["text"].str.len() > 0]

    return df.reset_index(drop=True)


def make_id(row):
    review_id = str(row.get("review_id", "")).replace(".0", "")
    sentence_id = str(row.get("sentence_id", "")).replace(".0", "")

    if review_id and sentence_id:
        return review_id + "_" + sentence_id

    return str(row.name)


# =====================================================
# CHAT-TEMPLATE PROMPTS
# =====================================================

def make_user_message(text):
    return (
        "Classify the sentiment of the following Croatian medical review sentence.\n"
        "Choose exactly one label from: positive, negative, neutral, mixed, sarcasm.\n"
        "Return only one label, without explanation.\n\n"
        f"Sentence: {text}"
    )


def make_training_text(text, label, tokenizer):
    messages = [
        {
            "role": "user",
            "content": make_user_message(text)
        },
        {
            "role": "assistant",
            "content": label
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )


def make_inference_prompt(text, tokenizer):
    messages = [
        {
            "role": "user",
            "content": make_user_message(text)
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def build_sft_dataset(df, tokenizer):
    texts = []

    for _, row in df.iterrows():
        chat_text = make_training_text(
            row["text"],
            row["label"],
            tokenizer
        )

        texts.append(chat_text)

    return Dataset.from_dict({
        "text": texts
    })


# =====================================================
# METRICS
# =====================================================

def compute_scores(gold_labels, pred_labels):
    return {
        "precision": precision_score(
            gold_labels,
            pred_labels,
            average="weighted",
            zero_division=0
        ),
        "recall": recall_score(
            gold_labels,
            pred_labels,
            average="weighted",
            zero_division=0
        ),
        "f1": f1_score(
            gold_labels,
            pred_labels,
            average="weighted",
            zero_division=0
        ),
        "accuracy": accuracy_score(
            gold_labels,
            pred_labels
        ),
    }


def format_scores(scores):
    return (
        f"P: {scores['precision']:.3f}, "
        f"R: {scores['recall']:.3f}, "
        f"F1: {scores['f1']:.3f}, "
        f"Acc: {scores['accuracy']:.3f}"
    )


# =====================================================
# LOAD DATA
# =====================================================

print("CUDA available:", torch.cuda.is_available())

train_df = clean_dataset(read_dataset(TRAIN_FILE))
validation_df = clean_dataset(read_dataset(VALIDATION_FILE))

print("TRAIN rows:", len(train_df))
print("VALIDATION rows:", len(validation_df))
print("Model:", MODEL_NAME)

print("\nLabel distribution in TRAIN:")
print(train_df["label"].value_counts())

print("\nLabel distribution in VALIDATION:")
print(validation_df["label"].value_counts())


# =====================================================
# LOAD MODEL WITH UNSLOTH
# =====================================================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=RANDOM_SEED,
)


# =====================================================
# BUILD SFT DATASETS
# =====================================================

train_dataset = build_sft_dataset(train_df, tokenizer)
validation_dataset = build_sft_dataset(validation_df, tokenizer)


# =====================================================
# TRAINING
# =====================================================

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
    weight_decay=WEIGHT_DECAY,

    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,

    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),

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
    )
except TypeError:
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        args=sft_args,
    )

print()
print("Starting CHAT-TEMPLATE IFT fine-tuning...")
trainer.train()

print()
print("Saving fine-tuned chat-template IFT model...")
trainer.save_model(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

print("Model saved to:", MODEL_OUTPUT_DIR)


# =====================================================
# FIXED LABEL SCORING
# =====================================================

FastLanguageModel.for_inference(model)

def score_label_fixed(text, candidate_label):
    prompt = make_inference_prompt(text, tokenizer)
    full_text = prompt + candidate_label

    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        add_special_tokens=False
    ).to("cuda")

    full_inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        add_special_tokens=False
    ).to("cuda")

    input_ids = full_inputs["input_ids"]
    attention_mask = full_inputs["attention_mask"]

    prompt_len = prompt_inputs["input_ids"].shape[-1]

    labels = input_ids.clone()

    # Ignore prompt; compute loss only on candidate label tokens.
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    vocab_size = shift_logits.size(-1)

    token_losses = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100
    )

    token_losses = token_losses.view(shift_labels.size())

    valid_tokens = (shift_labels != -100).float()

    loss = (
        token_losses.sum(dim=1)
        / valid_tokens.sum(dim=1).clamp(min=1)
    )

    return loss.item()


def predict_one_fixed(text):
    scores = {}

    for label in VALID_LABELS:
        scores[label] = score_label_fixed(text, label)

    predicted_label = min(scores, key=scores.get)

    return predicted_label, str(scores)


# =====================================================
# EVALUATION ON ALL TEST SETS
# =====================================================

result_row = {
    "#": "3.c",
    "method": "IFT",
    "algorithm": "Gemma 2 2B IT",
    "train": "TRAIN"
}

for test_name, test_path in TEST_FILES.items():
    print()
    print("Evaluating:", test_name)

    test_df = clean_dataset(read_dataset(test_path))

    print("Label distribution in", test_name)
    print(test_df["label"].value_counts())

    ids = []
    texts = []
    gold_labels = []
    pred_labels = []
    raw_outputs = []

    for idx, row in test_df.iterrows():
        if idx % 50 == 0:
            print(f"Prediction {idx}/{len(test_df)}")

        predicted_label, raw_answer = predict_one_fixed(row["text"])

        ids.append(make_id(row))
        texts.append(row["text"])
        gold_labels.append(row["label"])
        pred_labels.append(predicted_label)
        raw_outputs.append(raw_answer)

    scores = compute_scores(
        gold_labels,
        pred_labels
    )

    print(test_name, "->", format_scores(scores))

    result_row[test_name] = format_scores(scores)

    prediction_df = pd.DataFrame({
        "model": "Gemma 2 2B IT Chat-template IFT fixed label scoring",
        "id": ids,
        "text": texts,
        "predicted_label": pred_labels,
        "original_label": gold_labels,
        "a==b": [
            pred == gold
            for pred, gold in zip(pred_labels, gold_labels)
        ],
        "raw_output": raw_outputs
    })

    safe_test_name = (
        test_name
        .lower()
        .replace(" ", "_")
        .replace(":", "")
        .replace("(", "")
        .replace(")", "")
    )

    prediction_path = os.path.join(
        PREDICTIONS_DIR,
        f"{MODEL_SHORT_NAME}_{safe_test_name}_fixed_scoring_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8-sig",
        sep=";"
    )

    errors_df = prediction_df[prediction_df["a==b"] == False]

    errors_path = os.path.join(
        PREDICTIONS_DIR,
        f"{MODEL_SHORT_NAME}_{safe_test_name}_fixed_scoring_errors_only.csv"
    )

    errors_df.to_csv(
        errors_path,
        index=False,
        encoding="utf-8-sig",
        sep=";"
    )

    cm = confusion_matrix(
        gold_labels,
        pred_labels,
        labels=VALID_LABELS
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in VALID_LABELS],
        columns=[f"pred_{label}" for label in VALID_LABELS]
    )

    cm_path = os.path.join(
        CONFUSION_DIR,
        f"{MODEL_SHORT_NAME}_{safe_test_name}_fixed_scoring_confusion_matrix.csv"
    )

    cm_df.to_csv(
        cm_path,
        encoding="utf-8-sig",
        sep=";"
    )


# =====================================================
# RESULTS.MD
# =====================================================

results_df = pd.DataFrame([result_row])

ordered_columns = [
    "#",
    "method",
    "algorithm",
    "train",
    "Test 1: group 1",
    "Test 2: group 2",
    "Test 3: group 3 (OURS)",
    "Test 4: group 4",
]

results_df = results_df[ordered_columns]

markdown_table = results_df.to_markdown(index=False)

content = f"""# IFT Results - Gemma 2 2B Instruct Fixed Label Scoring

## Task: Implementation 3.2 - Large Language Models / Instruction Fine-Tuning

Model:

- `{MODEL_NAME}`

Training set:

- TRAIN

Validation set:

- VALIDATION

Prompt format:

- Gemma chat template was used for instruction fine-tuning.
- Each example was formatted as a user instruction followed by the assistant response containing only the sentiment label.

Prediction method:

- Fixed label scoring was used during evaluation.
- Each candidate label was scored separately.
- The label with the lowest loss was selected as the prediction.

Labels:

- positive
- negative
- neutral
- mixed
- sarcasm

Evaluation metrics:

- weighted precision
- weighted recall
- weighted F1-score
- accuracy

## Results

{markdown_table}

## Note

The Gemma IFT model showed a strong bias toward the majority class. Its performance was lower than the transformer classification models, especially BERTić.

## Hyperparameters

- max_seq_length: {MAX_SEQ_LENGTH}
- batch_size: {BATCH_SIZE}
- gradient_accumulation_steps: {GRADIENT_ACCUMULATION_STEPS}
- epochs: {EPOCHS}
- learning_rate: {LEARNING_RATE}
- weight_decay: {WEIGHT_DECAY}
- LoRA rank: 16
- LoRA alpha: 16
- load_in_4bit: True
- random_seed: {RANDOM_SEED}
- eval_strategy: epoch
- save_strategy: epoch
- best_model_metric: eval_loss

## Outputs

Model saved to:

- `{MODEL_OUTPUT_DIR}`

Prediction files saved to:

- `ift_predictions_chat/`

Confusion matrices saved to:

- `ift_confusion_matrices_chat/`
"""

with open(os.path.join(BASE_DIR, "results_ift_gemma_all_tests.md"), "w", encoding="utf-8") as f:
    f.write(content)

results_df.to_csv(
    os.path.join(BASE_DIR, "ift_results_gemma_all_tests.csv"),
    index=False,
    encoding="utf-8-sig",
    sep=";"
)

print()
print("Gotovo.")
print("Napravljen je results_ift_gemma_all_tests.md")
print("Napravljen je ift_results_gemma_all_tests.csv")
print("Model spremljen u:", MODEL_OUTPUT_DIR)
print("Predictions spremljene u:", PREDICTIONS_DIR)
print("Confusion matrices spremljene u:", CONFUSION_DIR)
