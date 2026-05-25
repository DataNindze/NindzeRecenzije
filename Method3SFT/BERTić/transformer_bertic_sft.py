import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)


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
    "Test 4: group 4": os.path.join(BASE_DIR,"test_4.tsv"),
}

MODEL_NAME = "classla/bcms-bertic"
MODEL_SHORT_NAME = "bertic"

VALID_LABELS = [
    "positive",
    "negative",
    "neutral",
    "mixed",
    "sarcasm"
]

MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42

MODEL_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "transformer_models",
    "bertic_sentiment"
)

CONFUSION_DIR = os.path.join(
    BASE_DIR,
    "transformer_confusion_matrices"
)

PREDICTIONS_DIR = os.path.join(
    BASE_DIR,
    "transformer_predictions"
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
# LABEL MAPPING
# =====================================================

labels_sorted = sorted(VALID_LABELS)

label_to_id = {
    label: idx
    for idx, label in enumerate(labels_sorted)
}

id_to_label = {
    idx: label
    for label, idx in label_to_id.items()
}


# =====================================================
# DATASET PREPARATION
# =====================================================

def prepare_hf_dataset(df, tokenizer):
    df = df.copy()

    df["labels"] = df["label"].map(label_to_id)

    dataset = Dataset.from_pandas(
        df[["text", "labels"]],
        preserve_index=False
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

    dataset = dataset.map(
        tokenize_batch,
        batched=True
    )

    dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch")

    return dataset


# =====================================================
# METRICS
# =====================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    precision = precision_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }


def format_scores(scores):
    return (
        f"P: {scores['precision']:.3f}, "
        f"R: {scores['recall']:.3f}, "
        f"F1: {scores['f1']:.3f}, "
        f"Acc: {scores['accuracy']:.3f}"
    )


# =====================================================
# MAIN
# =====================================================

print("CUDA available:", torch.cuda.is_available())
print("Model:", MODEL_NAME)

train_df = clean_dataset(
    read_dataset(TRAIN_FILE)
)

validation_df = clean_dataset(
    read_dataset(VALIDATION_FILE)
)

print("TRAIN rows:", len(train_df))
print("VALIDATION rows:", len(validation_df))
print("Labels:", label_to_id)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = prepare_hf_dataset(
    train_df,
    tokenizer
)

validation_dataset = prepare_hf_dataset(
    validation_df,
    tokenizer
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_to_id),
    id2label=id_to_label,
    label2id=label_to_id
)

training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    report_to="none",
    seed=RANDOM_SEED
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ]
)

print()
print("Starting fine-tuning...")
trainer.train()

print()
print("Saving final best model...")
trainer.save_model(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)


# =====================================================
# TEST EVALUATION
# =====================================================

result_row = {
    "#": "3.b",
    "method": "Transformers",
    "algorithm": "BERTić",
    "train": "TRAIN"
}

for test_name, test_path in TEST_FILES.items():
    print()
    print("Evaluating:", test_name)

    test_df = clean_dataset(
        read_dataset(test_path)
    )

    test_hf_dataset = prepare_hf_dataset(
        test_df,
        tokenizer
    )

    predictions_output = trainer.predict(test_hf_dataset)

    logits = predictions_output.predictions
    gold_ids = predictions_output.label_ids
    pred_ids = np.argmax(logits, axis=1)

    gold_labels = [
        id_to_label[int(i)]
        for i in gold_ids
    ]

    pred_labels = [
        id_to_label[int(i)]
        for i in pred_ids
    ]

    precision = precision_score(
        gold_labels,
        pred_labels,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        gold_labels,
        pred_labels,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        gold_labels,
        pred_labels,
        average="weighted",
        zero_division=0
    )

    accuracy = accuracy_score(
        gold_labels,
        pred_labels
    )

    scores = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

    result_row[test_name] = format_scores(scores)

    prediction_df = pd.DataFrame({
        "model": "BERTić TRAIN",
        "id": test_df.apply(make_id, axis=1),
        "text": test_df["text"],
        "predicted_label": pred_labels,
        "original_label": gold_labels,
    })

    prediction_df["a==b"] = (
        prediction_df["predicted_label"]
        == prediction_df["original_label"]
    )

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
        f"{MODEL_SHORT_NAME}_{safe_test_name}_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8-sig",
        sep=";"
    )

    errors_df = prediction_df[
        prediction_df["a==b"] == False
    ]

    errors_path = os.path.join(
        PREDICTIONS_DIR,
        f"{MODEL_SHORT_NAME}_{safe_test_name}_errors_only.csv"
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
        labels=labels_sorted
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels_sorted],
        columns=[f"pred_{label}" for label in labels_sorted]
    )

    cm_path = os.path.join(
        CONFUSION_DIR,
        f"{MODEL_SHORT_NAME}_{safe_test_name}_confusion_matrix.csv"
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

markdown_table = results_df.to_markdown(index=False)

content = f"""# Transformer Results - BERTić

## Task: Implementation 3.1 - Large Language Models / Transformers

Model:

- `{MODEL_NAME}`

Training set:

- TRAIN

Validation set:

- VALIDATION

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

## Hyperparameters

- max_length: {MAX_LENGTH}
- batch_size: {BATCH_SIZE}
- epochs: {EPOCHS}
- learning_rate: {LEARNING_RATE}
- weight_decay: {WEIGHT_DECAY}
- random_seed: {RANDOM_SEED}
- evaluation_strategy: epoch
- save_strategy: epoch
- early_stopping_patience: 2
- best_model_metric: eval_loss

## Outputs

Model saved to:

- `{MODEL_OUTPUT_DIR}`

Prediction files saved to:

- `transformer_predictions/`

Confusion matrices saved to:

- `transformer_confusion_matrices/`
"""

results_path = os.path.join(
    BASE_DIR,
    "results_transformers_bertic.md"
)

with open(results_path, "w", encoding="utf-8") as f:
    f.write(content)

results_df.to_csv(
    os.path.join(BASE_DIR, "transformer_results_bertic.csv"),
    index=False,
    encoding="utf-8-sig",
    sep=";"
)

print()
print("Gotovo.")
print("Napravljen je results_transformers_bertic.md")
print("Napravljen je transformer_results_bertic.csv")
print("Model spremljen u:", MODEL_OUTPUT_DIR)
print("Predictions spremljene u:", PREDICTIONS_DIR)
print("Confusion matrices spremljene u:", CONFUSION_DIR)
