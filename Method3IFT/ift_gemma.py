import os
import pandas as pd
import torch
import torch.nn.functional as F

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

from datasets import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig


BASE_DIR = "."

TRAIN_FILE = os.path.join(BASE_DIR, "TRAIN.csv")
VALIDATION_FILE = os.path.join(BASE_DIR, "VALIDATION.csv")
TEST_FILE = os.path.join(BASE_DIR, "test_3.csv")

MODEL_NAME = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
MODEL_SHORT_NAME = "gemma_3_1b_fixed_scoring"

VALID_LABELS = ["positive", "negative", "neutral", "mixed", "sarcasm"]

MAX_SEQ_LENGTH = 128
EPOCHS = 2
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42

MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "ift_models", "gemma_3_1b_fixed_scoring_test3")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "ift_predictions_gemma3_fixed_scoring_test3")
CONFUSION_DIR = os.path.join(BASE_DIR, "ift_confusion_matrices_gemma3_fixed_scoring_test3")

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
            df = pd.read_csv(path, sep=";", encoding="utf-8-sig", engine="python")
        except Exception:
            df = pd.read_csv(path, sep=",", encoding="utf-8-sig", engine="python")
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
    df = df[df["label"].isin(VALID_LABELS)]
    df = df[df["text"].str.len() > 0]

    return df.reset_index(drop=True)


def make_id(row):
    review_id = str(row.get("review_id", "")).replace(".0", "")
    sentence_id = str(row.get("sentence_id", "")).replace(".0", "")

    if review_id and sentence_id:
        return review_id + "_" + sentence_id

    return str(row.name)


def make_prompt(text):
    return (
        "Labels: positive, negative, neutral, mixed, sarcasm\n"
        f"Sentence: {text}\n"
        "Label:"
    )


def make_training_text(text, label, tokenizer):
    return make_prompt(text) + " " + label + tokenizer.eos_token


def build_sft_dataset(df, tokenizer):
    texts = [
        make_training_text(row["text"], row["label"], tokenizer)
        for _, row in df.iterrows()
    ]
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


print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

train_df = clean_dataset(read_dataset(TRAIN_FILE))
validation_df = clean_dataset(read_dataset(VALIDATION_FILE))
test_df = clean_dataset(read_dataset(TEST_FILE))

print("\nTRAIN rows:", len(train_df))
print("VALIDATION rows:", len(validation_df))
print("TEST 3 rows:", len(test_df))

print("\nTRAIN label distribution:")
print(train_df["label"].value_counts())

print("\nTEST 3 label distribution:")
print(test_df["label"].value_counts())

print("\nModel:", MODEL_NAME)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

sample_text = "Doktorica je stručna, empatična i brižna."
sample_training = make_training_text(sample_text, "positive", tokenizer)

print("\n========== SAMPLE TRAIN TEXT ==========")
print(sample_training)
print("Contains label:", "positive" in sample_training)
print("=======================================\n")

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

train_dataset = build_sft_dataset(train_df, tokenizer)
validation_dataset = build_sft_dataset(validation_df, tokenizer)

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
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    weight_decay=WEIGHT_DECAY,

    logging_strategy="epoch",
    eval_strategy="no",
    save_strategy="no",
    load_best_model_at_end=False,

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

print("\nStarting Gemma 3 1B fixed-label scoring IFT...")
trainer.train()

print("\nSaving model...")
trainer.save_model(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
print("Model saved to:", MODEL_OUTPUT_DIR)


FastLanguageModel.for_inference(model)


def score_all_labels_batched(text):
    prompt = make_prompt(text) + " "

    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        add_special_tokens=False,
    )

    prompt_len = prompt_inputs["input_ids"].shape[-1]

    full_texts = [prompt + label for label in VALID_LABELS]

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    full_inputs = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        add_special_tokens=False,
    ).to("cuda")

    tokenizer.padding_side = old_padding_side

    input_ids = full_inputs["input_ids"]
    attention_mask = full_inputs["attention_mask"]

    labels = input_ids.clone()

    for i in range(len(VALID_LABELS)):
        n_pad = (attention_mask[i] == 0).sum().item()
        labels[i, :n_pad + prompt_len] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    vocab_size = shift_logits.size(-1)

    token_losses = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )

    token_losses = token_losses.view(shift_labels.size())
    valid_tokens = (shift_labels != -100).float()

    seq_losses = token_losses.sum(dim=1) / valid_tokens.sum(dim=1).clamp(min=1)

    return {
        label: float(loss)
        for label, loss in zip(VALID_LABELS, seq_losses)
    }


def predict_one_fixed(text):
    scores = score_all_labels_batched(text)
    predicted_label = min(scores, key=scores.get)
    return predicted_label, str(scores)


print("\nEvaluating: Test 3")

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

scores = compute_scores(gold_labels, pred_labels)
print("Test 3 ->", format_scores(scores))

prediction_df = pd.DataFrame({
    "model": "Gemma 3 1B IT fixed-label scoring IFT",
    "id": ids,
    "text": texts,
    "predicted_label": pred_labels,
    "original_label": gold_labels,
    "a==b": [
        pred == gold
        for pred, gold in zip(pred_labels, gold_labels)
    ],
    "raw_output": raw_outputs,
})

prediction_path = os.path.join(
    PREDICTIONS_DIR,
    "gemma_3_1b_fixed_scoring_test_3_predictions.csv",
)

prediction_df.to_csv(
    prediction_path,
    index=False,
    encoding="utf-8-sig",
    sep=";",
)

errors_df = prediction_df[prediction_df["a==b"] == False]

errors_path = os.path.join(
    PREDICTIONS_DIR,
    "gemma_3_1b_fixed_scoring_test_3_errors_only.csv",
)

errors_df.to_csv(
    errors_path,
    index=False,
    encoding="utf-8-sig",
    sep=";",
)

cm = confusion_matrix(
    gold_labels,
    pred_labels,
    labels=VALID_LABELS,
)

cm_df = pd.DataFrame(
    cm,
    index=[f"true_{label}" for label in VALID_LABELS],
    columns=[f"pred_{label}" for label in VALID_LABELS],
)

cm_path = os.path.join(
    CONFUSION_DIR,
    "gemma_3_1b_fixed_scoring_test_3_confusion_matrix.csv",
)

cm_df.to_csv(
    cm_path,
    encoding="utf-8-sig",
    sep=";",
)

markdown_table = pd.DataFrame([{
    "#": "3.c",
    "method": "IFT",
    "algorithm": "Gemma 3 1B IT",
    "train": "TRAIN",
    "Test 3: group 3 (OURS)": format_scores(scores),
}]).to_markdown(index=False)

content = f"""# IFT Results - Gemma 3 1B Fixed-Label Scoring

## Task: Implementation 3.2 - Large Language Models / Instruction Fine-Tuning

Model:

- `{MODEL_NAME}`

Training set:

- TRAIN

Validation set:

- VALIDATION

Prompt format:

```text
Labels: positive, negative, neutral, mixed, sarcasm
Sentence: ...
Label: ...
Training setup:

Training was run for {EPOCHS} epochs.
max_seq_length: {MAX_SEQ_LENGTH}
batch_size: {BATCH_SIZE}
gradient_accumulation_steps: {GRADIENT_ACCUMULATION_STEPS}
learning_rate: {LEARNING_RATE}
cosine scheduler and warmup were used.
intermediate validation/checkpointing was disabled for stability.

Prediction method:

Batched fixed-label scoring was used.
All five candidate labels were scored in one batched forward pass.
The label with the lowest loss was selected as the prediction.

Labels:

positive
negative
neutral
mixed
sarcasm

Evaluation metrics:

weighted precision
weighted recall
weighted F1-score
accuracy
Result on Test 3

{markdown_table}

Hyperparameters
max_seq_length: {MAX_SEQ_LENGTH}
batch_size: {BATCH_SIZE}
gradient_accumulation_steps: {GRADIENT_ACCUMULATION_STEPS}
epochs: {EPOCHS}
learning_rate: {LEARNING_RATE}
warmup_ratio: 0.05
lr_scheduler_type: cosine
weight_decay: {WEIGHT_DECAY}
LoRA rank: 16
LoRA alpha: 16
load_in_4bit: True
random_seed: {RANDOM_SEED}
Outputs

Model saved to:

{MODEL_OUTPUT_DIR}

Prediction file:

{prediction_path}

Errors only:

{errors_path}

Confusion matrix:

{cm_path}
"""

results_path = os.path.join(
BASE_DIR,
"results_ift_gemma3_1b_fixed_scoring_test3.md",
)

with open(results_path, "w", encoding="utf-8") as f:
    f.write(content)

print("\nGotovo.")
print("Results:", results_path)
print("Predictions:", prediction_path)
print("Errors:", errors_path)
print("Confusion matrix:", cm_path)
