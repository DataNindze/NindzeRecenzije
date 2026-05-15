import os
import re
import csv
import pickle
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)


TRAIN_FILE = "TRAIN.csv"

TEST_FILES = {
    "Test 1: group 1": "test_1.xlsx",
    "Test 2: group 2": "test_2.xlsx",
    "Test 3: group 3 (OURS)": "test_3.csv",
    "Test 4: group 4": "test_4.tsv",
}

FASTTEXT_FILE = "cc.hr.300.vec"   # Croatian FastText embeddings

VALID_LABELS = [
    "positive",
    "negative",
    "neutral",
    "mixed",
    "sarcasm"
]

MAX_LEN = 60
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 0.001
EMBEDDING_DIM = 300
MIN_FREQ = 1

RANDOM_SEED = 42

MODELS_DIR = "dl_models"
CONFUSION_DIR = "confusion_matrices"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CONFUSION_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)



def read_dataset(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".xlsx":
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


def tokenize(text):
    text = str(text).lower()
    tokens = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    return tokens


def build_vocab(texts, min_freq=1):
    freq = {}

    for text in texts:
        for token in tokenize(text):
            freq[token] = freq.get(token, 0) + 1

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, count in freq.items():
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab, max_len):
    tokens = tokenize(text)

    ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]

    if len(ids) > max_len:
        ids = ids[:max_len]

    while len(ids) < max_len:
        ids.append(vocab["<PAD>"])

    return ids


# FASTTEXT EMBEDDINGS


def load_fasttext_embeddings(vocab, embedding_file, embedding_dim=300):
    embedding_matrix = np.random.normal(
        scale=0.6,
        size=(len(vocab), embedding_dim)
    ).astype(np.float32)

    embedding_matrix[vocab["<PAD>"]] = np.zeros(embedding_dim)

    if not os.path.exists(embedding_file):
        print("WARNING: FastText file not found.")
        print("Using randomly initialized embeddings.")
        return torch.tensor(embedding_matrix, dtype=torch.float32)

    print("Loading FastText embeddings. This may take a few minutes...")

    found = 0
    vocab_set = set(vocab.keys())

    with open(embedding_file, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

        for line in f:
            parts = line.rstrip().split(" ")

            if len(parts) < embedding_dim + 1:
                continue

            word = parts[0]

            if word in vocab_set:
                vector = np.array(parts[1:], dtype=np.float32)

                if len(vector) == embedding_dim:
                    embedding_matrix[vocab[word]] = vector
                    found += 1

    print("Embeddings found for words:", found, "/", len(vocab))

    return torch.tensor(embedding_matrix, dtype=torch.float32)


# DATASET CLASS


class SentimentDataset(Dataset):
    def __init__(self, df, vocab, label_to_id):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.vocab = vocab
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = encode_text(
            self.texts[idx],
            self.vocab,
            MAX_LEN
        )

        label_id = self.label_to_id[self.labels[idx]]

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label_id, dtype=torch.long)
        )


# MODELS


class CNNTextClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_classes):
        super().__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=False,
            padding_idx=0
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=128,
                kernel_size=k
            )
            for k in [3, 4, 5]
        ])

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(
            128 * 3,
            num_classes
        )

    def forward(self, x):
        embedded = self.embedding(x)

        embedded = embedded.permute(0, 2, 1)

        conv_outputs = [
            torch.relu(conv(embedded))
            for conv in self.convs
        ]

        pooled_outputs = [
            torch.max(conv_output, dim=2).values
            for conv_output in conv_outputs
        ]

        combined = torch.cat(pooled_outputs, dim=1)

        combined = self.dropout(combined)

        return self.fc(combined)


class GRUTextClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_classes):
        super().__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=False,
            padding_idx=0
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(
            128 * 2,
            num_classes
        )

    def forward(self, x):
        embedded = self.embedding(x)

        output, hidden = self.gru(embedded)

        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]

        hidden_combined = torch.cat(
            (hidden_forward, hidden_backward),
            dim=1
        )

        hidden_combined = self.dropout(hidden_combined)

        return self.fc(hidden_combined)


# TRAINING AND EVALUATION


def train_model(model, train_loader):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for batch_texts, batch_labels in train_loader:
            batch_texts = batch_texts.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            outputs = model(batch_texts)

            loss = criterion(outputs, batch_labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    return model


def evaluate_model(model, data_loader, num_classes):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_texts, batch_labels in data_loader:
            batch_texts = batch_texts.to(device)

            outputs = model(batch_texts)

            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(predictions)
            all_labels.extend(batch_labels.numpy())

    precision = precision_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(num_classes))
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "confusion_matrix": cm
    }


def format_scores(scores):
    return (
        f"P: {scores['precision']:.3f}, "
        f"R: {scores['recall']:.3f}, "
        f"F1: {scores['f1']:.3f}, "
        f"Acc: {scores['accuracy']:.3f}"
    )


# MAIN


print("Device:", device)

train_df = clean_dataset(read_dataset(TRAIN_FILE))

test_datasets = {}

for test_name, test_file in TEST_FILES.items():
    test_datasets[test_name] = clean_dataset(read_dataset(test_file))

print("TRAIN rows:", len(train_df))

for name, df_test in test_datasets.items():
    print(name, "rows:", len(df_test))

# label mapping
labels_sorted = sorted(VALID_LABELS)

label_to_id = {
    label: idx
    for idx, label in enumerate(labels_sorted)
}

id_to_label = {
    idx: label
    for label, idx in label_to_id.items()
}

print("Labels:", label_to_id)

# vocab
vocab = build_vocab(
    train_df["text"],
    min_freq=MIN_FREQ
)

print("Vocab size:", len(vocab))

# embeddings
embedding_matrix = load_fasttext_embeddings(
    vocab,
    FASTTEXT_FILE,
    EMBEDDING_DIM
)

# save vocab and labels
with open(os.path.join(MODELS_DIR, "vocab.pkl"), "wb") as f:
    pickle.dump(vocab, f)

with open(os.path.join(MODELS_DIR, "label_to_id.pkl"), "wb") as f:
    pickle.dump(label_to_id, f)

with open(os.path.join(MODELS_DIR, "id_to_label.pkl"), "wb") as f:
    pickle.dump(id_to_label, f)

# data loaders
train_dataset = SentimentDataset(
    train_df,
    vocab,
    label_to_id
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loaders = {}

for test_name, df_test in test_datasets.items():
    test_dataset = SentimentDataset(
        df_test,
        vocab,
        label_to_id
    )

    test_loaders[test_name] = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


# EXPERIMENTS


experiments = [
    {
        "row_id": "2.a",
        "method": "Shallow Deep Learning",
        "algorithm": "CNN",
        "train": "TRAIN",
        "model_class": CNNTextClassifier
    },
    {
        "row_id": "2.b",
        "method": "Shallow Deep Learning",
        "algorithm": "GRU",
        "train": "TRAIN",
        "model_class": GRUTextClassifier
    }
]

result_rows = []

for experiment in experiments:
    print()
    print("Training:", experiment["algorithm"])

    model = experiment["model_class"](
        embedding_matrix=embedding_matrix,
        num_classes=len(label_to_id)
    )

    model = train_model(
        model,
        train_loader
    )

    model_path = os.path.join(
        MODELS_DIR,
        f"{experiment['algorithm'].lower()}_TRAIN.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    row = {
        "#": experiment["row_id"],
        "method": experiment["method"],
        "algorithm": experiment["algorithm"],
        "train": experiment["train"]
    }

    for test_name, test_loader in test_loaders.items():
        scores = evaluate_model(
            model,
            test_loader,
            num_classes=len(label_to_id)
        )

        row[test_name] = format_scores(scores)

        cm_df = pd.DataFrame(
            scores["confusion_matrix"],
            index=[f"true_{id_to_label[i]}" for i in range(len(id_to_label))],
            columns=[f"pred_{id_to_label[i]}" for i in range(len(id_to_label))]
        )

        safe_test_name = (
            test_name
            .lower()
            .replace(" ", "_")
            .replace(":", "")
            .replace("(", "")
            .replace(")", "")
        )

        cm_path = os.path.join(
            CONFUSION_DIR,
            f"{experiment['algorithm'].lower()}_{safe_test_name}_confusion_matrix.csv"
        )

        cm_df.to_csv(
            cm_path,
            encoding="utf-8-sig",
            sep=";"
        )

    result_rows.append(row)



# RESULTS.MD


results_df = pd.DataFrame(result_rows)

markdown_table = results_df.to_markdown(index=False)

content = f"""# Deep Learning Results

## Task: Implementation 2 - Shallow Deep Learning

The following sentiment labels were used:

- positive
- negative
- neutral
- mixed
- sarcasm

Text representation: Croatian FastText word embeddings if available; otherwise randomly initialized embeddings.

Models:

- CNN
- GRU

Training set:

- TRAIN

Evaluation metrics:

- weighted precision
- weighted recall
- weighted F1-score
- accuracy

## Results

{markdown_table}

## Hyperparameters

### General

- random_seed: {RANDOM_SEED}
- max_len: {MAX_LEN}
- batch_size: {BATCH_SIZE}
- epochs: {EPOCHS}
- learning_rate: {LEARNING_RATE}
- embedding_dim: {EMBEDDING_DIM}

### Embeddings

- embedding source: Croatian FastText
- embedding file: `{FASTTEXT_FILE}`
- embeddings fine-tuned during training: True

### CNN

- filter sizes: 3, 4, 5
- number of filters per size: 128
- dropout: 0.5
- optimizer: Adam

### GRU

- hidden size: 128
- bidirectional: True
- dropout: 0.5
- optimizer: Adam

## Confusion Matrices

Confusion matrices were saved in the `confusion_matrices/` folder.

## Saved Models

Trained models were saved in the `dl_models/` folder.
"""

with open("results_deep_learning.md", "w", encoding="utf-8") as f:
    f.write(content)

results_df.to_csv(
    "deep_learning_results.csv",
    index=False,
    encoding="utf-8-sig",
    sep=";"
)

print()
print("Gotovo.")
print("Napravljen je results_deep_learning.md")
print("Napravljen je deep_learning_results.csv")
print("Modeli su spremljeni u folder dl_models/")
print("Confusion matrices su spremljene u folder confusion_matrices/")
