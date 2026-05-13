import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# =====================================================
# POSTAVKE
# =====================================================

OWN_TRAIN_FILE = "train_3.csv"
COMBINED_TRAIN_FILE = "TRAIN.csv"

TEST_FILES = {
    "Test 1: group 1": "test_1.xlsx",
    "Test 2: group 2": "test_2.xlsx",
    "Test 3: group 3 (OURS)": "test_3.csv",
    "Test 4: group 4": "test_4.tsv",
}

VALID_LABELS = ["positive", "negative", "neutral", "mixed", "sarcasm"]

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# =====================================================
# FUNKCIJE
# =====================================================

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
        df = pd.read_csv(
            path,
            sep=";",
            encoding="utf-8-sig",
            engine="python"
        )

    else:
        raise ValueError(f"Nepoznat format datoteke: {path}")

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

    return df


def train_and_evaluate(model_name, model, train_df, test_datasets, train_label):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000
    )

    X_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label"]

    model.fit(X_train, y_train)

    results = {}

    for test_name, test_df in test_datasets.items():
        X_test = vectorizer.transform(test_df["text"])
        y_test = test_df["label"]

        y_pred = model.predict(X_test)

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        accuracy = accuracy_score(y_test, y_pred)

        results[test_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }

    safe_model_name = model_name.lower().replace(" ", "_")
    safe_train_label = train_label.lower().replace("-", "_")

    joblib.dump(
        model,
        os.path.join(MODELS_DIR, f"{safe_model_name}_{safe_train_label}_model.joblib")
    )

    joblib.dump(
        vectorizer,
        os.path.join(MODELS_DIR, f"{safe_model_name}_{safe_train_label}_tfidf_vectorizer.joblib")
    )

    return results


def format_scores(scores):
    return (
        f"P: {scores['precision']:.3f}, "
        f"R: {scores['recall']:.3f}, "
        f"F1: {scores['f1']:.3f}, "
        f"Acc: {scores['accuracy']:.3f}"
    )


# =====================================================
# UČITAVANJE PODATAKA
# =====================================================

own_train = clean_dataset(read_dataset(OWN_TRAIN_FILE))
combined_train = clean_dataset(read_dataset(COMBINED_TRAIN_FILE))

test_datasets = {}

for test_name, test_file in TEST_FILES.items():
    test_datasets[test_name] = clean_dataset(read_dataset(test_file))

print("Train-3 rows:", len(own_train))
print("TRAIN rows:", len(combined_train))

for name, dataset in test_datasets.items():
    print(name, "rows:", len(dataset))


# =====================================================
# MODELI
# =====================================================

experiments = [
    {
        "row_id": "1.a.i",
        "method": "Machine learning",
        "algorithm": "Logistic Regression",
        "train_name": "Train-3",
        "train_df": own_train,
        "model": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    },
    {
        "row_id": "1.a.ii",
        "method": "Machine learning",
        "algorithm": "Logistic Regression",
        "train_name": "TRAIN",
        "train_df": combined_train,
        "model": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    },
    {
        "row_id": "1.b.i",
        "method": "Machine learning",
        "algorithm": "Multinomial Naive Bayes",
        "train_name": "Train-3",
        "train_df": own_train,
        "model": MultinomialNB(alpha=1.0)
    },
    {
        "row_id": "1.b.ii",
        "method": "Machine learning",
        "algorithm": "Multinomial Naive Bayes",
        "train_name": "TRAIN",
        "train_df": combined_train,
        "model": MultinomialNB(alpha=1.0)
    }
]


# =====================================================
# TRENIRANJE I EVALUACIJA
# =====================================================

result_rows = []

for experiment in experiments:

    print()
    print("Running:", experiment["row_id"], experiment["algorithm"], experiment["train_name"])

    scores = train_and_evaluate(
        model_name=experiment["algorithm"],
        model=experiment["model"],
        train_df=experiment["train_df"],
        test_datasets=test_datasets,
        train_label=experiment["train_name"]
    )

    row = {
        "#": experiment["row_id"],
        "method": experiment["method"],
        "algorithm": experiment["algorithm"],
        "train": experiment["train_name"],
    }

    for test_name in TEST_FILES.keys():
        row[test_name] = format_scores(scores[test_name])

    result_rows.append(row)


# =====================================================
# RESULTS.MD
# =====================================================

results_df = pd.DataFrame(result_rows)

markdown_table = results_df.to_markdown(index=False)

content = f"""# Machine Learning Results

## Task: Implementation 1 - Machine Learning

The following sentiment labels were used:

- positive
- negative
- neutral
- mixed
- sarcasm

Text representation: TF-IDF  
Evaluation metrics: weighted precision, weighted recall, weighted F1-score, accuracy.

## Results

{markdown_table}

## Hyperparameters

### TF-IDF Vectorizer

- lowercase: True
- ngram_range: (1, 2)
- min_df: 2
- max_features: 50000

### Logistic Regression

- max_iter: 1000
- class_weight: balanced
- random_state=42

### Multinomial Naive Bayes

- alpha: 1.0

## Saved Models

The trained models and TF-IDF vectorizers were saved in the `models/` folder using joblib.
"""

with open("results.md", "w", encoding="utf-8") as f:
    f.write(content)

results_df.to_csv(
    "ml_results.csv",
    index=False,
    encoding="utf-8-sig",
    sep=";"
)

print()
print("Gotovo.")
print("Napravljen je results.md")
print("Napravljen je ml_results.csv")
print("Modeli i TF-IDF vectorizeri spremljeni su u folder models/")
