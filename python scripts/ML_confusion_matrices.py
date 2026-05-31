import os
import joblib
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report

from machine_learning_results import read_dataset, clean_dataset, TEST_FILES, VALID_LABELS

MODELS_DIR = "models"
OUTPUT_FILE = "confusion_matrices.md"

SAVED_MODELS = {
    "logistic_regression_train_3": (
        "logistic_regression_train_3_model.joblib",
        "logistic_regression_train_3_tfidf_vectorizer.joblib"
    ),
    "logistic_regression_train": (
        "logistic_regression_train_model.joblib",
        "logistic_regression_train_tfidf_vectorizer.joblib"
    ),
    "multinomial_naive_bayes_train_3": (
        "multinomial_naive_bayes_train_3_model.joblib",
        "multinomial_naive_bayes_train_3_tfidf_vectorizer.joblib"
    ),
    "multinomial_naive_bayes_train": (
        "multinomial_naive_bayes_train_model.joblib",
        "multinomial_naive_bayes_train_tfidf_vectorizer.joblib"
    ),
}

md_content = "# Confusion Matrices\n\n"

for model_label, (model_file, vectorizer_file) in SAVED_MODELS.items():

    model_path = os.path.join(MODELS_DIR, model_file)
    vectorizer_path = os.path.join(MODELS_DIR, vectorizer_file)

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    md_content += f"## Model: {model_label}\n\n"

    for test_name, test_file in TEST_FILES.items():

        df = clean_dataset(read_dataset(test_file))

        X_test = vectorizer.transform(df["text"])
        y_true = df["label"]
        y_pred = model.predict(X_test)

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=VALID_LABELS
        )

        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{label}" for label in VALID_LABELS],
            columns=[f"pred_{label}" for label in VALID_LABELS]
        )

        md_content += f"### {test_name}\n\n"
        md_content += "#### Confusion matrix\n\n"
        md_content += cm_df.to_markdown()
        md_content += "\n\n"

        md_content += "#### Classification report\n\n"
        md_content += "```text\n"
        md_content += classification_report(
            y_true,
            y_pred,
            labels=VALID_LABELS,
            zero_division=0
        )
        md_content += "\n```\n\n"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"Gotovo. Sve confusion matrice spremljene su u {OUTPUT_FILE}")
