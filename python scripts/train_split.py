import pandas as pd
import random
import os

INPUT_FILE = "final_corpus_clean.csv"
OUTPUT_FOLDER = "split_1"

TEST_RATIO = 0.20
VALIDATION_RATIO = 0.10

RANDOM_SEED = 42

CSV_SEPARATOR = ";"

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig",
    sep=CSV_SEPARATOR,
    engine="python"
)


df["groupid"] = df["groupid"].astype("Int64")
df["review_id"] = df["review_id"].astype("Int64")
df["sentence_id"] = df["sentence_id"].astype("Int64")

print("Ukupno rečenica:", len(df))
print("Stupci:", df.columns.tolist())


df["review_id"] = df["review_id"].astype("Int64")
df["sentence_id"] = df["sentence_id"].astype("Int64")


review_counts = (
    df.groupby("review_id")
    .size()
    .reset_index(name="sentence_count")
)

review_list = list(
    zip(review_counts["review_id"], review_counts["sentence_count"])
)

random.seed(RANDOM_SEED)
random.shuffle(review_list)

total_sentences = len(df)

target_test = round(total_sentences * TEST_RATIO)
target_validation = round(total_sentences * VALIDATION_RATIO)

print("Ciljani test:", target_test)
print("Ciljani validation:", target_validation)


def choose_reviews(review_list, target_count):
    selected = []
    current_count = 0

    for review_id, sentence_count in review_list:
        if current_count >= target_count:
            break

        selected.append(review_id)
        current_count += sentence_count

    return selected, current_count


test_ids, test_count = choose_reviews(
    review_list,
    target_test
)

remaining_reviews = [
    item for item in review_list
    if item[0] not in test_ids
]


validation_ids, validation_count = choose_reviews(
    remaining_reviews,
    target_validation
)

test_ids = set(test_ids)
validation_ids = set(validation_ids)


test_df = df[df["review_id"].isin(test_ids)]
validation_df = df[df["review_id"].isin(validation_ids)]
train_df = df[
    ~df["review_id"].isin(test_ids)
    & ~df["review_id"].isin(validation_ids)
]


print()
print("Train rečenica:", len(train_df))
print("Validation rečenica:", len(validation_df))
print("Test rečenica:", len(test_df))

print("Train %:", round(len(train_df) / total_sentences * 100, 2))
print("Validation %:", round(len(validation_df) / total_sentences * 100, 2))
print("Test %:", round(len(test_df) / total_sentences * 100, 2))

print()
print("Ukupno nakon splita:", len(train_df) + len(validation_df) + len(test_df))


train_reviews = set(train_df["review_id"])
validation_reviews = set(validation_df["review_id"])
test_reviews = set(test_df["review_id"])

overlap_train_val = train_reviews & validation_reviews
overlap_train_test = train_reviews & test_reviews
overlap_val_test = validation_reviews & test_reviews

print("Overlap train-validation:", len(overlap_train_val))
print("Overlap train-test:", len(overlap_train_test))
print("Overlap validation-test:", len(overlap_val_test))


os.makedirs(OUTPUT_FOLDER, exist_ok=True)

train_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "train_1.csv"),
    index=False,
    encoding="utf-8-sig",
    sep=CSV_SEPARATOR
)

validation_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "validation_1.csv"),
    index=False,
    encoding="utf-8-sig",
    sep=CSV_SEPARATOR
)

test_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "test_1.csv"),
    index=False,
    encoding="utf-8-sig",
    sep=CSV_SEPARATOR
)

print()
print("Gotovo.")
print("Datoteke su spremljene u folder:", OUTPUT_FOLDER)
