import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

df = pd.read_excel("opj zadnje.xlsx")

labels = ["positive", "neutral", "negative", "mixed", "sarcasm"]

# tvoji stupci
annotator_cols = [
    "label_member1",
    "label_member2",
    "label_member3",
    "label_member4"
]

df = df.dropna(subset=annotator_cols)

matrix = []

for _, row in df.iterrows():

    counts = []

    for label in labels:

        count = 0

        for col in annotator_cols:

            if str(row[col]).strip().lower() == label:
                count += 1

        counts.append(count)

    matrix.append(counts)

kappa = fleiss_kappa(matrix)

print("Broj rečenica:", len(matrix))
print("Fleiss' kappa:", round(kappa, 3))
