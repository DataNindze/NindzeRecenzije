# Machine Learning Results

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

| #      | method           | algorithm               | train   | Test 1: group 1                           | Test 2: group 2                           | Test 3: group 3 (OURS)                    | Test 4: group 4                           |
|:-------|:-----------------|:------------------------|:--------|:------------------------------------------|:------------------------------------------|:------------------------------------------|:------------------------------------------|
| 1.a.i  | Machine learning | Logistic Regression     | Train-3 | P: 0.685, R: 0.593, F1: 0.616, Acc: 0.593 | P: 0.603, R: 0.610, F1: 0.603, Acc: 0.610 | P: 0.740, R: 0.725, F1: 0.730, Acc: 0.725 | P: 0.708, R: 0.662, F1: 0.680, Acc: 0.662 |
| 1.a.ii | Machine learning | Logistic Regression     | TRAIN   | P: 0.765, R: 0.668, F1: 0.692, Acc: 0.668 | P: 0.664, R: 0.650, F1: 0.656, Acc: 0.650 | P: 0.761, R: 0.720, F1: 0.736, Acc: 0.720 | P: 0.787, R: 0.698, F1: 0.730, Acc: 0.698 |
| 1.b.i  | Machine learning | Multinomial Naive Bayes | Train-3 | P: 0.444, R: 0.606, F1: 0.513, Acc: 0.606 | P: 0.481, R: 0.507, F1: 0.461, Acc: 0.507 | P: 0.615, R: 0.725, F1: 0.659, Acc: 0.725 | P: 0.612, R: 0.676, F1: 0.623, Acc: 0.676 |
| 1.b.ii | Machine learning | Multinomial Naive Bayes | TRAIN   | P: 0.470, R: 0.647, F1: 0.545, Acc: 0.647 | P: 0.689, R: 0.564, F1: 0.515, Acc: 0.564 | P: 0.746, R: 0.739, F1: 0.680, Acc: 0.739 | P: 0.761, R: 0.738, F1: 0.703, Acc: 0.738 |


The best result was achieved by Logistic Regression trained on the combined TRAIN set, with the highest weighted F1-score of 0.736 on Test 3: group 3 (OURS). This suggests that training on a larger combined dataset improved the model’s performance.

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
