# Transformer Results - multilingual BERT

## Task: Implementation 3.1 - Large Language Models / Transformers

Model:

- `bert-base-multilingual-cased`

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

| #   | method       | algorithm         | train   | Test 1: group 1                           | Test 2: group 2                           | Test 3: group 3 (OURS)                    | Test 4: group 4                           |
|:----|:-------------|:------------------|:--------|:------------------------------------------|:------------------------------------------|:------------------------------------------|:------------------------------------------|
| 3.a | Transformers | multilingual BERT | TRAIN   | P: 0.784, R: 0.696, F1: 0.680, Acc: 0.696 | P: 0.636, R: 0.647, F1: 0.621, Acc: 0.647 | P: 0.786, R: 0.816, F1: 0.795, Acc: 0.816 | P: 0.766, R: 0.782, F1: 0.773, Acc: 0.782 |

The multilingual BERT model outperformed the previously implemented machine learning and shallow deep learning baselines. The best result was achieved on Test 3, with a weighted F1-score of 0.795 and accuracy of 0.816. The model also performed strongly on Test 4, while Test 2 remained the most challenging test set, similarly to previous experiments.

## Confusion Matrices


### Test 1: group 1

|               |   pred_mixed |   pred_negative |   pred_neutral |   pred_positive |   pred_sarcasm |
|:--------------|-------------:|----------------:|---------------:|----------------:|---------------:|
| true_mixed    |            0 |               0 |              0 |               2 |              0 |
| true_negative |            0 |              41 |              1 |               9 |              0 |
| true_neutral  |            0 |              62 |             24 |              20 |              0 |
| true_positive |            0 |              21 |              4 |             207 |              0 |
| true_sarcasm  |            0 |               0 |              0 |               0 |              0 |



### Test 2: group 2

|               |   pred_mixed |   pred_negative |   pred_neutral |   pred_positive |   pred_sarcasm |
|:--------------|-------------:|----------------:|---------------:|----------------:|---------------:|
| true_mixed    |            0 |              12 |              2 |              11 |              0 |
| true_negative |            0 |             269 |             22 |              78 |              0 |
| true_neutral  |            0 |              50 |             29 |              35 |              0 |
| true_positive |            0 |               9 |              4 |             122 |              0 |
| true_sarcasm  |            0 |               4 |              0 |               2 |              0 |



### Test 3: group 3 (OURS)

|               |   pred_mixed |   pred_negative |   pred_neutral |   pred_positive |   pred_sarcasm |
|:--------------|-------------:|----------------:|---------------:|----------------:|---------------:|
| true_mixed    |            0 |               8 |              0 |               8 |              0 |
| true_negative |            0 |             118 |              8 |              24 |              0 |
| true_neutral  |            0 |              20 |             28 |              24 |              0 |
| true_positive |            0 |              15 |              4 |             352 |              0 |
| true_sarcasm  |            0 |               0 |              0 |               1 |              0 |



### Test 4: group 4

|               |   pred_mixed |   pred_negative |   pred_neutral |   pred_positive |   pred_sarcasm |
|:--------------|-------------:|----------------:|---------------:|----------------:|---------------:|
| true_mixed    |            0 |               1 |              0 |               2 |              0 |
| true_negative |            0 |              72 |              4 |              17 |              0 |
| true_neutral  |            0 |              11 |             11 |               6 |              0 |
| true_positive |            0 |              12 |              7 |             132 |              0 |
| true_sarcasm  |            0 |               0 |              0 |               0 |              0 |



## Hyperparameters

- max_length: 128
- batch_size: 8
- epochs: 4
- learning_rate: 2e-05
- weight_decay: 0.01
- random_seed: 42
- evaluation_strategy: epoch
- save_strategy: epoch
- early_stopping_patience: 2
- best_model_metric: eval_loss

## Outputs

Model saved to:

- `/content/drive/MyDrive/Obrada_prirodnog_jezika (1)/transformer_models/multilingual_bert_sentiment`

Prediction files saved to:

- `transformer_predictions/`

Confusion matrices saved to:

- `transformer_confusion_matrices/`
