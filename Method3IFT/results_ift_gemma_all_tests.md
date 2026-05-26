# IFT Results - Gemma 2 2B Instruct Fixed Label Scoring

## Task: Implementation 3.2 - Large Language Models / Instruction Fine-Tuning

Model:

- `unsloth/gemma-2-2b-it`

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

| #   | method   | algorithm     | train   | Test 1: group 1                           | Test 2: group 2                           | Test 3: group 3 (OURS)                    | Test 4: group 4                           |
|:----|:---------|:--------------|:--------|:------------------------------------------|:------------------------------------------|:------------------------------------------|:------------------------------------------|
| 3.c | IFT      | Gemma 2 2B IT | TRAIN   | P: 0.352, R: 0.593, F1: 0.442, Acc: 0.593 | P: 0.043, R: 0.208, F1: 0.072, Acc: 0.208 | P: 0.370, R: 0.608, F1: 0.460, Acc: 0.608 | P: 0.302, R: 0.549, F1: 0.389, Acc: 0.549 |

## Confusion Matrices

## Gemma 2 2B IT IFT - Confusion Matrices

### Test 1: group 1

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |             232 |               0 |              0 |            0 |              0 |
| true_negative |              51 |               0 |              0 |            0 |              0 |
| true_neutral  |             106 |               0 |              0 |            0 |              0 |
| true_mixed    |               2 |               0 |              0 |            0 |              0 |
| true_sarcasm  |               0 |               0 |              0 |            0 |              0 |

### Test 2: group 2

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |             135 |               0 |              0 |            0 |              0 |
| true_negative |             369 |               0 |              0 |            0 |              0 |
| true_neutral  |             114 |               0 |              0 |            0 |              0 |
| true_mixed    |              25 |               0 |              0 |            0 |              0 |
| true_sarcasm  |               6 |               0 |              0 |            0 |              0 |

### Test 3: group 3 (OURS)

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |             371 |               0 |              0 |            0 |              0 |
| true_negative |             150 |               0 |              0 |            0 |              0 |
| true_neutral  |              72 |               0 |              0 |            0 |              0 |
| true_mixed    |              16 |               0 |              0 |            0 |              0 |
| true_sarcasm  |               1 |               0 |              0 |            0 |              0 |

### Test 4: group 4

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |             151 |               0 |              0 |            0 |              0 |
| true_negative |              93 |               0 |              0 |            0 |              0 |
| true_neutral  |              28 |               0 |              0 |            0 |              0 |
| true_mixed    |               3 |               0 |              0 |            0 |              0 |
| true_sarcasm  |               0 |               0 |              0 |            0 |              0 |

## Note

The confusion matrices show that the Gemma 2 2B IT IFT model predicted only the `positive` label across all test sets. This indicates a strong majority-class bias. Although the model achieved some accuracy due to the high number of positive examples, it failed to distinguish negative, neutral, mixed, and sarcasm classes. Therefore, its weighted F1-score remained substantially lower than the transformer classification models, especially BERTić.

## Outputs

Prediction files saved to:

- `ift_predictions_chat/`

Confusion matrices saved to:

- `ift_confusion_matrices_chat/`
