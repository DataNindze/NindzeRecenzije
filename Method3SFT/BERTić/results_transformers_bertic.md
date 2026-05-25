# Transformer Results - BERTić

## Task: Implementation 3.1 - Large Language Models / Transformers

Model:

- `classla/bcms-bertic`

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

| #   | method       | algorithm   | train   | Test 1: group 1                           | Test 2: group 2                           | Test 3: group 3 (OURS)                    | Test 4: group 4                           |
|:----|:-------------|:------------|:--------|:------------------------------------------|:------------------------------------------|:------------------------------------------|:------------------------------------------|
| 3.b | Transformers | BERTić      | TRAIN   | P: 0.850, R: 0.824, F1: 0.826, Acc: 0.824 | P: 0.781, R: 0.817, F1: 0.798, Acc: 0.817 | P: 0.856, R: 0.877, F1: 0.866, Acc: 0.877 | P: 0.858, R: 0.851, F1: 0.853, Acc: 0.851 |

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

## Confusion Matrices 


Test 1: group 1
  pred_mixed	pred_negative	pred_neutral	pred_positive	pred_sarcasm
true_mixed	0	2	0	0	0
true_negative	0	46	4	1	0
true_neutral	0	34	63	9	0
true_positive	0	7	12	213	0
true_sarcasm	0	0	0	0	0


Test 2: group 2
	pred_mixed	pred_negative	pred_neutral	pred_positive	pred_sarcasm
true_mixed	0	14	6	5	0
true_negative	0	334	30	5	0
true_neutral	0	25	79	10	0
true_positive	1	8	9	117	0
true_sarcasm	0	3	3	0	0

Test 3: group 3 (OURS)
	pred_mixed	pred_negative	pred_neutral	pred_positive	pred_sarcasm
true_mixed	0	11	0	5	0
true_negative	0	130	13	7	0
true_neutral	0	12	54	6	0
true_positive	0	7	13	351	0
true_sarcasm	0	0	0	1	0

Test 4: group 4
	pred_mixed	pred_negative	pred_neutral	pred_positive	pred_sarcasm
true_mixed	0	0	0	3	0
true_negative	0	78	12	3	0
true_neutral	0	9	18	1	0
true_positive	0	5	8	138	0
true_sarcasm	0	0	0	0	0

## Outputs

Model saved to:

- `/content/transformer_models/bertic_sentiment`

Prediction files saved to:

- `transformer_predictions/`

Confusion matrices saved to:

- `transformer_confusion_matrices/`
