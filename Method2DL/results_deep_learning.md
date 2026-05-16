# Deep Learning Results

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

Validation set:

- VALIDATION = validation-1 + validation-2 + validation-3 + validation-4
- Validation loss was computed after every epoch.
- Early stopping was applied based on validation loss.

Evaluation metrics:

- weighted precision
- weighted recall
- weighted F1-score
- accuracy

## Results

| #   | method                | algorithm   | train   | Test 1: group 1                           | Test 2: group 2                           | Test 3: group 3 (OURS)                    | Test 4: group 4                           |
|:----|:----------------------|:------------|:--------|:------------------------------------------|:------------------------------------------|:------------------------------------------|:------------------------------------------|
| 2.a | Shallow Deep Learning | CNN         | TRAIN   | P: 0.712, R: 0.509, F1: 0.588, Acc: 0.509 | P: 0.594, R: 0.388, F1: 0.441, Acc: 0.388 | P: 0.709, R: 0.584, F1: 0.632, Acc: 0.584 | P: 0.752, R: 0.535, F1: 0.609, Acc: 0.535 |
| 2.b | Shallow Deep Learning | GRU         | TRAIN   | P: 0.714, R: 0.568, F1: 0.621, Acc: 0.568 | P: 0.624, R: 0.550, F1: 0.579, Acc: 0.550 | P: 0.738, R: 0.646, F1: 0.682, Acc: 0.646 | P: 0.746, R: 0.556, F1: 0.624, Acc: 0.556 |

## Hyperparameters

### General

- random_seed: 42
- max_len: 60
- batch_size: 32
- epochs: 30
- early_stopping_patience: 3
- min_delta: 0.0001
- learning_rate: 0.0005
- embedding_dim: 300
- min_freq: 2
- dropout: 0.6
- freeze_embeddings: True
- class_weights: True

### Embeddings

- embedding source: Croatian FastText
- embedding file: `cc.hr.300.vec`
- embeddings fine-tuned during training: False

### CNN

- filter sizes: 3, 4, 5
- number of filters per size: 64
- dropout: 0.6
- optimizer: Adam

### GRU

- hidden size: 64
- bidirectional: True
- dropout: 0.6
- optimizer: Adam

## Training History

Training and validation losses were saved in:

- `dl_models/cnn_training_history.csv`
- `dl_models/gru_training_history.csv`

## Confusion Matrices

Confusion matrices were saved in the `confusion_matrices/` folder.

## Saved Models

Trained models were saved in the `dl_models/` folder.
