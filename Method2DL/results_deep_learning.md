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

Evaluation metrics:

- weighted precision
- weighted recall
- weighted F1-score
- accuracy

## Results

| #   | method                | algorithm   | train   | Test 1: group 1                           | Test 2: group 2                           | Test 3: group 3 (OURS)                    | Test 4: group 4                           |
|:----|:----------------------|:------------|:--------|:------------------------------------------|:------------------------------------------|:------------------------------------------|:------------------------------------------|
| 2.a | Shallow Deep Learning | CNN         | TRAIN   | P: 0.625, R: 0.624, F1: 0.551, Acc: 0.624 | P: 0.617, R: 0.413, F1: 0.398, Acc: 0.413 | P: 0.607, R: 0.646, F1: 0.585, Acc: 0.646 | P: 0.619, R: 0.622, F1: 0.577, Acc: 0.622 |
| 2.b | Shallow Deep Learning | GRU         | TRAIN   | P: 0.648, R: 0.655, F1: 0.607, Acc: 0.655 | P: 0.616, R: 0.476, F1: 0.471, Acc: 0.476 | P: 0.651, R: 0.670, F1: 0.633, Acc: 0.670 | P: 0.646, R: 0.644, F1: 0.606, Acc: 0.644 |

## Hyperparameters

### General

- random_seed: 42
- max_len: 60
- batch_size: 32
- epochs: 8
- learning_rate: 0.001
- embedding_dim: 300

### Embeddings

- embedding source: Croatian FastText
- embedding file: `cc.hr.300.vec`
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
