# IFT Results – Gemma 3 1B (Advanced Fixed-Label Scoring, All Tests)

## Task: Multi-Dataset Evaluation with Instruction Fine-Tuning

### Model
- `unsloth/gemma-3-1b-it-unsloth-bnb-4bit`

### Datasets
- **Training Set:** TRAIN
- **Validation Set:** VALIDATION

### Prompt Format (Chat Template)

`text
<|im_start|>user
Odredi sentiment sljedeće hrvatske rečenice. Odgovori isključivo jednom od oznaka: positive, negative, neutral, mixed, sarcasm.

Rečenica: ...<|im_end|>
<|im_start|>assistant
...`

### Training Configuration

Training was performed for **5 epochs** with **Early Stopping** enabled (`patience = 3`).

| Parameter | Value |
|------------|--------|
| max_seq_length | 256 |
| batch_size | 16 |
| gradient_accumulation_steps | 8 (effective batch size: 128) |
| learning_rate | 0.0002 |
| scheduler | Cosine |
| warmup_ratio | 0.03 |

### Prediction Method

Batched fixed-label scoring was performed using the official Gemma 3 chat format.

All five candidate labels were evaluated using a causal log-likelihood masked forward pass. The label with the **lowest loss** was selected as the final prediction.

### Labels

- positive
- negative
- neutral
- mixed
- sarcasm

### Evaluation Metrics

- Weighted Precision
- Weighted Recall
- Weighted F1-score
- Accuracy

---

## Summary Results

| # | Method | Algorithm | Training Set | Test Dataset 1 | Test Dataset 2 | Test Dataset 3 | Test Dataset 4 |
|---|---------|-----------|-------------|----------------|----------------|----------------|----------------|
| 3.1 | IFT (Chat Template) | Gemma 3 1B IT | TRAIN | P: 0.855, R: 0.788, F1: 0.787, Acc: 0.788 | — | — | — |
| 3.2 | IFT (Chat Template) | Gemma 3 1B IT | TRAIN | — | P: 0.764, R: 0.800, F1: 0.780, Acc: 0.800 | — | — |
| 3.3 | IFT (Chat Template) | Gemma 3 1B IT | TRAIN | — | — | P: 0.861, R: 0.859, F1: 0.851, Acc: 0.859 | — |
| 3.4 | IFT (Chat Template) | Gemma 3 1B IT | TRAIN | — | — | — | P: 0.855, R: 0.855, F1: 0.854, Acc: 0.855 |

---

## Hyperparameters

| Parameter | Value |
|------------|--------|
| max_seq_length | 256 |
| batch_size | 16 |
| gradient_accumulation_steps | 8 |
| epochs | 5 |
| learning_rate | 0.0002 |
| warmup_ratio | 0.03 |
| lr_scheduler_type | cosine |
| weight_decay | 0.01 |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| load_in_4bit | True |
| random_seed | 42 |

---

## Output Directories

**Models:**

`./ift_models/gemma_3_1b_fixed_scoring_all_tests`

**Predictions:**

`./ift_predictions_gemma3_fixed_scoring_all_tests`

**Confusion Matrices:**

`./ift_confusion_matrices_gemma3_fixed_scoring_all_tests`
