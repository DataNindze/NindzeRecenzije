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

## Confusion Matrices
Test 1
|               | pred_positive | pred_negative | pred_neutral | pred_mixed | pred_sarcasm |
|---------------|--------------:|--------------:|-------------:|-----------:|-------------:|
| true_positive | 216 | 12 | 4  | 0 | 0 |
| true_negative | 2   | 47 | 2  | 0 | 0 |
| true_neutral  | 13  | 49 | 44 | 0 | 0 |
| true_mixed    | 0   | 1  | 0  | 1 | 0 |
| true_sarcasm  | 0   | 0  | 0  | 0 | 0 |

Test 2

|               | pred_positive | pred_negative | pred_neutral | pred_mixed | pred_sarcasm |
|---------------|--------------:|--------------:|-------------:|-----------:|-------------:|
| true_positive | 122 | 9   | 4  | 0 | 0 |
| true_negative | 13  | 326 | 30 | 0 | 0 |
| true_neutral  | 17  | 26  | 71 | 0 | 0 |
| true_mixed    | 10  | 8   | 7  | 0 | 0 |
| true_sarcasm  | 3   | 2   | 1  | 0 | 0 |

Test 3

|               | pred_positive | pred_negative | pred_neutral | pred_mixed | pred_sarcasm |
|---------------|--------------:|--------------:|-------------:|-----------:|-------------:|
| true_positive | 350 | 13 | 8  | 0 | 0 |
| true_negative | 12  | 125 | 13 | 0 | 0 |
| true_neutral  | 9   | 16  | 47 | 0 | 0 |
| true_mixed    | 4   | 9   | 1  | 2 | 0 |
| true_sarcasm  | 1   | 0   | 0  | 0 | 0 |

Test 4

|               | pred_positive | pred_negative | pred_neutral | pred_mixed | pred_sarcasm |
|---------------|--------------:|--------------:|-------------:|-----------:|-------------:|
| true_positive | 135 | 8  | 7  | 1 | 0 |
| true_negative | 5   | 83 | 5  | 0 | 0 |
| true_neutral  | 3   | 8  | 17 | 0 | 0 |
| true_mixed    | 1   | 1  | 0  | 1 | 0 |
| true_sarcasm  | 0   | 1  | 0  | 0 | 0 |

---

### Short Interpretation - comparison with BERTić

When comparing the two approaches for Croatian sentiment analysis, BERTić (Transformer) demonstrates overall superior performance compared to Gemma 3 1B (Instruction Fine-Tuning).  

# Key Performance Comparison

Test 1: BERTić wins significantly with an F1-score of 0.826 and Accuracy of 0.824, compared to Gemma's F1: 0.787 and Acc: 0.788.  
Test 2: BERTić wins with an F1-score of 0.798 and Accuracy of 0.817, beating Gemma's F1: 0.780 and Acc: 0.800.  
Test 3: BERTić wins with an F1-score of 0.866 and Accuracy of 0.877, outperforming Gemma's F1: 0.851 and Acc: 0.859.  
Test 4: Both models perform comparably/almost identical, with Gemma showing a microscopic edge in F1-score (0.854 vs. 0.853) and BERTić slightly leading in Accuracy (0.855 vs 0.851).  

# Conclusion

BERTić is the preferred model. As a dedicated regional encoder-based transformer, it consistently manages the nuances of the Croatian language better across the board, whereas the smaller LLM (Gemma 3 1B), despite the advanced fixed-label scoring method, falls slightly short in generalization across the different test groups. 

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
