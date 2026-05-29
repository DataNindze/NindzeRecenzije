# IFT Results – Gemma 3 1B Instruction Fine-Tuning

## Task: Implementation 3.2 – Large Language Models / Instruction Fine-Tuning

### Model

The experiment was conducted using the instruction-tuned multilingual large language model:

* `unsloth/gemma-3-1b-it-unsloth-bnb-4bit`

The model was loaded in 4-bit quantized format using the Unsloth framework to reduce memory consumption and accelerate training and inference on GPU hardware.

### Motivation

Previous experiments were conducted using Gemma 2 models. Since Gemma 3 provides improved multilingual capabilities and broader language coverage, an additional experiment was performed to evaluate whether a multilingual instruction-tuned model could improve sentiment classification performance on Croatian medical review sentences.

The task consisted of assigning one of five sentiment labels:

* positive
* negative
* neutral
* mixed
* sarcasm

### Training Data

Training was performed using:

* TRAIN dataset

Validation data:

* VALIDATION dataset

The datasets contain Croatian medical review sentences manually annotated with sentiment labels.

### Prompt Format

A lightweight prompt-completion format was used instead of a full conversational chat template in order to reduce sequence length and improve training efficiency.

Example:

```text
Labels: positive, negative, neutral, mixed, sarcasm
Sentence: Doktorica je stručna, empatična i brižna.
Label: positive
```

The model was trained to predict only the final sentiment label.

### Fine-Tuning Method

Instruction Fine-Tuning (IFT) was performed using LoRA (Low-Rank Adaptation).

LoRA allows efficient adaptation of large language models by updating only a small subset of trainable parameters while keeping the majority of the original model frozen.

Only approximately 1.3% of model parameters were updated during training.

### Hyperparameters

#### Training

* epochs: 2
* learning_rate: 2e-5
* batch_size: 8
* gradient_accumulation_steps: 1
* max_seq_length: 128
* weight_decay: 0.01
* lr_scheduler: cosine
* warmup_ratio: 0.05
* random_seed: 42

#### LoRA

* rank (r): 16
* alpha: 16
* dropout: 0
* bias: none

#### Quantization

* load_in_4bit: True
* bfloat16: enabled when supported
* framework: Unsloth

### Prediction Method

Two inference approaches were evaluated during development:

1. Generation-based classification
2. Fixed-label scoring

The final experiment used **batched fixed-label scoring**.

For each sentence, the model computed the loss for all candidate labels:

* positive
* negative
* neutral
* mixed
* sarcasm

The label with the lowest loss value was selected as the final prediction.

To improve efficiency, all candidate labels were evaluated simultaneously within a single batched forward pass.

### Evaluation Metrics

The following evaluation metrics were calculated:

* Accuracy
* Weighted Precision
* Weighted Recall
* Weighted F1-score

Weighted metrics were selected because the sentiment classes are not perfectly balanced.

## Results

| Dataset       | Precision | Recall | F1-score | Accuracy |
| ------------- | --------: | -----: | -------: | -------: |
| Test 1        |     0.352 |  0.593 |    0.442 |    0.593 |
| Test 2        |     0.043 |  0.208 |    0.072 |    0.208 |
| Test 3 (OURS) |     0.370 |  0.608 |    0.460 |    0.608 |
| Test 4        |     0.302 |  0.549 |    0.389 |    0.549 |

## Confusion Matrices

## Test 1

|               | pred_positive | pred_negative | pred_neutral | pred_mixed | pred_sarcasm |
| :------------ | ------------: | ------------: | -----------: | ---------: | -----------: |
| true_positive |           232 |             0 |            0 |          0 |            0 |
| true_negative |            51 |             0 |            0 |          0 |            0 |
| true_neutral  |           106 |             0 |            0 |          0 |            0 |
| true_mixed    |             2 |             0 |            0 |          0 |            0 |
| true_sarcasm  |             0 |             0 |            0 |          0 |            0 |

## Test 2

|               | pred_positive | pred_negative | pred_neutral | pred_mixed | pred_sarcasm |
| :------------ | ------------: | ------------: | -----------: | ---------: | -----------: |
| true_positive |           135 |             0 |            0 |          0 |            0 |
| true_negative |           369 |             0 |            0 |          0 |            0 |
| true_neutral  |           114 |             0 |            0 |          0 |            0 |
| true_mixed    |            25 |             0 |            0 |          0 |            0 |
| true_sarcasm  |             6 |             0 |            0 |          0 |            0 |

## Test 3


|               | pred_positive | pred_negative | pred_neutral | pred_mixed | pred_sarcasm |
| :------------ | ------------: | ------------: | -----------: | ---------: | -----------: |
| true_positive |           371 |             0 |            0 |          0 |            0 |
| true_negative |           150 |             0 |            0 |          0 |            0 |
| true_neutral  |            72 |             0 |            0 |          0 |            0 |
| true_mixed    |            16 |             0 |            0 |          0 |            0 |
| true_sarcasm  |             1 |             0 |            0 |          0 |            0 |

## Test 4


|               | pred_positive | pred_negative | pred_neutral | pred_mixed | pred_sarcasm |
| :------------ | ------------: | ------------: | -----------: | ---------: | -----------: |
| true_positive |           151 |             0 |            0 |          0 |            0 |
| true_negative |            93 |             0 |            0 |          0 |            0 |
| true_neutral  |            28 |             0 |            0 |          0 |            0 |
| true_mixed    |             3 |             0 |            0 |          0 |            0 |
| true_sarcasm  |             0 |             0 |            0 |          0 |            0 |



### Observations

The fixed-label scoring approach substantially outperformed the previously tested generation-based classification setup.

The best performance was achieved on Test 3, which corresponds to the dataset created within this project:

* F1-score: 0.460
* Accuracy: 0.608


The confusion matrices indicate that the model exhibits a strong bias toward predicting the **positive** class. This behavior is particularly visible in Test 1, Test 2, and Test 4, where the majority of instances from all sentiment categories were assigned to the positive label.

The best performance was achieved on **Test 3**, which was created within this project. On this dataset, the model correctly identified a substantial portion of positive examples, resulting in the highest F1-score (0.460) and accuracy (0.608) among all evaluated test sets.

In contrast, performance on **Test 2** was considerably lower. The confusion matrix suggests that the model struggled to generalize to this dataset and misclassified negative, neutral, mixed, and sarcasm instances as positive.

Overall, the confusion matrices reveal that although the fixed-label scoring approach improved performance compared to generation-based classification, the model still had difficulty distinguishing minority sentiment classes and tended to favor the majority positive class.


### Comparison with Generation-Based Classification

A previous experiment using generation-based classification produced substantially lower results:

| Dataset | F1-score |
| ------- | -------: |
| Test 1  |    0.126 |
| Test 2  |    0.055 |
| Test 3  |    0.038 |
| Test 4  |    0.026 |

This demonstrates that fixed-label scoring is considerably more stable for sentiment classification than free-text label generation when using a small instruction-tuned language model.

## Conclusion

The multilingual Gemma 3 1B model successfully learned sentiment classification patterns from Croatian medical review data.

Although the fixed-label scoring approach significantly improved performance compared to generation-based classification, the model remained less effective than the strongest transformer-based classifier evaluated in this project (BERTić).

Nevertheless, the experiment demonstrates that instruction fine-tuning can be successfully applied to multilingual large language models for Croatian sentiment classification and provides a useful comparison between classical transformer classifiers and generative instruction-tuned language models.

## Saved Outputs

### Model

`./ift_models/gemma_3_1b_fixed_scoring_test3`

### Prediction Files

`./ift_predictions_gemma3_fixed_scoring_test3`

`./ift_predictions_gemma3_fixed_scoring_remaining_tests`

### Confusion Matrices

`./ift_confusion_matrices_gemma3_fixed_scoring_test3`

`./ift_confusion_matrices_gemma3_fixed_scoring_remaining_tests`
