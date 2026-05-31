# Confusion Matrices

## Model: logistic_regression_train_3

### Test 1: group 1

#### Confusion matrix

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |             164 |              40 |             26 |            1 |              1 |
| true_negative |              10 |              34 |              6 |            1 |              0 |
| true_neutral  |              15 |              56 |             33 |            2 |              0 |
| true_mixed    |               1 |               0 |              0 |            1 |              0 |
| true_sarcasm  |               0 |               0 |              0 |            0 |              0 |

#### Classification report

```text
              precision    recall  f1-score   support

    positive       0.86      0.71      0.78       232
    negative       0.26      0.67      0.38        51
     neutral       0.51      0.31      0.39       106
       mixed       0.20      0.50      0.29         2
     sarcasm       0.00      0.00      0.00         0

    accuracy                           0.59       391
   macro avg       0.37      0.44      0.36       391
weighted avg       0.68      0.59      0.62       391

```

### Test 2: group 2

#### Confusion matrix

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |              96 |              28 |              9 |            2 |              0 |
| true_negative |              54 |             258 |             52 |            5 |              0 |
| true_neutral  |              17 |              52 |             40 |            4 |              1 |
| true_mixed    |               6 |              12 |              6 |            1 |              0 |
| true_sarcasm  |               0 |               4 |              1 |            0 |              1 |

#### Classification report

```text
              precision    recall  f1-score   support

    positive       0.55      0.71      0.62       135
    negative       0.73      0.70      0.71       369
     neutral       0.37      0.35      0.36       114
       mixed       0.08      0.04      0.05        25
     sarcasm       0.50      0.17      0.25         6

    accuracy                           0.61       649
   macro avg       0.45      0.39      0.40       649
weighted avg       0.60      0.61      0.60       649

```

### Test 3: group 3 (OURS)

#### Confusion matrix

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |             299 |              46 |             22 |            4 |              0 |
| true_negative |              19 |             106 |             19 |            6 |              0 |
| true_neutral  |              16 |              19 |             34 |            3 |              0 |
| true_mixed    |               4 |               7 |              2 |            3 |              0 |
| true_sarcasm  |               1 |               0 |              0 |            0 |              0 |

#### Classification report

```text
              precision    recall  f1-score   support

    positive       0.88      0.81      0.84       371
    negative       0.60      0.71      0.65       150
     neutral       0.44      0.47      0.46        72
       mixed       0.19      0.19      0.19        16
     sarcasm       0.00      0.00      0.00         1

    accuracy                           0.72       610
   macro avg       0.42      0.43      0.43       610
weighted avg       0.74      0.72      0.73       610

```

### Test 4: group 4

#### Confusion matrix

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |             107 |              19 |             24 |            1 |              0 |
| true_negative |              15 |              63 |             12 |            3 |              0 |
| true_neutral  |               9 |               7 |             12 |            0 |              0 |
| true_mixed    |               0 |               2 |              1 |            0 |              0 |
| true_sarcasm  |               0 |               0 |              0 |            0 |              0 |

#### Classification report

```text
              precision    recall  f1-score   support

    positive       0.82      0.71      0.76       151
    negative       0.69      0.68      0.68        93
     neutral       0.24      0.43      0.31        28
       mixed       0.00      0.00      0.00         3
     sarcasm       0.00      0.00      0.00         0

    accuracy                           0.66       275
   macro avg       0.35      0.36      0.35       275
weighted avg       0.71      0.66      0.68       275

```

## Model: logistic_regression_train

### Test 1: group 1

#### Confusion matrix

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |             172 |              35 |             21 |            4 |              0 |
| true_negative |               6 |              40 |              1 |            4 |              0 |
| true_neutral  |              13 |              39 |             47 |            7 |              0 |
| true_mixed    |               0 |               0 |              0 |            2 |              0 |
| true_sarcasm  |               0 |               0 |              0 |            0 |              0 |

#### Classification report

```text
              precision    recall  f1-score   support

    positive       0.90      0.74      0.81       232
    negative       0.35      0.78      0.48        51
     neutral       0.68      0.44      0.54       106
       mixed       0.12      1.00      0.21         2
     sarcasm       0.00      0.00      0.00         0

    accuracy                           0.67       391
   macro avg       0.41      0.59      0.41       391
weighted avg       0.77      0.67      0.69       391

```

### Test 2: group 2

#### Confusion matrix

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |              93 |              16 |             17 |            9 |              0 |
| true_negative |              30 |             265 |             59 |           15 |              0 |
| true_neutral  |               8 |              47 |             58 |            1 |              0 |
| true_mixed    |               2 |              11 |              6 |            6 |              0 |
| true_sarcasm  |               0 |               4 |              0 |            2 |              0 |

#### Classification report

```text
              precision    recall  f1-score   support

    positive       0.70      0.69      0.69       135
    negative       0.77      0.72      0.74       369
     neutral       0.41      0.51      0.46       114
       mixed       0.18      0.24      0.21        25
     sarcasm       0.00      0.00      0.00         6

    accuracy                           0.65       649
   macro avg       0.41      0.43      0.42       649
weighted avg       0.66      0.65      0.66       649

```

### Test 3: group 3 (OURS)

#### Confusion matrix

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |             297 |              40 |             27 |            7 |              0 |
| true_negative |              12 |             100 |             26 |           12 |              0 |
| true_neutral  |              11 |              20 |             36 |            5 |              0 |
| true_mixed    |               2 |               4 |              3 |            6 |              1 |
| true_sarcasm  |               1 |               0 |              0 |            0 |              0 |

#### Classification report

```text
              precision    recall  f1-score   support

    positive       0.92      0.80      0.86       371
    negative       0.61      0.67      0.64       150
     neutral       0.39      0.50      0.44        72
       mixed       0.20      0.38      0.26        16
     sarcasm       0.00      0.00      0.00         1

    accuracy                           0.72       610
   macro avg       0.42      0.47      0.44       610
weighted avg       0.76      0.72      0.74       610

```

### Test 4: group 4

#### Confusion matrix

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |             110 |              15 |             21 |            5 |              0 |
| true_negative |               5 |              65 |             17 |            6 |              0 |
| true_neutral  |               3 |               9 |             16 |            0 |              0 |
| true_mixed    |               0 |               1 |              1 |            1 |              0 |
| true_sarcasm  |               0 |               0 |              0 |            0 |              0 |

#### Classification report

```text
              precision    recall  f1-score   support

    positive       0.93      0.73      0.82       151
    negative       0.72      0.70      0.71        93
     neutral       0.29      0.57      0.39        28
       mixed       0.08      0.33      0.13         3
     sarcasm       0.00      0.00      0.00         0

    accuracy                           0.70       275
   macro avg       0.41      0.47      0.41       275
weighted avg       0.79      0.70      0.73       275

```

## Model: multinomial_naive_bayes_train_3

### Test 1: group 1

#### Confusion matrix

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |             219 |              13 |              0 |            0 |              0 |
| true_negative |              33 |              18 |              0 |            0 |              0 |
| true_neutral  |              60 |              46 |              0 |            0 |              0 |
| true_mixed    |               2 |               0 |              0 |            0 |              0 |
| true_sarcasm  |               0 |               0 |              0 |            0 |              0 |

#### Classification report

```text
              precision    recall  f1-score   support

    positive       0.70      0.94      0.80       232
    negative       0.23      0.35      0.28        51
     neutral       0.00      0.00      0.00       106
       mixed       0.00      0.00      0.00         2
     sarcasm       0.00      0.00      0.00         0

    accuracy                           0.61       391
   macro avg       0.19      0.26      0.22       391
weighted avg       0.44      0.61      0.51       391

```

### Test 2: group 2

#### Confusion matrix

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |             123 |              12 |              0 |            0 |              0 |
| true_negative |             163 |             206 |              0 |            0 |              0 |
| true_neutral  |              63 |              51 |              0 |            0 |              0 |
| true_mixed    |              13 |              12 |              0 |            0 |              0 |
| true_sarcasm  |               2 |               4 |              0 |            0 |              0 |

#### Classification report

```text
              precision    recall  f1-score   support

    positive       0.34      0.91      0.49       135
    negative       0.72      0.56      0.63       369
     neutral       0.00      0.00      0.00       114
       mixed       0.00      0.00      0.00        25
     sarcasm       0.00      0.00      0.00         6

    accuracy                           0.51       649
   macro avg       0.21      0.29      0.22       649
weighted avg       0.48      0.51      0.46       649

```

### Test 3: group 3 (OURS)

#### Confusion matrix

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |             357 |              14 |              0 |            0 |              0 |
| true_negative |              65 |              85 |              0 |            0 |              0 |
| true_neutral  |              52 |              20 |              0 |            0 |              0 |
| true_mixed    |              10 |               6 |              0 |            0 |              0 |
| true_sarcasm  |               1 |               0 |              0 |            0 |              0 |

#### Classification report

```text
              precision    recall  f1-score   support

    positive       0.74      0.96      0.83       371
    negative       0.68      0.57      0.62       150
     neutral       0.00      0.00      0.00        72
       mixed       0.00      0.00      0.00        16
     sarcasm       0.00      0.00      0.00         1

    accuracy                           0.72       610
   macro avg       0.28      0.31      0.29       610
weighted avg       0.61      0.72      0.66       610

```

### Test 4: group 4

#### Confusion matrix

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |             140 |              11 |              0 |            0 |              0 |
| true_negative |              47 |              46 |              0 |            0 |              0 |
| true_neutral  |              23 |               5 |              0 |            0 |              0 |
| true_mixed    |               3 |               0 |              0 |            0 |              0 |
| true_sarcasm  |               0 |               0 |              0 |            0 |              0 |

#### Classification report

```text
              precision    recall  f1-score   support

    positive       0.66      0.93      0.77       151
    negative       0.74      0.49      0.59        93
     neutral       0.00      0.00      0.00        28
       mixed       0.00      0.00      0.00         3
     sarcasm       0.00      0.00      0.00         0

    accuracy                           0.68       275
   macro avg       0.28      0.28      0.27       275
weighted avg       0.61      0.68      0.62       275

```

## Model: multinomial_naive_bayes_train

### Test 1: group 1

#### Confusion matrix

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |             224 |               8 |              0 |            0 |              0 |
| true_negative |              22 |              29 |              0 |            0 |              0 |
| true_neutral  |              69 |              37 |              0 |            0 |              0 |
| true_mixed    |               2 |               0 |              0 |            0 |              0 |
| true_sarcasm  |               0 |               0 |              0 |            0 |              0 |

#### Classification report

```text
              precision    recall  f1-score   support

    positive       0.71      0.97      0.82       232
    negative       0.39      0.57      0.46        51
     neutral       0.00      0.00      0.00       106
       mixed       0.00      0.00      0.00         2
     sarcasm       0.00      0.00      0.00         0

    accuracy                           0.65       391
   macro avg       0.22      0.31      0.26       391
weighted avg       0.47      0.65      0.54       391

```

### Test 2: group 2

#### Confusion matrix

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |             126 |               9 |              0 |            0 |              0 |
| true_negative |             131 |             238 |              0 |            0 |              0 |
| true_neutral  |              63 |              49 |              2 |            0 |              0 |
| true_mixed    |              15 |              10 |              0 |            0 |              0 |
| true_sarcasm  |               1 |               5 |              0 |            0 |              0 |

#### Classification report

```text
              precision    recall  f1-score   support

    positive       0.38      0.93      0.54       135
    negative       0.77      0.64      0.70       369
     neutral       1.00      0.02      0.03       114
       mixed       0.00      0.00      0.00        25
     sarcasm       0.00      0.00      0.00         6

    accuracy                           0.56       649
   macro avg       0.43      0.32      0.25       649
weighted avg       0.69      0.56      0.52       649

```

### Test 3: group 3 (OURS)

#### Confusion matrix

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |             360 |              11 |              0 |            0 |              0 |
| true_negative |              62 |              88 |              0 |            0 |              0 |
| true_neutral  |              49 |              20 |              3 |            0 |              0 |
| true_mixed    |              11 |               5 |              0 |            0 |              0 |
| true_sarcasm  |               1 |               0 |              0 |            0 |              0 |

#### Classification report

```text
              precision    recall  f1-score   support

    positive       0.75      0.97      0.84       371
    negative       0.71      0.59      0.64       150
     neutral       1.00      0.04      0.08        72
       mixed       0.00      0.00      0.00        16
     sarcasm       0.00      0.00      0.00         1

    accuracy                           0.74       610
   macro avg       0.49      0.32      0.31       610
weighted avg       0.75      0.74      0.68       610

```

### Test 4: group 4

#### Confusion matrix

|               |   pred_positive |   pred_negative |   pred_neutral |   pred_mixed |   pred_sarcasm |
|:--------------|----------------:|----------------:|---------------:|-------------:|---------------:|
| true_positive |             144 |               7 |              0 |            0 |              0 |
| true_negative |              38 |              55 |              0 |            0 |              0 |
| true_neutral  |              16 |               8 |              4 |            0 |              0 |
| true_mixed    |               3 |               0 |              0 |            0 |              0 |
| true_sarcasm  |               0 |               0 |              0 |            0 |              0 |

#### Classification report

```text
              precision    recall  f1-score   support

    positive       0.72      0.95      0.82       151
    negative       0.79      0.59      0.67        93
     neutral       1.00      0.14      0.25        28
       mixed       0.00      0.00      0.00         3
     sarcasm       0.00      0.00      0.00         0

    accuracy                           0.74       275
   macro avg       0.50      0.34      0.35       275
weighted avg       0.76      0.74      0.70       275

```

