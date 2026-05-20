# Exploratory Data Analysis

## Label Distribution

The table below shows the label distribution for the training, validation, and test splits.

| Split                  |   positive |   negative |   neutral |   mixed |   sarcasm |   total |
|:-----------------------|-----------:|-----------:|----------:|--------:|----------:|--------:|
| TRAIN                  |       5434 |       2687 |      1514 |     251 |        54 |    9940 |
| VALIDATION             |       1059 |        446 |       221 |      45 |        18 |    1789 |
| Test 1: group 1        |        232 |         51 |       106 |       2 |         0 |     391 |
| Test 2: group 2        |        135 |        369 |       114 |      25 |         6 |     649 |
| Test 3: group 3 (OURS) |        371 |        150 |        72 |      16 |         1 |     610 |
| Test 4: group 4        |        151 |         93 |        28 |       3 |         0 |     275 |

## Notes

- The following sentiment labels were included:
  - positive
  - negative
  - neutral
  - mixed
  - sarcasm

- TRAIN represents the combined training set.
- VALIDATION represents the combined validation set.
- Test 3 is the test set created by our group.
