| Dataset          | Domains                 | Order        |   Sigma |   Accuracy |   Precision |   Recall |    F1 |   AUC |   BalancedAccuracy |   WeightedF1 |   LogLoss |
|:-----------------|:------------------------|:-------------|--------:|-----------:|------------:|---------:|------:|------:|-------------------:|-------------:|----------:|
| UrbanObservatory | series                  | 1NN-ED       |   -0.60 |      80.00 |       77.30 |    69.10 | 68.50 | 96.80 |              69.10 |        77.80 |      1.44 |
| UrbanObservatory | bos                     | SVM > RF     |   -0.60 |      91.20 |       89.30 |    88.30 | 88.20 | 98.90 |              88.30 |        91.30 |      0.38 |
| UrbanObservatory | series + bos            | 1NN-ED > RF  |   -0.60 |      91.20 |       89.30 |    88.30 | 88.20 | 98.70 |              88.30 |        91.30 |      0.38 |
| Swissex          | series                  | 1NN-ED       |   -0.60 |      58.70 |       57.70 |    56.30 | 55.30 | 90.70 |              56.30 |        59.90 |      1.43 |
| Swissex          | bos                     | SVM > RF     |   -0.40 |      82.70 |       80.20 |    80.70 | 80.20 | 95.00 |              80.70 |        82.20 |      1.28 |
| Swissex          | series + bos            | 1NN-ED > RF  |    0.40 |      82.70 |       79.80 |    81.50 | 80.00 | 92.00 |              81.50 |        81.90 |      3.43 |
| ThingSpeakEU     | series                  | 1NN-ED       |   -0.60 |      70.60 |       41.20 |    48.40 | 43.20 | 89.60 |              48.40 |        69.70 |      1.53 |
| ThingSpeakEU     | bos                     | RF           |   -0.60 |      76.00 |       51.30 |    54.50 | 51.90 | 91.60 |              54.50 |        74.00 |      1.07 |
| ThingSpeakEU     | metadata                | DDL-NLP      |   -0.60 |      73.20 |       75.10 |    70.90 | 71.90 | 95.70 |              70.90 |        72.90 |      1.16 |
| ThingSpeakEU     | series + bos            | 1NN-ED > RF  |    1.00 |      76.80 |       50.70 |    54.20 | 51.50 | 80.40 |              54.20 |        74.70 |      4.30 |
| ThingSpeakEU     | series + metadata       | DDL-NLP      |   -0.60 |      73.20 |       75.10 |    70.90 | 71.90 | 95.70 |              70.90 |        72.90 |      1.16 |
| ThingSpeakEU     | bos + metadata          | DDL-NLP > RF |    0.20 |      88.50 |       84.50 |    77.40 | 79.40 | 94.20 |              77.40 |        88.20 |      2.03 |
| ThingSpeakEU     | series + bos + metadata | DDL-NLP > RF |    0.20 |      88.50 |       84.50 |    77.40 | 79.40 | 94.20 |              77.40 |        88.20 |      2.03 |