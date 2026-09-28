| Dataset          | Method                 |   Accuracy |   Precision |   Recall |    F1 |   AUC |   BalancedAccuracy |   WeightedF1 |   LogLoss |
|:-----------------|:-----------------------|-----------:|------------:|---------:|------:|------:|-------------------:|-------------:|----------:|
| UrbanObservatory | Best standalone        |      91.20 |       89.30 |    88.30 | 88.20 | 98.90 |              88.30 |        91.30 |      0.43 |
| UrbanObservatory | Soft voting            |      90.60 |       89.40 |    89.10 | 88.50 | 99.30 |              89.10 |        90.80 |      0.48 |
| UrbanObservatory | MACE Top-k             |      91.20 |       89.30 |    88.30 | 88.20 | 98.90 |              88.30 |        91.30 |      0.37 |
| UrbanObservatory | MACE PF                |      91.20 |       89.30 |    88.30 | 88.20 | 98.90 |              88.30 |        91.30 |      0.37 |
| UrbanObservatory | MACE SoF               |      91.20 |       89.30 |    88.30 | 88.20 | 98.70 |              88.30 |        91.30 |      0.38 |
| UrbanObservatory | MACE brute-force (SoF) |      92.50 |       91.00 |    89.50 | 89.80 | 97.30 |              89.50 |        92.40 |      1.45 |
| Swissex          | Best standalone        |      84.60 |       83.00 |    84.00 | 83.20 | 97.20 |              84.00 |        84.40 |      0.63 |
| Swissex          | Soft voting            |      81.70 |       81.00 |    77.30 | 77.80 | 97.10 |              77.30 |        80.90 |      0.73 |
| Swissex          | MACE Top-k             |      84.60 |       83.00 |    84.00 | 83.20 | 97.20 |              84.00 |        84.40 |      0.63 |
| Swissex          | MACE PF                |      84.60 |       83.00 |    84.00 | 83.20 | 97.20 |              84.00 |        84.40 |      0.63 |
| Swissex          | MACE SoF               |      82.70 |       79.80 |    81.50 | 80.00 | 92.00 |              81.50 |        81.90 |      3.43 |
| Swissex          | MACE brute-force (SoF) |      86.50 |       90.00 |    82.30 | 83.90 | 90.40 |              82.30 |        85.70 |      4.85 |
| ThingSpeakEU     | Best standalone        |      76.00 |       75.10 |    70.90 | 71.90 | 95.70 |              70.90 |        74.00 |      1.18 |
| ThingSpeakEU     | Soft voting            |      80.10 |       59.90 |    59.70 | 58.80 | 97.10 |              59.70 |        78.10 |      0.75 |
| ThingSpeakEU     | MACE Top-k             |      84.30 |       70.70 |    67.20 | 67.60 | 92.90 |              67.20 |        83.70 |      1.88 |
| ThingSpeakEU     | MACE PF                |      83.50 |       71.80 |    63.30 | 64.80 | 92.40 |              63.30 |        82.80 |      1.62 |
| ThingSpeakEU     | MACE SoF               |      88.50 |       84.50 |    77.40 | 79.40 | 94.20 |              77.40 |        88.20 |      2.03 |
| ThingSpeakEU     | MACE brute-force (SoF) |      89.00 |       84.50 |    76.80 | 79.00 | 95.50 |              76.80 |        88.40 |      1.03 |