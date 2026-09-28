| Subset   | Method                               |   Rows |   AccuracyMean |   AccuracyStd |   F1Mean |   F1Std |
|:---------|:-------------------------------------|-------:|---------------:|--------------:|---------:|--------:|
| Seen     | Case-folded MACE + missing-name rule |   2477 |         93.386 |         3.913 |   87.743 |  12.465 |
| Seen     | MACE SoF                             |   2477 |         92.943 |         3.801 |   87.614 |  12.181 |
| Seen     | MACE SoF + missing-name rule         |   2477 |         92.943 |         3.801 |   87.614 |  12.181 |
| Seen     | RF                                   |   2477 |         76.086 |         1.393 |   55.129 |   5.487 |
| Seen     | RG-MACE                              |   2477 |         92.943 |         4.060 |   86.986 |  12.871 |
| Unseen   | Case-folded MACE + missing-name rule |   1982 |         79.300 |         2.888 |   60.829 |   6.642 |
| Unseen   | MACE SoF                             |   1982 |         79.086 |         2.411 |   59.971 |   4.698 |
| Unseen   | MACE SoF + missing-name rule         |   1982 |         79.086 |         2.411 |   59.971 |   4.698 |
| Unseen   | RF                                   |   1982 |         71.071 |         2.267 |   42.814 |   3.056 |
| Unseen   | RG-MACE                              |   1982 |         79.329 |         3.223 |   58.971 |   6.080 |