| Criterion                              | Target                                            |   ObservedValue | Met   | Detail                                                      |
|:---------------------------------------|:--------------------------------------------------|----------------:|:------|:------------------------------------------------------------|
| Clean accuracy at seed 100             | >= 88.0%                                          |          89.200 | True  |                                                             |
| Clean macro F1 at seed 100             | >= 78.9%                                          |          78.800 | False |                                                             |
| Seven-seed clean non-inferiority       | Lower 95% bound > -1 point                        |          -0.702 | True  |                                                             |
| Clean log loss                         | Better than MACE SoF                              |          -1.202 | True  |                                                             |
| Clean Brier score                      | Better than MACE SoF                              |          -0.020 | True  |                                                             |
| Unseen-name accuracy                   | Lower paired seed CI above 0                      |          -0.766 | False |                                                             |
| Unseen-name macro F1                   | Lower paired seed CI above 0                      |          -5.283 | False |                                                             |
| 25% missing names                      | >= MACE SoF + 5 points                            |          15.933 | True  |                                                             |
| 100% missing names                     | >= 74.0%                                          |          75.500 | True  |                                                             |
| Primary corruption AUC                 | Higher for all four modes                         |          -0.045 | False | missing: +15.83; case: +4.16; delete: +0.27; replace: -0.05 |
| Learned gate on unseen names           | Lower paired seed CI above missing-name rule      |          -0.766 | False |                                                             |
| Learned gate on non-missing corruption | Higher AUC than rule for case, delete and replace |          -0.045 | False | case: +4.16; delete: +0.14; replace: -0.05                  |
| Learned gate on missing corruption     | Not worse than missing-name rule                  |           0.298 | True  |                                                             |
| Channel split clean                    | No more than 1 point below MACE                   |           0.400 | True  |                                                             |
| Channel split 25% missing              | Higher than MACE SoF                              |          13.900 | True  |                                                             |
| Prediction overhead                    | Under 10%                                         |          -6.612 | True  |                                                             |