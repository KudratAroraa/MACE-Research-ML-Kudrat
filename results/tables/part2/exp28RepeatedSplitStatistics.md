| Condition   | Comparison                                     | Metric     |   MeanDifference |   Lower95 |   Upper95 |   WilcoxonStatistic |   WilcoxonPValue |   MedianDifference |
|:------------|:-----------------------------------------------|:-----------|-----------------:|----------:|----------:|--------------------:|-----------------:|-------------------:|
| Clean       | RG-MACE - MACE SoF                             | Accuracy   |            0.086 |    -0.702 |     0.873 |              11.500 |            0.734 |             -0.300 |
| Clean       | RG-MACE - MACE SoF                             | F1         |           -2.014 |    -5.106 |     1.077 |               5.000 |            0.156 |             -1.300 |
| Clean       | RG-MACE - MACE SoF                             | AUC        |            3.486 |    -0.777 |     7.749 |               3.000 |            0.078 |              2.600 |
| Clean       | RG-MACE - MACE SoF                             | LogLoss    |           -1.837 |    -3.198 |    -0.477 |               0.000 |            0.016 |             -1.202 |
| Clean       | RG-MACE - MACE SoF                             | BrierScore |           -0.020 |    -0.040 |    -0.000 |               3.000 |            0.078 |             -0.020 |
| Clean       | RG-MACE - MACE SoF + missing-name rule         | Accuracy   |            0.086 |    -0.702 |     0.873 |              11.500 |            0.734 |             -0.300 |
| Clean       | RG-MACE - MACE SoF + missing-name rule         | F1         |           -2.014 |    -5.106 |     1.077 |               5.000 |            0.156 |             -1.300 |
| Clean       | RG-MACE - MACE SoF + missing-name rule         | AUC        |            3.486 |    -0.777 |     7.749 |               3.000 |            0.078 |              2.600 |
| Clean       | RG-MACE - MACE SoF + missing-name rule         | LogLoss    |           -1.837 |    -3.198 |    -0.477 |               0.000 |            0.016 |             -1.202 |
| Clean       | RG-MACE - MACE SoF + missing-name rule         | BrierScore |           -0.020 |    -0.040 |    -0.000 |               3.000 |            0.078 |             -0.020 |
| Clean       | RG-MACE - Case-folded MACE + missing-name rule | Accuracy   |           -0.257 |    -0.703 |     0.189 |               5.000 |            0.281 |             -0.200 |
| Clean       | RG-MACE - Case-folded MACE + missing-name rule | F1         |           -1.800 |    -3.972 |     0.372 |               0.000 |            0.016 |             -0.300 |
| Clean       | RG-MACE - Case-folded MACE + missing-name rule | AUC        |            3.914 |     0.048 |     7.781 |               0.000 |            0.016 |              2.500 |
| Clean       | RG-MACE - Case-folded MACE + missing-name rule | LogLoss    |           -1.877 |    -3.341 |    -0.413 |               0.000 |            0.016 |             -1.225 |
| Clean       | RG-MACE - Case-folded MACE + missing-name rule | BrierScore |           -0.017 |    -0.038 |     0.003 |               1.000 |            0.031 |             -0.009 |
| Clean       | RG-MACE - Soft stacking                        | Accuracy   |           -0.914 |    -3.439 |     1.610 |              11.000 |            0.688 |              0.300 |
| Clean       | RG-MACE - Soft stacking                        | F1         |           -1.300 |    -9.901 |     7.301 |              13.000 |            0.938 |              2.300 |
| Clean       | RG-MACE - Soft stacking                        | AUC        |           -3.143 |    -5.300 |    -0.986 |               0.000 |            0.016 |             -3.100 |
| Clean       | RG-MACE - Soft stacking                        | LogLoss    |            0.372 |     0.142 |     0.601 |               0.000 |            0.016 |              0.354 |
| Clean       | RG-MACE - Soft stacking                        | BrierScore |            0.018 |    -0.024 |     0.060 |              14.000 |            1.000 |             -0.005 |
| Missing25   | RG-MACE - MACE SoF                             | Accuracy   |            9.286 |     1.243 |    17.328 |               3.000 |            0.078 |             15.500 |
| Missing25   | RG-MACE - MACE SoF                             | F1         |            2.900 |    -1.030 |     6.830 |               4.000 |            0.109 |              2.900 |
| Missing25   | RG-MACE - MACE SoF                             | AUC        |            9.657 |     5.757 |    13.557 |               0.000 |            0.016 |             10.400 |
| Missing25   | RG-MACE - MACE SoF                             | LogLoss    |           -5.981 |    -9.050 |    -2.912 |               0.000 |            0.016 |             -7.995 |
| Missing25   | RG-MACE - MACE SoF                             | BrierScore |           -0.223 |    -0.392 |    -0.053 |               0.000 |            0.016 |             -0.344 |
| Missing25   | RG-MACE - MACE SoF + missing-name rule         | Accuracy   |           -0.157 |    -0.590 |     0.276 |               9.000 |            0.453 |             -0.200 |
| Missing25   | RG-MACE - MACE SoF + missing-name rule         | F1         |           -1.386 |    -3.028 |     0.256 |               2.000 |            0.094 |             -0.900 |
| Missing25   | RG-MACE - MACE SoF + missing-name rule         | AUC        |            2.529 |    -1.004 |     6.061 |               4.000 |            0.109 |              2.200 |
| Missing25   | RG-MACE - MACE SoF + missing-name rule         | LogLoss    |           -1.432 |    -2.501 |    -0.362 |               0.000 |            0.016 |             -1.070 |
| Missing25   | RG-MACE - MACE SoF + missing-name rule         | BrierScore |           -0.014 |    -0.029 |     0.001 |               3.000 |            0.078 |             -0.012 |
| Missing25   | RG-MACE - Case-folded MACE + missing-name rule | Accuracy   |           -0.429 |    -0.794 |    -0.064 |               2.000 |            0.047 |             -0.400 |
| Missing25   | RG-MACE - Case-folded MACE + missing-name rule | F1         |           -1.171 |    -2.843 |     0.500 |               3.000 |            0.078 |             -0.500 |
| Missing25   | RG-MACE - Case-folded MACE + missing-name rule | AUC        |            2.871 |    -0.517 |     6.260 |               0.000 |            0.016 |              1.100 |
| Missing25   | RG-MACE - Case-folded MACE + missing-name rule | LogLoss    |           -1.443 |    -2.608 |    -0.278 |               0.000 |            0.016 |             -0.962 |
| Missing25   | RG-MACE - Case-folded MACE + missing-name rule | BrierScore |           -0.011 |    -0.030 |     0.007 |               5.000 |            0.156 |             -0.003 |
| Missing25   | RG-MACE - Soft stacking                        | Accuracy   |           -0.914 |    -3.122 |     1.293 |               7.500 |            0.328 |             -1.300 |
| Missing25   | RG-MACE - Soft stacking                        | F1         |            0.043 |    -7.600 |     7.685 |              13.000 |            0.891 |              2.100 |
| Missing25   | RG-MACE - Soft stacking                        | AUC        |           -2.700 |    -4.173 |    -1.227 |               0.000 |            0.016 |             -2.100 |
| Missing25   | RG-MACE - Soft stacking                        | LogLoss    |            0.430 |     0.294 |     0.566 |               0.000 |            0.016 |              0.382 |
| Missing25   | RG-MACE - Soft stacking                        | BrierScore |            0.013 |    -0.020 |     0.047 |              12.000 |            0.812 |              0.006 |