| Condition   | Comparison                                     | Metric   |   ObservedDifference |   Lower95 |   Upper95 |   ProbabilityDifferenceAtMostZero |
|:------------|:-----------------------------------------------|:---------|---------------------:|----------:|----------:|----------------------------------:|
| Clean       | RG-MACE - MACE SoF                             | Accuracy |                0.628 |    -1.099 |     2.512 |                             0.266 |
| Clean       | RG-MACE - MACE SoF                             | F1       |               -0.535 |    -5.139 |     2.898 |                             0.637 |
| Clean       | RG-MACE - MACE SoF                             | LogLoss  |               -1.202 |    -1.777 |    -0.653 |                             1.000 |
| Unseen      | RG-MACE - MACE SoF                             | Accuracy |                1.053 |    -2.456 |     4.561 |                             0.309 |
| Unseen      | RG-MACE - MACE SoF                             | F1       |                0.237 |    -5.920 |     5.806 |                             0.490 |
| Unseen      | RG-MACE - MACE SoF                             | LogLoss  |               -2.223 |    -3.231 |    -1.284 |                             1.000 |
| Missing25   | RG-MACE - MACE SoF                             | Accuracy |               17.896 |    14.600 |    21.193 |                             0.000 |
| Missing25   | RG-MACE - MACE SoF                             | F1       |                6.805 |     3.746 |    10.417 |                             0.000 |
| Missing25   | RG-MACE - MACE SoF                             | LogLoss  |               -8.810 |    -9.970 |    -7.614 |                             1.000 |
| Unseen      | RG-MACE - missing-name rule                    | Accuracy |                1.053 |    -2.456 |     4.561 |                             0.309 |
| Unseen      | RG-MACE - missing-name rule                    | F1       |                0.237 |    -5.920 |     5.806 |                             0.490 |
| Unseen      | RG-MACE - missing-name rule                    | LogLoss  |               -2.223 |    -3.231 |    -1.284 |                             1.000 |
| Clean       | RG-MACE - Case-folded MACE + missing-name rule | Accuracy |                0.000 |    -1.256 |     1.256 |                             0.558 |
| Clean       | RG-MACE - Case-folded MACE + missing-name rule | F1       |               -0.175 |    -1.025 |     0.566 |                             0.668 |
| Clean       | RG-MACE - Case-folded MACE + missing-name rule | LogLoss  |               -1.093 |    -1.586 |    -0.653 |                             1.000 |
| Missing25   | RG-MACE - Case-folded MACE + missing-name rule | Accuracy |               -0.471 |    -1.727 |     0.628 |                             0.830 |
| Missing25   | RG-MACE - Case-folded MACE + missing-name rule | F1       |               -0.158 |    -1.083 |     0.680 |                             0.640 |
| Missing25   | RG-MACE - Case-folded MACE + missing-name rule | LogLoss  |               -0.852 |    -1.280 |    -0.461 |                             1.000 |