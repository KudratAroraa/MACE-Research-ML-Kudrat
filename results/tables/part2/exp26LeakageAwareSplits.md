| SplitStrategy   | Condition   | Method                               |   Accuracy |   Precision |   Recall |     F1 |    AUC |   BalancedAccuracy |   WeightedF1 |   LogLoss |   BrierScore |
|:----------------|:------------|:-------------------------------------|-----------:|------------:|---------:|-------:|-------:|-------------------:|-------------:|----------:|-------------:|
| channelAware    | Clean       | Case-folded MACE + missing-name rule |     87.000 |      84.700 |   79.400 | 80.600 | 93.700 |             79.400 |       86.700 |     1.875 |        0.202 |
| channelAware    | Clean       | MACE SoF                             |     86.700 |      85.100 |   78.700 | 80.600 | 92.600 |             78.700 |       86.400 |     2.202 |        0.208 |
| channelAware    | Clean       | MACE SoF + missing-name rule         |     86.700 |      85.100 |   78.700 | 80.600 | 92.600 |             78.700 |       86.400 |     2.202 |        0.208 |
| channelAware    | Clean       | RF                                   |     72.500 |      58.400 |   49.200 | 49.600 | 92.700 |             49.200 |       71.500 |     1.187 |        0.387 |
| channelAware    | Clean       | RG-MACE                              |     87.100 |      84.900 |   79.500 | 80.800 | 98.000 |             79.500 |       86.500 |     0.468 |        0.184 |
| channelAware    | Missing25   | Case-folded MACE + missing-name rule |     82.400 |      79.300 |   70.500 | 72.200 | 94.800 |             70.500 |       82.000 |     1.641 |        0.251 |
| channelAware    | Missing25   | MACE SoF                             |     68.400 |      83.700 |   61.900 | 68.700 | 81.800 |             61.900 |       72.400 |     9.313 |        0.585 |
| channelAware    | Missing25   | MACE SoF + missing-name rule         |     82.000 |      79.000 |   69.500 | 71.800 | 94.200 |             69.500 |       81.600 |     1.931 |        0.259 |
| channelAware    | Missing25   | RF                                   |     72.500 |      58.400 |   49.200 | 49.600 | 92.700 |             49.200 |       71.500 |     1.187 |        0.387 |
| channelAware    | Missing25   | RG-MACE                              |     82.300 |      81.300 |   70.400 | 73.000 | 96.100 |             70.400 |       81.900 |     0.732 |        0.244 |
| groupAware      | Clean       | Case-folded MACE + missing-name rule |     88.100 |      86.500 |   82.700 | 84.000 | 92.800 |             82.700 |       88.000 |     2.840 |        0.210 |
| groupAware      | Clean       | MACE SoF                             |     90.400 |      87.000 |   82.700 | 84.200 | 93.500 |             82.700 |       90.200 |     1.741 |        0.171 |
| groupAware      | Clean       | MACE SoF + missing-name rule         |     90.400 |      87.000 |   82.700 | 84.200 | 93.500 |             82.700 |       90.200 |     1.741 |        0.171 |
| groupAware      | Clean       | RF                                   |     73.700 |      48.400 |   45.300 | 45.400 | 90.100 |             45.300 |       71.700 |     1.286 |        0.372 |
| groupAware      | Clean       | RG-MACE                              |     89.800 |      87.500 |   83.100 | 84.600 | 96.100 |             83.100 |       89.500 |     0.418 |        0.168 |
| groupAware      | Missing25   | Case-folded MACE + missing-name rule |     85.600 |      85.200 |   75.700 | 78.600 | 92.000 |             75.700 |       85.300 |     2.399 |        0.237 |
| groupAware      | Missing25   | MACE SoF                             |     70.900 |      86.700 |   65.900 | 73.100 | 82.600 |             65.900 |       74.700 |     9.293 |        0.567 |
| groupAware      | Missing25   | MACE SoF + missing-name rule         |     87.100 |      86.100 |   76.200 | 79.300 | 92.400 |             76.200 |       86.800 |     1.623 |        0.210 |
| groupAware      | Missing25   | RF                                   |     73.700 |      48.400 |   45.300 | 45.400 | 90.100 |             45.300 |       71.700 |     1.286 |        0.372 |
| groupAware      | Missing25   | RG-MACE                              |     86.300 |      86.500 |   74.800 | 78.700 | 94.500 |             74.800 |       86.000 |     0.647 |        0.215 |
| stratified      | Clean       | Case-folded MACE + missing-name rule |     89.200 |      86.700 |   76.600 | 79.000 | 92.300 |             76.600 |       88.700 |     1.916 |        0.189 |
| stratified      | Clean       | MACE SoF                             |     88.500 |      84.500 |   77.400 | 79.400 | 94.200 |             77.400 |       88.200 |     2.025 |        0.200 |
| stratified      | Clean       | MACE SoF + missing-name rule         |     88.500 |      84.500 |   77.400 | 79.400 | 94.200 |             77.400 |       88.200 |     2.025 |        0.200 |
| stratified      | Clean       | RF                                   |     76.000 |      51.300 |   54.500 | 51.900 | 90.200 |             54.500 |       74.000 |     1.338 |        0.344 |
| stratified      | Clean       | RG-MACE                              |     89.200 |      87.200 |   76.100 | 78.800 | 94.800 |             76.100 |       88.400 |     0.824 |        0.180 |
| stratified      | Missing25   | Case-folded MACE + missing-name rule |     87.100 |      82.800 |   71.100 | 74.000 | 92.000 |             71.100 |       86.400 |     1.775 |        0.212 |
| stratified      | Missing25   | MACE SoF                             |     68.800 |      81.600 |   59.900 | 67.000 | 82.100 |             59.900 |       72.600 |     9.733 |        0.607 |
| stratified      | Missing25   | MACE SoF + missing-name rule         |     86.500 |      80.400 |   70.600 | 73.200 | 92.000 |             70.600 |       85.800 |     1.993 |        0.223 |
| stratified      | Missing25   | RF                                   |     76.000 |      51.300 |   54.500 | 51.900 | 90.200 |             54.500 |       74.000 |     1.338 |        0.344 |
| stratified      | Missing25   | RG-MACE                              |     86.700 |      83.400 |   70.400 | 73.800 | 94.200 |             70.400 |       85.700 |     0.923 |        0.211 |