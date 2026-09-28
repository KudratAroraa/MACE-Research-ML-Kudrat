# verifies vectorised cascade equivalence across all paper configurations
from itertools import combinations, permutations

import numpy as np
import pytest

from mace.cascadeEngine import cascade, cascadeReference
from mace.publishedResults import PAPER_DATASET_ORDER, PAPER_POOLS


# check 5,865 dataset configurations against the literal implementation
@pytest.mark.slow
def test_cascade_equivalence_across_datasets(paperResults) -> None:
    checked = 0
    sigmaValues = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    percentageValues = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for datasetName in PAPER_DATASET_ORDER:
        result = paperResults["datasets"][datasetName]
        classCount = result["prepared"]["classes"].size
        probabilityCache = result["prepared"]["testCache"]
        pool = PAPER_POOLS[datasetName]
        for size in range(1, 4):
            for subset in combinations(pool, size):
                for order in permutations(subset):
                    matrices = [probabilityCache[name] for name in order]
                    configurations = [("sof", value) for value in sigmaValues]
                    configurations.extend(("pf", value) for value in percentageValues)
                    configurations.extend(("topK", [value] * size) for value in [1, 2, 3, 5, 8])
                    for criterion, parameter in configurations:
                        expectedPrediction, expectedScores = cascadeReference(
                            matrices,
                            classCount,
                            criterion,
                            parameter,
                        )
                        prediction, scores = cascade(matrices, classCount, criterion, parameter)
                        assert np.array_equal(prediction, expectedPrediction)
                        assert np.allclose(scores, expectedScores, atol=1.0e-12)
                        checked += 1
    assert checked == 5865
