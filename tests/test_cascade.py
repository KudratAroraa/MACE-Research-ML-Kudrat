# validates MACE filtering rules and probability alignment
import numpy as np

from mace.cascadeEngine import cascade
from mace.evaluation import alignProbabilities


# check PF floor keeps one class and PF ceiling keeps two so their predictions differ
def test_percentage_floor_and_ceiling_differ() -> None:
    probabilityList = [np.array([[0.5, 0.3, 0.15, 0.05]]), np.array([[0.1, 0.9, 0.0, 0.0]])]
    floorPrediction, floorScores = cascade(probabilityList, 4, "pf", 0.4, "floor")
    ceilingPrediction, ceilingScores = cascade(probabilityList, 4, "pf", 0.4, "ceiling")
    assert np.count_nonzero(floorScores[0]) == 1
    assert np.count_nonzero(ceilingScores[0]) == 2
    assert floorPrediction[0] == 0
    assert ceilingPrediction[0] == 1


# check SoF keeps the top class when no class passes a very high threshold
def test_sof_always_retains_one_class() -> None:
    probabilityList = [np.array([[0.4, 0.3, 0.3]]), np.array([[0.1, 0.8, 0.1]])]
    prediction, scores = cascade(probabilityList, 3, "sof", 100.0)
    assert np.count_nonzero(scores[0]) == 1
    assert prediction[0] == 0


# check probability columns follow the requested class order with zero for a class the model never saw
def test_probability_alignment_preserves_requested_class_order() -> None:
    probabilities = np.array([[0.7, 0.3]])
    aligned = alignProbabilities(probabilities, np.array([2, 0]), np.array([0, 1, 2]))
    assert aligned[0, 2] > aligned[0, 0]
    assert aligned[0, 1] == 0.0
    assert aligned.shape == (1, 3)
    np.testing.assert_allclose(aligned.sum(axis=1), 1.0)
