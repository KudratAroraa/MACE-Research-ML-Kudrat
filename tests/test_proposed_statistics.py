# verifies the deterministic paired statistical procedures used by Part 2
import numpy as np

from mace.proposedStatistics import (
    areaUnderRobustnessCurve,
    mcnemarExact,
    pairedBootstrapDifference,
    seedDifferenceInterval,
)


# check exact McNemar discordance counts and binomial p value on a hand-made example
def test_mcnemar_exact_known_example() -> None:
    truth = np.array([0, 0, 1, 1, 0, 1])
    first = np.array([0, 0, 1, 0, 1, 1])
    second = np.array([0, 1, 1, 1, 0, 1])
    result = mcnemarExact(truth, first, second)
    assert result["AOnlyCorrect"] == 1
    assert result["BOnlyCorrect"] == 2
    assert np.isclose(result["PValue"], 1.0)


# check paired bootstrap output is repeatable and brackets its observed difference
def test_paired_bootstrap_is_deterministic() -> None:
    truth = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    first = {"predictions": np.array([0, 1, 1, 0, 0, 1, 1, 1]), "probabilities": np.tile([0.5, 0.5], (8, 1))}
    second = {"predictions": np.array([0, 0, 1, 1, 0, 1, 1, 1]), "probabilities": np.tile([0.5, 0.5], (8, 1))}
    one = pairedBootstrapDifference(truth, first, second, "Accuracy", 200, 100)
    two = pairedBootstrapDifference(truth, first, second, "Accuracy", 200, 100)
    assert one == two
    assert one["Lower95"] <= one["ObservedDifference"] <= one["Upper95"]


# check normalised robustness area preserves a constant performance curve
def test_constant_robustness_curve() -> None:
    rates = np.array([0.0, 0.25, 0.5, 1.0])
    values = np.full(rates.size, 82.5)
    assert np.isclose(areaUnderRobustnessCurve(rates, values, [0.0, 0.5]), 82.5)


# check identical seed differences have a zero-width interval at zero
def test_identical_seed_differences() -> None:
    assert seedDifferenceInterval(np.zeros(7), 0.95) == (0.0, 0.0)
