# validates dataset parsing, deterministic splitting and duplicate fingerprints
import numpy as np

from mace.data import DatasetBundle, seriesFingerprints, splitDataset


# create a balanced in-memory dataset without depending on source files
def syntheticBundle() -> DatasetBundle:
    series = np.arange(120, dtype=float).reshape(12, 10)
    return DatasetBundle(
        "synthetic",
        "Synthetic",
        np.array([f"id{index}" for index in range(12)]),
        series,
        np.repeat([0, 1, 2], 4),
        np.array(["stream"] * 12),
        None,
    )


# check the same seed gives the same stratified train and test indices
def test_stratified_split_is_deterministic() -> None:
    bundle = syntheticBundle()
    first = splitDataset(bundle, testSize=0.25, seed=100)
    second = splitDataset(bundle, testSize=0.25, seed=100)
    np.testing.assert_array_equal(first.trainIndex, second.trainIndex)
    np.testing.assert_array_equal(first.testIndex, second.testIndex)


# check identical series share a fingerprint and different series do not
def test_fingerprints_match_only_equal_series() -> None:
    values = np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 1.0]])
    fingerprints = seriesFingerprints(values)
    assert fingerprints[0] == fingerprints[1]
    assert fingerprints[0] != fingerprints[2]
