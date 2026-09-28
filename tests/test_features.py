# validates published BOS feature order and constant-series handling
import warnings

import numpy as np
from scipy.stats import kurtosis, skew

from mace.data import loadDataset
from mace.features import BOS_FEATURE_NAMES, extractBOSFeatures


# check the eleven BOS features come out in the published order and shape
def test_bos_feature_order_and_shape() -> None:
    series = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    features = extractBOSFeatures(series)
    assert features.columns.tolist() == BOS_FEATURE_NAMES
    assert features.shape == (2, 11)


# check a constant series gives finite features with skewness and kurtosis set to zero
def test_constant_series_produces_finite_legacy_features() -> None:
    features = extractBOSFeatures(np.ones((1, 8)), constantMomentPolicy="zero")
    assert np.isfinite(features.to_numpy()).all()
    assert features.loc[0, "Kurtosis"] == 0.0
    assert features.loc[0, "Skewness"] == 0.0


# compare BOS extraction with a plain per-row loop on every source dataset
def test_constant_series_rows_match_per_row_reference() -> None:
    for datasetName in ["thingspeak", "swissex", "urban"]:
        series = loadDataset(datasetName).series
        features = extractBOSFeatures(series)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reference = np.array([[np.std(row), skew(row), kurtosis(row)] for row in series])
        reference = np.nan_to_num(reference, nan=0.0)
        assert np.array_equal(features["StandardDeviation"].to_numpy(), reference[:, 0])
        assert np.array_equal(features["Skewness"].to_numpy(), reference[:, 1])
        assert np.array_equal(features["Kurtosis"].to_numpy(), reference[:, 2])
