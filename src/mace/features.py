# extracts the eleven statistical BOS features in the order used by the authors
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import iqr, kurtosis, skew


BOS_FEATURE_NAMES = [
    "Range",
    "Maximum",
    "StandardDeviation",
    "RootMeanSquare",
    "Mean",
    "Minimum",
    "Quantile40",
    "Median",
    "Kurtosis",
    "Skewness",
    "InterquartileRange",
]


# apply an explicit policy to undefined skewness and kurtosis values
def _resolveNonFinite(values: np.ndarray, policy: str) -> np.ndarray:
    if policy == "zero":
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if policy == "nan":
        return values
    if policy == "raise" and not np.isfinite(values).all():
        raise ValueError("BOS extraction produced non-finite values")
    if policy not in {"zero", "nan", "raise"}:
        raise ValueError("constantMomentPolicy must be 'zero', 'nan' or 'raise'")
    return values


# convert fixed-length series into the eleven published statistical descriptors one series at a time
# a vectorised axis=1 reduction can leave a tiny non-zero std on constant series which turns
# skewness into plus or minus one and kurtosis into minus two and changes ThingSpeak tree splits
def extractBOSFeatures(series: np.ndarray, constantMomentPolicy: str = "zero") -> pd.DataFrame:
    values = np.asarray(series, dtype=float)
    if values.ndim != 2:
        raise ValueError("series must be a two-dimensional array")

    # follow the authors' per-row loop and suppress expected precision warnings
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for row in values:
            rows.append(
                [
                    row.max() - row.min(),
                    row.max(),
                    row.std(),
                    np.sqrt(np.mean(np.square(row))),
                    row.mean(),
                    row.min(),
                    np.quantile(row, 0.4),
                    np.median(row),
                    kurtosis(row, fisher=True, bias=True),
                    skew(row, bias=True),
                    iqr(row),
                ]
            )
    featureMatrix = np.asarray(rows, dtype=float).reshape(-1, len(BOS_FEATURE_NAMES))
    featureMatrix = _resolveNonFinite(featureMatrix, constantMomentPolicy)
    return pd.DataFrame(featureMatrix, columns=BOS_FEATURE_NAMES)


# normalise each time series independently without producing divide-by-zero values
def zNormaliseSeries(series: np.ndarray, epsilon: float = 1.0e-12) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    means = values.mean(axis=1, keepdims=True)
    scales = values.std(axis=1, keepdims=True)
    safeScales = np.where(scales < epsilon, 1.0, scales)
    return (values - means) / safeScales
