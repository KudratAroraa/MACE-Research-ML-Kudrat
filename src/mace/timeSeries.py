# provides transparent time-series comparators for methods missing from the released code
from __future__ import annotations

import numpy as np
from numba import njit, prange
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .classifiers import ClasswiseNearestClassifier


# calculate banded dynamic time warping in pure Python as the reference for the compiled version
def dynamicTimeWarpingReference(
    left: np.ndarray,
    right: np.ndarray,
    windowFraction: float = 0.10,
) -> float:
    window = max(abs(len(left) - len(right)), int(np.ceil(max(len(left), len(right)) * windowFraction)))
    previous = np.full(len(right) + 1, np.inf)
    previous[0] = 0.0
    for leftIndex, leftValue in enumerate(left, start=1):
        current = np.full(len(right) + 1, np.inf)
        start = max(1, leftIndex - window)
        stop = min(len(right), leftIndex + window)
        for rightIndex in range(start, stop + 1):
            rightValue = right[rightIndex - 1]
            cost = abs(leftValue - rightValue)
            current[rightIndex] = cost + min(current[rightIndex - 1], previous[rightIndex], previous[rightIndex - 1])
        previous = current
    return float(previous[-1])


# calculate classic DTW with absolute-difference cost and a Sakoe-Chiba band
# a window at least as long as the series makes the alignment unconstrained
@njit(cache=True)
def _dtwPair(left: np.ndarray, right: np.ndarray, window: int) -> float:
    leftLength, rightLength = left.shape[0], right.shape[0]
    band = max(window, abs(leftLength - rightLength))
    previous = np.full(rightLength + 1, np.inf)
    current = np.full(rightLength + 1, np.inf)
    previous[0] = 0.0
    for leftIndex in range(1, leftLength + 1):
        current[:] = np.inf
        start = max(1, leftIndex - band)
        stop = min(rightLength, leftIndex + band)
        for rightIndex in range(start, stop + 1):
            best = previous[rightIndex - 1]
            if previous[rightIndex] < best:
                best = previous[rightIndex]
            if current[rightIndex - 1] < best:
                best = current[rightIndex - 1]
            current[rightIndex] = abs(left[leftIndex - 1] - right[rightIndex - 1]) + best
        previous, current = current, previous
    return previous[rightLength]


# calculate every query-reference DTW distance with test series spread across all CPU cores
@njit(parallel=True, cache=True)
def dtwDistanceMatrix(queries: np.ndarray, references: np.ndarray, window: int) -> np.ndarray:
    distances = np.empty((queries.shape[0], references.shape[0]))
    for queryIndex in prange(queries.shape[0]):
        for referenceIndex in range(references.shape[0]):
            distances[queryIndex, referenceIndex] = _dtwPair(queries[queryIndex], references[referenceIndex], window)
    return distances


# use the authors' 1NN-DTW class scores of one over the nearest class distance plus 0.001
class DTWNearestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, windowFraction: float | None = None, epsilon: float = 0.001):
        # none leaves the alignment unconstrained as in the authors' DTWDistance with w=1000
        self.windowFraction = windowFraction
        self.epsilon = epsilon

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DTWNearestClassifier":
        self.series_ = np.ascontiguousarray(X, dtype=float)
        self.labels_ = np.asarray(y)
        self.classes_ = np.unique(self.labels_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        queries = np.ascontiguousarray(X, dtype=float)
        length = max(queries.shape[1], self.series_.shape[1])
        window = length if self.windowFraction is None else int(np.ceil(length * self.windowFraction))
        distances = dtwDistanceMatrix(queries, self.series_, window)
        nearest = np.column_stack([distances[:, self.labels_ == label].min(axis=1) for label in self.classes_])
        scores = 1.0 / (nearest + self.epsilon)
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# summarise first differences as a fixed histogram for an explicit SDE proxy
class SlopeDistributionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bins: int = 20):
        self.bins = bins

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SlopeDistributionTransformer":
        slopes = np.diff(np.asarray(X, dtype=float), axis=1).ravel()
        lower, upper = np.quantile(slopes, [0.01, 0.99])
        if lower == upper:
            lower, upper = lower - 1.0, upper + 1.0
        self.edges_ = np.linspace(lower, upper, self.bins + 1)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        rows = []
        for series in np.asarray(X, dtype=float):
            histogram, _ = np.histogram(np.diff(series), bins=self.edges_, density=False)
            rows.append(histogram / max(histogram.sum(), 1))
        return np.asarray(rows)


# create a transparent value-transition histogram as a BOPF proxy
class QuantisedPatternTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bins: int = 8):
        self.bins = bins

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "QuantisedPatternTransformer":
        values = np.asarray(X, dtype=float).ravel()
        quantiles = np.linspace(0.0, 1.0, self.bins + 1)[1:-1]
        self.edges_ = np.unique(np.quantile(values, quantiles))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        rows = []
        classCount = self.edges_.size + 1
        for series in np.asarray(X, dtype=float):
            symbols = np.digitize(series, self.edges_)
            transitions = symbols[:-1] * classCount + symbols[1:]
            histogram = np.bincount(transitions, minlength=classCount**2).astype(float)
            rows.append(histogram / max(histogram.sum(), 1.0))
        return np.asarray(rows)


# create fixed local summaries as a shapelet-inspired LTS proxy
class SegmentSummaryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, segments: int = 16):
        self.segments = segments

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SegmentSummaryTransformer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        rows = []
        for series in np.asarray(X, dtype=float):
            chunks = np.array_split(series, self.segments)
            means = [float(np.mean(chunk)) for chunk in chunks]
            deviations = [float(np.std(chunk)) for chunk in chunks]
            rows.append(means + deviations)
        return np.asarray(rows)


# build exact available methods and clearly labelled proxies for unreleased methods
def buildTimeSeriesComparators(seed: int = 100) -> dict[str, BaseEstimator]:
    comparators: dict[str, BaseEstimator] = {
        "1NN-ED": ClasswiseNearestClassifier(epsilon=0.001),
        "1NN-DTW": DTWNearestClassifier(),
        "1NN-DTW-band10": DTWNearestClassifier(windowFraction=0.10),
        "BOPF-Proxy": Pipeline(
            [
                ("patterns", QuantisedPatternTransformer(bins=8)),
                ("model", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)),
            ]
        ),
        "SDE-Proxy": Pipeline(
            [
                ("slopes", SlopeDistributionTransformer(bins=20)),
                ("scale", StandardScaler()),
                ("model", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)),
            ]
        ),
        "LTS-Proxy": Pipeline(
            [
                ("segments", SegmentSummaryTransformer(segments=16)),
                ("scale", StandardScaler()),
                ("model", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)),
            ]
        ),
    }
    return comparators
