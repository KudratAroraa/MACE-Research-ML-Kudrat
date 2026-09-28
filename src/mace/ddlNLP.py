# implements the metadata distance classifier used for ThingSpeak stream names
from __future__ import annotations

import numpy as np
from pyxdameraulevenshtein import damerau_levenshtein_distance
from sklearn.base import BaseEstimator, ClassifierMixin


# calculate the restricted distance used by the released dependency
def damerauLevenshtein(left: str, right: str) -> int:
    return int(damerau_levenshtein_distance(left, right))


# calculate unrestricted distance for the labelled sensitivity analysis
def unrestrictedDamerauLevenshtein(left: str, right: str) -> int:
    leftLength, rightLength = len(left), len(right)
    maximum = leftLength + rightLength
    matrix = np.zeros((leftLength + 2, rightLength + 2), dtype=int)
    matrix[0, :] = maximum
    matrix[:, 0] = maximum
    matrix[1, 1 : rightLength + 2] = np.arange(rightLength + 1)
    matrix[1 : leftLength + 2, 1] = np.arange(leftLength + 1)
    lastSeen: dict[str, int] = {}

    # update the dynamic-programming matrix while tracking transpositions
    for leftIndex in range(1, leftLength + 1):
        matchColumn = 0
        for rightIndex in range(1, rightLength + 1):
            previousLeft = lastSeen.get(right[rightIndex - 1], 0)
            previousRight = matchColumn
            cost = 1
            if left[leftIndex - 1] == right[rightIndex - 1]:
                cost = 0
                matchColumn = rightIndex
            matrix[leftIndex + 1, rightIndex + 1] = min(
                matrix[leftIndex, rightIndex] + cost,
                matrix[leftIndex + 1, rightIndex] + 1,
                matrix[leftIndex, rightIndex + 1] + 1,
                matrix[previousLeft, previousRight]
                + (leftIndex - previousLeft - 1)
                + 1
                + (rightIndex - previousRight - 1),
            )
        lastSeen[left[leftIndex - 1]] = leftIndex
    return int(matrix[leftLength + 1, rightLength + 1])


# convert nearest classwise metadata distances into comparable class scores
class DDLNLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        distancePower: int = 2,
        caseSensitive: bool = True,
        normaliseLength: bool = True,
        epsilon: float = 1.0e-6,
        distanceVariant: str = "restricted",
    ):
        self.distancePower = distancePower
        self.caseSensitive = caseSensitive
        self.normaliseLength = normaliseLength
        self.epsilon = epsilon
        self.distanceVariant = distanceVariant

    def _prepare(self, values: np.ndarray) -> np.ndarray:
        prepared = np.asarray(values, dtype=str).reshape(-1)
        if not self.caseSensitive:
            prepared = np.char.lower(prepared)
        return prepared

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DDLNLPClassifier":
        self.metadata_ = self._prepare(X)
        self.labels_ = np.asarray(y)
        self.classes_ = np.unique(self.labels_)
        if self.distancePower <= 0:
            raise ValueError("distancePower must be positive")
        if self.distanceVariant not in {"restricted", "unrestricted"}:
            raise ValueError("distanceVariant must be 'restricted' or 'unrestricted'")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        queries = self._prepare(X)
        scores = np.zeros((queries.size, self.classes_.size), dtype=float)
        for rowIndex, query in enumerate(queries):
            for classIndex, label in enumerate(self.classes_):
                candidates = self.metadata_[self.labels_ == label]
                distances = []
                for candidate in candidates:
                    distanceFunction = (
                        damerauLevenshtein
                        if self.distanceVariant == "restricted"
                        else unrestrictedDamerauLevenshtein
                    )
                    distance = float(distanceFunction(query, candidate))
                    if self.normaliseLength:
                        distance /= max(len(query), len(candidate), 1)
                    distances.append(distance)
                nearest = min(distances)
                scores[rowIndex, classIndex] = 1.0 / ((nearest + self.epsilon) ** self.distancePower)
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
