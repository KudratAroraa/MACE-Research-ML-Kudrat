# defines independently implemented baseline classifiers for each information domain
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .ddlNLP import DDLNLPClassifier


# use the nearest training series within each class to form class probabilities
class ClasswiseNearestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, epsilon: float = 0.001, distancePower: float = 1.0):
        self.epsilon = epsilon
        self.distancePower = distancePower

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClasswiseNearestClassifier":
        self.series_ = np.asarray(X, dtype=float)
        self.labels_ = np.asarray(y)
        self.classes_ = np.unique(self.labels_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        queries = np.asarray(X, dtype=float)
        distances = cdist(queries, self.series_, metric="euclidean")
        scores = np.empty((queries.shape[0], self.classes_.size), dtype=float)
        for classIndex, label in enumerate(self.classes_):
            nearest = distances[:, self.labels_ == label].min(axis=1)
            scores[:, classIndex] = 1.0 / ((nearest + self.epsilon) ** self.distancePower)
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# convert estimators without predict_proba into normalised class scores
class DecisionFunctionProbabilityAdapter(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator: Any):
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionFunctionProbabilityAdapter":
        self.estimator_ = self.estimator.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = np.asarray(self.estimator_.decision_function(X), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict(X)


# build the eight Bag-of-Summaries classifiers reported in the paper
# jobs sets Random Forest parallelism and is one inside parallel workers to avoid nested parallelism
def buildBOSClassifiers(seed: int = 100, scale: bool = False, jobs: int = -1) -> dict[str, BaseEstimator]:
    def scaled(estimator: BaseEstimator) -> BaseEstimator:
        return Pipeline([("scale", StandardScaler()), ("model", estimator)]) if scale else estimator

    return {
        "Ridge": DecisionFunctionProbabilityAdapter(scaled(RidgeClassifier())),
        "SVM": scaled(SVC(kernel="rbf", probability=True, random_state=seed)),
        "kNN": scaled(KNeighborsClassifier(n_neighbors=5)),
        "LogisticRegression": scaled(LogisticRegression(max_iter=1000, random_state=seed)),
        "DecisionTree": DecisionTreeClassifier(criterion="entropy", random_state=seed),
        "GaussianNB": GaussianNB(),
        "RandomForest": RandomForestClassifier(n_estimators=1000, random_state=seed, n_jobs=jobs),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=seed),
    }


# build the five-model candidate pool used by each MACE experiment
def buildCandidatePool(
    datasetKey: str,
    seed: int = 100,
    ddlDistancePower: int = 2,
    ddlDistanceVariant: str = "restricted",
    jobs: int = -1,
) -> dict[str, tuple[str, BaseEstimator]]:
    bos = buildBOSClassifiers(seed, jobs=jobs)
    pool = {
        "1NN-ED": ("series", ClasswiseNearestClassifier()),
        "kNN": ("bos", bos["kNN"]),
    }
    if datasetKey == "thingspeak":
        pool["DDL-NLP"] = (
            "metadata",
            DDLNLPClassifier(
                distancePower=ddlDistancePower,
                distanceVariant=ddlDistanceVariant,
            ),
        )
    else:
        pool["SVM"] = ("bos", bos["SVM"])
    pool["GradientBoosting"] = ("bos", bos["GradientBoosting"])
    pool["RandomForest"] = ("bos", bos["RandomForest"])
    return pool
