# stores every protocol setting in one importable and notebook-readable module
from copy import deepcopy
from typing import Any


REPRODUCTION_CONFIG: dict[str, Any] = {
    "randomSeed": 100,
    "testSize": 0.30,
    "splitStrategy": "stratified",
    "constantMomentPolicy": "zero",
    "legacyPercentageFloor": True,
    "legacyFullCascade": True,
    "ddlDistancePower": 2,
    "datasets": {
        "thingspeak": {
            "file": "ThingspeakEU.meta.csv",
            "displayName": "ThingSpeakEU",
            "metadataIndex": 1,
            "labelIndex": 8,
            "seriesStartIndex": 9,
            "seriesLength": 96,
        },
        "swissex": {
            "file": "Swissex.meta.csv",
            "displayName": "Swissex",
            "metadataIndex": 1,
            "labelIndex": 8,
            "seriesStartIndex": 9,
            "seriesLength": 445,
        },
        "urban": {
            "file": "UrbanObservatory.meta.csv",
            "displayName": "UrbanObservatory",
            "metadataIndex": 1,
            "labelIndex": 8,
            "seriesStartIndex": 9,
            "seriesLength": 864,
        },
    },
    "selection": {
        "topKValues": [1, 2, 3, 4, 5],
        "sigmaValues": [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "qValues": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "cvFolds": 5,
        "scoring": "accuracy",
    },
    "models": {
        "randomForest": {"nEstimators": 1000, "randomState": 100},
        "gradientBoosting": {"nEstimators": 100, "learningRate": 0.1, "randomState": 100},
        "knn": {"nNeighbors": 5},
        "svm": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "probability": True},
    },
}

CRITICAL_ANALYSIS_CONFIG: dict[str, Any] = {
    "randomSeeds": [7, 21, 42, 77, 100, 123, 202],
    "bootstrapIterations": 1000,
    "confidenceLevel": 0.95,
    "metadataCorruptionRates": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    "labelNoiseRates": [0.0, 0.05, 0.1, 0.2],
    "duplicatePrecision": 12,
    "groupAwareFolds": 5,
    "calibrationBins": 10,
    "selectionFolds": 5,
    "primaryMetric": "f1_macro",
}


# return an isolated copy of the requested protocol settings
def loadConfig(name: str = "reproduction") -> dict[str, Any]:
    if name == "reproduction":
        return deepcopy(REPRODUCTION_CONFIG)
    if name == "criticalAnalysis":
        return deepcopy(CRITICAL_ANALYSIS_CONFIG)
    raise KeyError(f"Unknown configuration: {name}")
