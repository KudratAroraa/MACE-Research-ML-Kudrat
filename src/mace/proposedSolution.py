# implements Reliability-Gated MACE and every directly comparable Part 2 baseline
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cascadeEngine import cascade, softVoting
from .evaluation import alignProbabilities
from .proposedConfig import PROPOSED_CONFIG
from .reliabilityGate import (
    RELIABILITY_FEATURES,
    ReliabilityDDL,
    _alignDistances,
    buildGateDataset,
    buildMaceExpert,
    fitGate,
    mixtureProbabilities,
    outOfFoldMatrix,
    reliabilityFeatures,
    selectGate,
)
from .reproductionPipeline import percentageMetrics
from .robustness import multiclassBrierScore


METHOD_NAMES = [
    "RF",
    "Soft voting",
    "MACE SoF",
    "MACE SoF + missing-name rule",
    "Case-folded MACE + missing-name rule",
    "Soft stacking",
    "RG-MACE (hard gate)",
    "RG-MACE",
]


# convert one fitted ReliabilityDDL output to the shared class order
def _ddlOutputs(
    model: ReliabilityDDL,
    names: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = alignProbabilities(model.predict_proba(names), model.classes_, classes)
    distances = _alignDistances(model.nearestDistances(names), model.classes_, classes)
    return probabilities, distances


# return the reproduced Random Forest expert without refitting it
def predictRandomForest(prepared: dict[str, Any]) -> dict[str, Any]:
    probabilities = prepared["testCache"]["RandomForest"].copy()
    return {"predictions": probabilities.argmax(axis=1), "probabilities": probabilities}


# average the full five-model pool after recomputing only name-dependent scores when required
def predictSoftVoting(
    prepared: dict[str, Any],
    testNames: np.ndarray | None = None,
    caseFold: bool = False,
    maceExpert: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cache = {name: values.copy() for name, values in prepared["testCache"].items()}
    if testNames is not None or caseFold:
        expert = maceExpert or buildMaceExpert(prepared, caseFold=caseFold)
        names = prepared["testInputs"]["metadata"] if testNames is None else np.asarray(testNames, dtype=str)
        probabilities, _ = _ddlOutputs(expert["fullDdlModel"], names, prepared["classes"])
        cache["DDL-NLP"] = probabilities
    pool = [specification.name for specification in prepared["specifications"]]
    predictions, probabilities = softVoting([cache[name] for name in pool])
    return {"predictions": predictions, "probabilities": probabilities}


# evaluate the paper SoF cascade with either its original name handling or the Part 2 case fold
def predictMaceSof(
    prepared: dict[str, Any],
    testNames: np.ndarray | None = None,
    caseFold: bool = False,
    maceExpert: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expert = maceExpert or buildMaceExpert(prepared, caseFold=caseFold)
    cache = {name: values.copy() for name, values in expert["testCache"].items()}
    names = prepared["testInputs"]["metadata"] if testNames is None else np.asarray(testNames, dtype=str)
    if testNames is not None:
        probabilities, _ = _ddlOutputs(expert["fullDdlModel"], names, prepared["classes"])
        cache["DDL-NLP"] = probabilities
    predictions, probabilities = cascade(
        [cache[name] for name in expert["order"]],
        prepared["classes"].size,
        "sof",
        expert["sigma"],
    )
    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "configuration": {"order": expert["order"], "sigma": expert["sigma"], "caseFold": caseFold},
        "expert": expert,
    }


# replace the MACE output with RF for every stream whose name is empty
def _missingNameFallback(
    prepared: dict[str, Any],
    testNames: np.ndarray | None,
    caseFold: bool,
    maceExpert: dict[str, Any] | None,
) -> dict[str, Any]:
    names = prepared["testInputs"]["metadata"] if testNames is None else np.asarray(testNames, dtype=str)
    mace = predictMaceSof(
        prepared,
        testNames=names,
        caseFold=caseFold,
        maceExpert=maceExpert,
    )
    rf = predictRandomForest(prepared)
    missing = np.asarray([not value.strip() for value in names], dtype=bool)
    probabilities = mace["probabilities"].copy()
    probabilities[missing] = rf["probabilities"][missing]
    return {
        "predictions": probabilities.argmax(axis=1),
        "probabilities": probabilities,
        "fallbackMask": missing,
    }


# protect only empty names with an explicit RF fallback so the learned gate has a fair rule baseline
def predictMissingNameRule(
    prepared: dict[str, Any],
    testNames: np.ndarray | None = None,
    maceExpert: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _missingNameFallback(prepared, testNames, False, maceExpert)


# apply the same RF fallback to the case-folded MACE expert that RG-MACE uses
# this isolates the learned gate because case folding and the missing-name rule are then shared by both
def predictCaseFoldedMissingNameRule(
    prepared: dict[str, Any],
    testNames: np.ndarray | None = None,
    maceExpert: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _missingNameFallback(prepared, testNames, True, maceExpert)


# train a generic stacked generalisation baseline on five out-of-fold probability matrices
class SoftStackingReference:
    # store the training-only regularisation grid and deterministic validation protocol
    def __init__(
        self,
        cGrid: list[float] | None = None,
        innerFolds: int = 5,
        seed: int = 100,
    ):
        self.cGrid = cGrid or list(PROPOSED_CONFIG["gateCGrid"])
        self.innerFolds = innerFolds
        self.seed = seed

    # fit a generic stacked model on case-folded five-model out-of-fold probabilities
    def fit(self, prepared: dict[str, Any]) -> "SoftStackingReference":
        self.prepared_ = prepared
        self.expert_ = buildMaceExpert(prepared, caseFold=True)
        self.pool_ = [specification.name for specification in prepared["specifications"]]
        temporary = {**prepared, "foldCache": self.expert_["foldCache"]}
        trainingBlocks = [outOfFoldMatrix(temporary, name) for name in self.pool_]
        X = np.hstack(trainingBlocks)
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, random_state=self.seed)),
            ]
        )
        search = GridSearchCV(
            pipeline,
            {"model__C": sorted(self.cGrid)},
            scoring="accuracy",
            cv=StratifiedKFold(self.innerFolds, shuffle=False),
            n_jobs=1,
            refit=True,
        )
        search.fit(X, prepared["yTrain"])
        self.model_ = search.best_estimator_
        self.chosenC_ = float(search.best_params_["model__C"])
        self.selectionTable_ = pd.DataFrame(search.cv_results_)[
            ["param_model__C", "mean_test_score", "std_test_score", "rank_test_score"]
        ].rename(
            columns={
                "param_model__C": "C",
                "mean_test_score": "MeanCVAccuracy",
                "std_test_score": "CVAccuracyStd",
                "rank_test_score": "Rank",
            }
        )
        return self

    # apply the fitted stacker after recomputing only the name-dependent test block
    def predict(self, testNames: np.ndarray | None = None) -> dict[str, Any]:
        names = (
            self.prepared_["testInputs"]["metadata"]
            if testNames is None
            else np.asarray(testNames, dtype=str)
        )
        cache = {name: values.copy() for name, values in self.expert_["testCache"].items()}
        ddl, _ = _ddlOutputs(self.expert_["fullDdlModel"], names, self.prepared_["classes"])
        cache["DDL-NLP"] = ddl
        X = np.hstack([cache[name] for name in self.pool_])
        raw = self.model_.predict_proba(X)
        probabilities = alignProbabilities(raw, self.model_.classes_, self.prepared_["classes"])
        return {"predictions": probabilities.argmax(axis=1), "probabilities": probabilities}


# train and apply the proposed per-stream reliability-gated soft mixture
class ReliabilityGatedMace:
    # store every fixed design choice and the one training-only tuning grid
    def __init__(
        self,
        caseFold: bool = True,
        corruptionShare: float = 0.5,
        modes: list[str] | None = None,
        cGrid: list[float] | None = None,
        innerFolds: int = 5,
        cleanToleranceF1: float = 0.5,
        seed: int = 100,
        featureSubset: list[str] | None = None,
        useClassBalance: bool = True,
        useCorruption: bool = True,
        hardGate: bool = False,
        selectionMetric: str = "F1",
    ):
        self.caseFold = caseFold
        self.corruptionShare = corruptionShare
        self.modes = modes or list(PROPOSED_CONFIG["corruptionModes"])
        self.cGrid = cGrid or list(PROPOSED_CONFIG["gateCGrid"])
        self.innerFolds = innerFolds
        self.cleanToleranceF1 = cleanToleranceF1
        self.seed = seed
        self.featureSubset = featureSubset
        self.useClassBalance = useClassBalance
        self.useCorruption = useCorruption
        self.hardGate = hardGate
        self.selectionMetric = selectionMetric

    # build the case-aware MACE expert leakage-free gate data and final selected gate
    def fit(self, prepared: dict[str, Any]) -> "ReliabilityGatedMace":
        self.prepared_ = prepared
        started = time.perf_counter()
        self.maceExpert_ = buildMaceExpert(prepared, self.caseFold)
        self.maceExpertSeconds_ = time.perf_counter() - started
        started = time.perf_counter()
        self.gateDataset_ = buildGateDataset(
            prepared,
            self.maceExpert_,
            self.caseFold,
            self.corruptionShare if self.useCorruption else 0.0,
            self.modes,
            self.seed,
        )
        self.gateDatasetSeconds_ = time.perf_counter() - started
        started = time.perf_counter()
        self.selection_ = selectGate(
            self.gateDataset_,
            self.cGrid,
            self.innerFolds,
            self.cleanToleranceF1,
            self.seed,
            self.featureSubset,
            self.useClassBalance,
            self.selectionMetric,
        )
        self.chosenC_ = self.selection_["chosenC"]
        self.gate_ = fitGate(
            self.gateDataset_,
            self.chosenC_,
            self.seed,
            self.featureSubset,
            self.useClassBalance,
        )
        self.gateSelectionAndFitSeconds_ = time.perf_counter() - started
        featureNames = self.featureSubset or RELIABILITY_FEATURES
        coefficients = self.gate_.named_steps["model"].coef_.reshape(-1)
        self.gateCoefficients_ = pd.DataFrame(
            {"Feature": featureNames, "Coefficient": coefficients, "AbsoluteCoefficient": np.abs(coefficients)}
        ).sort_values("AbsoluteCoefficient", ascending=False, ignore_index=True)
        return self

    # produce per-stream reliability weights and hard or soft expert fusion
    def predict(
        self,
        prepared: dict[str, Any] | None = None,
        testNames: np.ndarray | None = None,
    ) -> dict[str, Any]:
        prepared = prepared or self.prepared_
        names = prepared["testInputs"]["metadata"] if testNames is None else np.asarray(testNames, dtype=str)
        started = time.perf_counter()
        ddl, distances = _ddlOutputs(self.maceExpert_["fullDdlModel"], names, prepared["classes"])
        cache = {name: values.copy() for name, values in self.maceExpert_["testCache"].items()}
        cache["DDL-NLP"] = ddl
        _, mace = cascade(
            [cache[name] for name in self.maceExpert_["order"]],
            prepared["classes"].size,
            "sof",
            self.maceExpert_["sigma"],
        )
        rf = prepared["testCache"]["RandomForest"]
        features = reliabilityFeatures(
            names,
            ddl,
            distances,
            rf,
            mace,
            self.gateDataset_["trainNameLengthMedian"],
        )
        gate = self.gate_.predict_proba(features[self.gate_.featureNames_])
        gateClasses = self.gate_.named_steps["model"].classes_
        gate = gate[:, int(np.flatnonzero(gateClasses == 1)[0])]
        if self.hardGate:
            useMace = gate >= 0.5
            probabilities = rf.copy()
            probabilities[useMace] = mace[useMace]
        else:
            probabilities = mixtureProbabilities(gate, mace, rf)
        elapsed = time.perf_counter() - started
        return {
            "predictions": probabilities.argmax(axis=1),
            "probabilities": probabilities,
            "gate": gate,
            "maceProbabilities": mace,
            "rfProbabilities": rf,
            "features": features,
            "predictionSeconds": elapsed,
        }


# report every task metric and calibration score from one common prediction object
def evaluatePredictions(
    prepared: dict[str, Any],
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    metrics = percentageMetrics(
        prepared["yTest"],
        np.asarray(predictions, dtype=int),
        np.asarray(probabilities, dtype=float),
        prepared["classes"],
    )
    metrics["BrierScore"] = multiclassBrierScore(
        prepared["yTest"],
        probabilities,
        prepared["classes"],
    )
    return metrics
