# builds leakage-aware reliability evidence and the regularised RG-MACE gate
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cascadeEngine import cascade, selectFromTopKAccuracy, selectOrder, topKFromFolds, tuneParameter
from .config import loadConfig
from .ddlNLP import DDLNLPClassifier, damerauLevenshtein, unrestrictedDamerauLevenshtein
from .evaluation import alignProbabilities
from .reproductionPipeline import classPositions
from .robustness import corruptMetadata


NAME_DISTANCE_FEATURES = [
    "nameMissing",
    "exactNameSeen",
    "nearestNameDistance",
    "nameDistanceGap",
    "nameLengthLogRatio",
]

CONFIDENCE_FEATURES = [
    "ddlMaxProbability",
    "ddlMargin",
    "ddlEntropy",
    "rfMaxProbability",
    "rfMargin",
    "rfEntropy",
    "maceMaxProbability",
    "maceMargin",
    "maceSurvivingClassShare",
    "expertsAgree",
    "maceSupportForRfChoice",
    "rfSupportForMaceChoice",
]

RELIABILITY_FEATURES = NAME_DISTANCE_FEATURES + CONFIDENCE_FEATURES


# audit whether case folding merges names and whether any merge crosses class labels
def caseFoldCollisionAudit(names: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    original = np.asarray(names, dtype=str)
    folded = np.asarray([value.casefold() for value in original], dtype=str)
    frame = pd.DataFrame({"Original": original, "Folded": folded, "Label": labels})
    crossClass = frame.groupby("Folded")["Label"].nunique().gt(1)
    return pd.DataFrame(
        [
            {
                "Rows": original.size,
                "UniqueNamesBefore": np.unique(original).size,
                "UniqueNamesAfter": np.unique(folded).size,
                "NamesMergedByCaseFold": np.unique(original).size - np.unique(folded).size,
                "CrossClassFoldedConflicts": int(crossClass.sum()),
            }
        ]
    )


# extend the paper classifier with reusable nearest-class distance evidence
class ReliabilityDDL(DDLNLPClassifier):
    # prepare names with Unicode-aware case folding when requested
    def _prepare(self, values: np.ndarray) -> np.ndarray:
        prepared = np.asarray(values, dtype=str).reshape(-1)
        if not self.caseSensitive:
            prepared = np.asarray([value.casefold() for value in prepared], dtype=str)
        return prepared

    # fit the released classifier state and reset the per-query distance cache
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ReliabilityDDL":
        super().fit(X, y)
        self.distanceCache_: dict[str, np.ndarray] = {}
        return self

    # return nearest length-normalised metadata distance for every class
    def nearestDistances(self, X: np.ndarray) -> np.ndarray:
        queries = self._prepare(X)
        distances = np.empty((queries.size, self.classes_.size), dtype=float)
        distanceFunction = (
            damerauLevenshtein if self.distanceVariant == "restricted" else unrestrictedDamerauLevenshtein
        )
        for rowIndex, query in enumerate(queries):
            if query not in self.distanceCache_:
                classDistances = np.empty(self.classes_.size, dtype=float)
                for classIndex, label in enumerate(self.classes_):
                    candidates = self.metadata_[self.labels_ == label]
                    candidateDistances = []
                    for candidate in candidates:
                        distance = float(distanceFunction(query, candidate))
                        if self.normaliseLength:
                            distance /= max(len(query), len(candidate), 1)
                        candidateDistances.append(distance)
                    classDistances[classIndex] = min(candidateDistances) if candidateDistances else 1000.0
                self.distanceCache_[query] = classDistances
            distances[rowIndex] = self.distanceCache_[query]
        return distances

    # convert cached nearest distances through the released inverse-square scoring rule
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        distances = self.nearestDistances(X)
        scores = 1.0 / ((distances + self.epsilon) ** self.distancePower)
        return scores / scores.sum(axis=1, keepdims=True)


# reproduce the exact held-out fold indices used by the Part 1 cache
def foldIndices(prepared: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = loadConfig()["selection"]["cvFolds"]
    splitter = StratifiedKFold(n_splits=folds, shuffle=False)
    return [
        (np.asarray(fitIndex), np.asarray(heldOutIndex))
        for fitIndex, heldOutIndex in splitter.split(np.zeros(prepared["yTrain"].size), prepared["yTrain"])
    ]


# restore one fold cache to the original training-row order and verify its labels
def outOfFoldMatrix(prepared: dict[str, Any], modelName: str) -> np.ndarray:
    classCount = prepared["classes"].size
    matrix = np.empty((prepared["yTrain"].size, classCount), dtype=float)
    coverage = np.zeros(prepared["yTrain"].size, dtype=int)
    for foldNumber, (_, heldOutIndex) in enumerate(foldIndices(prepared)):
        probabilities, storedLabels = prepared["foldCache"][modelName][foldNumber]
        expectedLabels = classPositions(prepared["yTrain"][heldOutIndex], prepared["classes"])
        if not np.array_equal(storedLabels, expectedLabels):
            raise AssertionError(f"Stored labels do not match fold {foldNumber} for {modelName}")
        matrix[heldOutIndex] = probabilities
        coverage[heldOutIndex] += 1
    if not np.all(coverage == 1):
        raise AssertionError("Every training row must appear in exactly one held-out fold")
    return matrix


# align classwise distances while treating an absent class as maximally distant
def _alignDistances(
    distances: np.ndarray,
    sourceClasses: np.ndarray,
    targetClasses: np.ndarray,
) -> np.ndarray:
    values = np.asarray(distances, dtype=float)
    aligned = np.full((values.shape[0], len(targetClasses)), 1.0, dtype=float)
    targetLookup = {label: index for index, label in enumerate(targetClasses)}
    for sourceIndex, label in enumerate(sourceClasses):
        if label in targetLookup:
            aligned[:, targetLookup[label]] = values[:, sourceIndex]
    return aligned


# copy the released DDL settings into the Part 2 reliability-aware subclass
def _reliabilityDdlFromPrepared(prepared: dict[str, Any], caseFold: bool) -> ReliabilityDDL:
    source = prepared["fittedModels"]["DDL-NLP"]
    return ReliabilityDDL(
        distancePower=source.distancePower,
        caseSensitive=not caseFold,
        normaliseLength=source.normaliseLength,
        epsilon=source.epsilon,
        distanceVariant=source.distanceVariant,
    )


# rebuild the ThingSpeak MACE expert with optional case folding and training-only reselection
def buildMaceExpert(prepared: dict[str, Any], caseFold: bool) -> dict[str, Any]:
    if prepared["bundle"].key != "thingspeak":
        raise ValueError("The reliability gate requires the multi-domain ThingSpeak dataset")
    classCount = prepared["classes"].size
    foldCache = {name: list(values) for name, values in prepared["foldCache"].items()}
    testCache = {name: values.copy() for name, values in prepared["testCache"].items()}
    foldModels: list[ReliabilityDDL] = []
    oofDistances = np.empty((prepared["yTrain"].size, classCount), dtype=float)

    if caseFold:
        rebuiltFolds = []
        for fitIndex, heldOutIndex in foldIndices(prepared):
            model = _reliabilityDdlFromPrepared(prepared, caseFold=True)
            model.fit(prepared["trainInputs"]["metadata"][fitIndex], prepared["yTrain"][fitIndex])
            if not np.array_equal(model.classes_, prepared["classes"]):
                raise AssertionError("Every fold DDL model must observe every target class")
            raw = model.predict_proba(prepared["trainInputs"]["metadata"][heldOutIndex])
            probabilities = alignProbabilities(raw, model.classes_, prepared["classes"])
            distances = _alignDistances(
                model.nearestDistances(prepared["trainInputs"]["metadata"][heldOutIndex]),
                model.classes_,
                prepared["classes"],
            )
            labels = classPositions(prepared["yTrain"][heldOutIndex], prepared["classes"])
            rebuiltFolds.append((probabilities, labels))
            oofDistances[heldOutIndex] = distances
            foldModels.append(model)
        foldCache["DDL-NLP"] = rebuiltFolds
        fullModel = _reliabilityDdlFromPrepared(prepared, caseFold=True)
        fullModel.fit(prepared["trainInputs"]["metadata"], prepared["yTrain"])
        rawTest = fullModel.predict_proba(prepared["testInputs"]["metadata"])
        testCache["DDL-NLP"] = alignProbabilities(rawTest, fullModel.classes_, prepared["classes"])
    else:
        for fitIndex, heldOutIndex in foldIndices(prepared):
            model = _reliabilityDdlFromPrepared(prepared, caseFold=False)
            model.fit(prepared["trainInputs"]["metadata"][fitIndex], prepared["yTrain"][fitIndex])
            if not np.array_equal(model.classes_, prepared["classes"]):
                raise AssertionError("Every fold DDL model must observe every target class")
            distances = _alignDistances(
                model.nearestDistances(prepared["trainInputs"]["metadata"][heldOutIndex]),
                model.classes_,
                prepared["classes"],
            )
            oofDistances[heldOutIndex] = distances
            foldModels.append(model)
        fullModel = _reliabilityDdlFromPrepared(prepared, caseFold=False)
        fullModel.fit(prepared["trainInputs"]["metadata"], prepared["yTrain"])

    pool = [specification.name for specification in prepared["specifications"]]
    curves = [topKFromFolds(foldCache[name], classCount) for name in pool]
    selected, rawTopK = selectFromTopKAccuracy(curves, classCount)
    order, topK = selectOrder(selected, pool, rawTopK)
    sigma, sigmaCurve = tuneParameter(
        order,
        foldCache,
        classCount,
        "sof",
        loadConfig()["selection"]["sigmaValues"],
    )
    return {
        "caseFold": caseFold,
        "order": order,
        "topK": topK,
        "sigma": sigma,
        "sigmaCurve": sigmaCurve,
        "foldCache": foldCache,
        "testCache": testCache,
        "foldDdlModels": foldModels,
        "fullDdlModel": fullModel,
        "oofDdlDistances": oofDistances,
    }


# apply the unchanged Part 1 SoF cascade to one ordered set of expert scores
def maceExpertProbabilities(
    orderedProbabilities: list[np.ndarray],
    classCount: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    return cascade(orderedProbabilities, classCount, "sof", sigma)


# calculate maximum probability margin and normalised entropy for one expert
def _confidence(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = np.sort(probabilities, axis=1)
    maximum = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    clipped = np.clip(probabilities, 1.0e-15, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1) / np.log(probabilities.shape[1])
    return maximum, margin, entropy


# transform metadata and expert outputs into finite per-stream reliability features
def reliabilityFeatures(
    namesPrepared: np.ndarray,
    ddlProbabilities: np.ndarray,
    ddlDistances: np.ndarray,
    rfProbabilities: np.ndarray,
    maceProbabilities: np.ndarray,
    trainNameLengthMedian: float,
) -> pd.DataFrame:
    names = np.asarray(namesPrepared, dtype=str).reshape(-1)
    orderedDistances = np.sort(ddlDistances, axis=1)
    ddlMaximum, ddlMargin, ddlEntropy = _confidence(ddlProbabilities)
    rfMaximum, rfMargin, rfEntropy = _confidence(rfProbabilities)
    maceMaximum, maceMargin, _ = _confidence(maceProbabilities)
    rfChoice = rfProbabilities.argmax(axis=1)
    maceChoice = maceProbabilities.argmax(axis=1)
    rows = np.arange(names.size)
    table = pd.DataFrame(
        {
            "nameMissing": np.asarray([not value.strip() for value in names], dtype=float),
            "exactNameSeen": (orderedDistances[:, 0] == 0.0).astype(float),
            "nearestNameDistance": orderedDistances[:, 0],
            "nameDistanceGap": orderedDistances[:, 1] - orderedDistances[:, 0],
            "nameLengthLogRatio": np.log(
                (np.asarray([len(value) for value in names], dtype=float) + 1.0)
                / (float(trainNameLengthMedian) + 1.0)
            ),
            "ddlMaxProbability": ddlMaximum,
            "ddlMargin": ddlMargin,
            "ddlEntropy": ddlEntropy,
            "rfMaxProbability": rfMaximum,
            "rfMargin": rfMargin,
            "rfEntropy": rfEntropy,
            "maceMaxProbability": maceMaximum,
            "maceMargin": maceMargin,
            "maceSurvivingClassShare": np.mean(maceProbabilities > 0.0, axis=1),
            "expertsAgree": (maceChoice == rfChoice).astype(float),
            "maceSupportForRfChoice": maceProbabilities[rows, rfChoice],
            "rfSupportForMaceChoice": rfProbabilities[rows, maceChoice],
        },
        columns=RELIABILITY_FEATURES,
    )
    if not np.isfinite(table.to_numpy(float)).all():
        raise ValueError("Reliability features must all be finite")
    return table


# restore all pool scores to training order for corruption-aware gate construction
def _outOfFoldCache(prepared: dict[str, Any], maceExpert: dict[str, Any]) -> dict[str, np.ndarray]:
    matrices = {}
    for name in maceExpert["foldCache"]:
        source = {**prepared, "foldCache": maceExpert["foldCache"]}
        matrices[name] = outOfFoldMatrix(source, name)
    return matrices


# calculate balanced binary routing weights without dropping tied rows from evaluation
def _routingWeights(targets: np.ndarray, gaps: np.ndarray, fitMask: np.ndarray) -> np.ndarray:
    weights = np.asarray(gaps, dtype=float).copy()
    fittedTargets = targets[fitMask]
    for value in (0, 1):
        count = int(np.sum(fittedTargets == value))
        if count:
            weights[(targets == value) & fitMask] *= fittedTargets.size / (2.0 * count)
    weights[~fitMask] = 0.0
    return weights


# derive binary better-expert targets tie exclusions and balanced support-gap weights
def routingTargets(
    maceProbabilities: np.ndarray,
    rfProbabilities: np.ndarray,
    truePositions: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.arange(len(truePositions))
    pMace = np.asarray(maceProbabilities)[rows, truePositions]
    pRf = np.asarray(rfProbabilities)[rows, truePositions]
    gaps = np.abs(pMace - pRf)
    fitMask = gaps > 1.0e-15
    targets = np.full(len(truePositions), -1, dtype=int)
    targets[fitMask] = (pMace[fitMask] > pRf[fitMask]).astype(int)
    return {
        "targets": targets,
        "fitMask": fitMask,
        "probabilityGap": gaps,
        "sampleWeights": _routingWeights(targets, gaps, fitMask),
    }


# build the leakage-free out-of-fold gate data and controlled metadata-failure copies
def buildGateDataset(
    prepared: dict[str, Any],
    maceExpert: dict[str, Any],
    caseFold: bool,
    corruptionShare: float,
    modes: list[str],
    seed: int,
) -> dict[str, Any]:
    if not 0.0 <= corruptionShare <= 1.0:
        raise ValueError("corruptionShare must be between zero and one")
    names = np.asarray(prepared["trainInputs"]["metadata"], dtype=str)
    originalNames = names.copy()
    classCount = prepared["classes"].size
    oofCache = _outOfFoldCache(prepared, maceExpert)
    ddl = oofCache["DDL-NLP"]
    rf = oofCache["RandomForest"]
    _, mace = maceExpertProbabilities(
        [oofCache[name] for name in maceExpert["order"]],
        classCount,
        maceExpert["sigma"],
    )
    distances = maceExpert["oofDdlDistances"]
    nameLengthMedian = float(np.median([len(value) for value in names]))
    cleanFeatures = reliabilityFeatures(names, ddl, distances, rf, mace, nameLengthMedian)
    trueLabels = np.asarray(prepared["yTrain"])
    truePositions = classPositions(trueLabels, prepared["classes"])

    featureTables = [cleanFeatures]
    maceTables = [mace]
    rfTables = [rf]
    ddlTables = [ddl]
    distanceTables = [distances]
    labelTables = [trueLabels]
    positionTables = [truePositions]
    groups = [np.arange(names.size)]
    corruptedFlags = [np.zeros(names.size, dtype=bool)]
    corruptionLabels = [np.full(names.size, "clean", dtype=object)]
    sourceRows = [np.arange(names.size)]
    corruptedNames = [names.copy()]

    copyCount = int(round(corruptionShare * names.size))
    if copyCount:
        rng = np.random.default_rng(seed)
        selectedRows = rng.choice(names.size, size=copyCount, replace=False)
        assignedModes = rng.choice(np.asarray(modes, dtype=object), size=copyCount, replace=True)
        heldOutFold = np.empty(names.size, dtype=int)
        for foldNumber, (_, heldOutIndex) in enumerate(foldIndices(prepared)):
            heldOutFold[heldOutIndex] = foldNumber
        copiedDdl = np.empty((copyCount, classCount), dtype=float)
        copiedDistances = np.empty((copyCount, classCount), dtype=float)
        copiedMace = np.empty((copyCount, classCount), dtype=float)
        copiedNames = np.empty(copyCount, dtype=object)
        for copyIndex, (rowIndex, mode) in enumerate(zip(selectedRows, assignedModes)):
            rowSeed = int(seed * 100000 + rowIndex * 17 + copyIndex)
            changed = corruptMetadata(np.asarray([names[rowIndex]]), 1.0, rowSeed, str(mode))[0]
            copiedNames[copyIndex] = changed
            model = maceExpert["foldDdlModels"][heldOutFold[rowIndex]]
            rawDdl = model.predict_proba(np.asarray([changed]))
            copiedDdl[copyIndex] = alignProbabilities(rawDdl, model.classes_, prepared["classes"])[0]
            rawDistances = model.nearestDistances(np.asarray([changed]))
            copiedDistances[copyIndex] = _alignDistances(
                rawDistances,
                model.classes_,
                prepared["classes"],
            )[0]
            stages = []
            for modelName in maceExpert["order"]:
                stage = copiedDdl[copyIndex : copyIndex + 1] if modelName == "DDL-NLP" else oofCache[modelName][rowIndex : rowIndex + 1]
                stages.append(stage)
            copiedMace[copyIndex] = maceExpertProbabilities(stages, classCount, maceExpert["sigma"])[1][0]
        copiedRf = rf[selectedRows]
        copiedFeatures = reliabilityFeatures(
            copiedNames,
            copiedDdl,
            copiedDistances,
            copiedRf,
            copiedMace,
            nameLengthMedian,
        )
        featureTables.append(copiedFeatures)
        maceTables.append(copiedMace)
        rfTables.append(copiedRf)
        ddlTables.append(copiedDdl)
        distanceTables.append(copiedDistances)
        labelTables.append(trueLabels[selectedRows])
        positionTables.append(truePositions[selectedRows])
        groups.append(selectedRows)
        corruptedFlags.append(np.ones(copyCount, dtype=bool))
        corruptionLabels.append(assignedModes.astype(object))
        sourceRows.append(selectedRows)
        corruptedNames.append(copiedNames.astype(str))

    features = pd.concat(featureTables, ignore_index=True)
    maceProbabilities = np.vstack(maceTables)
    rfProbabilities = np.vstack(rfTables)
    ddlProbabilities = np.vstack(ddlTables)
    ddlDistances = np.vstack(distanceTables)
    labels = np.concatenate(labelTables)
    positions = np.concatenate(positionTables)
    groupValues = np.concatenate(groups)
    isCorrupted = np.concatenate(corruptedFlags)
    corruptionMode = np.concatenate(corruptionLabels)
    sourceRow = np.concatenate(sourceRows)
    namesUsed = np.concatenate(corruptedNames).astype(str)
    routing = routingTargets(maceProbabilities, rfProbabilities, positions)

    if not np.array_equal(names, originalNames):
        raise AssertionError("Gate construction modified the original metadata")
    return {
        "features": features,
        "targets": routing["targets"],
        "fitMask": routing["fitMask"],
        "probabilityGap": routing["probabilityGap"],
        "sampleWeights": routing["sampleWeights"],
        "groups": groupValues,
        "isCorrupted": isCorrupted,
        "corruptionMode": corruptionMode,
        "sourceRow": sourceRow,
        "names": namesUsed,
        "trueLabels": labels,
        "truePositions": positions,
        "maceProbabilities": maceProbabilities,
        "rfProbabilities": rfProbabilities,
        "ddlProbabilities": ddlProbabilities,
        "ddlDistances": ddlDistances,
        "classes": prepared["classes"],
        "trainNameLengthMedian": nameLengthMedian,
        "originalMetadataUnchanged": bool(np.array_equal(names, originalNames)),
    }


# construct the scaled regularised logistic reliability gate
def gatePipeline(C: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=5000, random_state=seed)),
        ]
    )


# blend both experts without explicitly eliminating any class
def mixtureProbabilities(
    gateProbability: np.ndarray,
    maceProbabilities: np.ndarray,
    rfProbabilities: np.ndarray,
) -> np.ndarray:
    gate = np.asarray(gateProbability, dtype=float).reshape(-1, 1)
    mixture = gate * np.asarray(maceProbabilities, dtype=float) + (1.0 - gate) * np.asarray(rfProbabilities, dtype=float)
    totals = mixture.sum(axis=1, keepdims=True)
    return np.divide(mixture, totals, out=np.zeros_like(mixture), where=totals > 0.0)


# extract the probability assigned to routing through MACE from a fitted binary gate
def _maceGateProbability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = model.named_steps["model"].classes_
    location = int(np.flatnonzero(classes == 1)[0])
    return probabilities[:, location]


# fit one gate using only non-tied routing targets while retaining ties for mixture evaluation
def fitGate(
    gateDataset: dict[str, Any],
    C: float,
    seed: int,
    featureSubset: list[str] | None = None,
    useClassBalance: bool = True,
) -> Pipeline:
    columns = featureSubset or RELIABILITY_FEATURES
    mask = np.asarray(gateDataset["fitMask"], dtype=bool)
    model = gatePipeline(C, seed)
    weights = (
        gateDataset["sampleWeights"] if useClassBalance else gateDataset["probabilityGap"]
    )
    model.fit(
        gateDataset["features"].loc[mask, columns],
        gateDataset["targets"][mask],
        model__sample_weight=np.asarray(weights)[mask],
    )
    model.featureNames_ = list(columns)
    return model


# select gate regularisation by grouped out-of-fold mixture performance on clean and corrupted rows
def selectGate(
    gateDataset: dict[str, Any],
    cGrid: list[float],
    innerFolds: int,
    cleanToleranceF1: float,
    seed: int,
    featureSubset: list[str] | None = None,
    useClassBalance: bool = True,
    selectionMetric: str = "F1",
) -> dict[str, Any]:
    if selectionMetric not in {"F1", "Accuracy"}:
        raise ValueError("selectionMetric must be 'F1' or 'Accuracy'")
    columns = featureSubset or RELIABILITY_FEATURES
    targets = np.asarray(gateDataset["targets"])
    groups = np.asarray(gateDataset["groups"])
    fitMask = np.asarray(gateDataset["fitMask"], dtype=bool)
    splitTargets = np.where(fitMask, targets, 0)
    splitter = StratifiedGroupKFold(n_splits=innerFolds, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(targets.size), splitTargets, groups))
    rows = []
    predictionsByC: dict[float, dict[str, np.ndarray]] = {}
    for C in sorted(cGrid):
        heldOutGate = np.full(targets.size, np.nan, dtype=float)
        for fitIndex, heldOutIndex in splits:
            usableFit = fitIndex[fitMask[fitIndex]]
            model = gatePipeline(C, seed)
            baseWeights = (
                gateDataset["sampleWeights"] if useClassBalance else gateDataset["probabilityGap"]
            )
            model.fit(
                gateDataset["features"].iloc[usableFit][columns],
                targets[usableFit],
                model__sample_weight=np.asarray(baseWeights)[usableFit],
            )
            heldOutGate[heldOutIndex] = _maceGateProbability(
                model,
                gateDataset["features"].iloc[heldOutIndex][columns],
            )
        mixture = mixtureProbabilities(
            heldOutGate,
            gateDataset["maceProbabilities"],
            gateDataset["rfProbabilities"],
        )
        mixturePredictions = mixture.argmax(axis=1)
        macePredictions = gateDataset["maceProbabilities"].argmax(axis=1)
        clean = ~np.asarray(gateDataset["isCorrupted"], dtype=bool)
        corrupted = ~clean
        labels = np.arange(gateDataset["classes"].size)
        cleanF1 = f1_score(
            gateDataset["truePositions"][clean],
            mixturePredictions[clean],
            labels=labels,
            average="macro",
            zero_division=0,
        ) * 100.0
        corruptedF1 = (
            f1_score(
                gateDataset["truePositions"][corrupted],
                mixturePredictions[corrupted],
                labels=labels,
                average="macro",
                zero_division=0,
            )
            * 100.0
            if np.any(corrupted)
            else cleanF1
        )
        maceCleanF1 = f1_score(
            gateDataset["truePositions"][clean],
            macePredictions[clean],
            labels=labels,
            average="macro",
            zero_division=0,
        ) * 100.0
        cleanAccuracy = float(np.mean(mixturePredictions[clean] == gateDataset["truePositions"][clean]) * 100.0)
        corruptedAccuracy = (
            float(np.mean(mixturePredictions[corrupted] == gateDataset["truePositions"][corrupted]) * 100.0)
            if np.any(corrupted)
            else cleanAccuracy
        )
        objective = (
            (cleanF1 + corruptedF1) / 2.0
            if selectionMetric == "F1"
            else (cleanAccuracy + corruptedAccuracy) / 2.0
        )
        constraintMet = cleanF1 >= maceCleanF1 - cleanToleranceF1
        rows.append(
            {
                "C": C,
                "CleanF1": cleanF1,
                "CorruptedF1": corruptedF1,
                "CleanAccuracy": cleanAccuracy,
                "CorruptedAccuracy": corruptedAccuracy,
                "MaceCleanF1": maceCleanF1,
                "MinimumAllowedCleanF1": maceCleanF1 - cleanToleranceF1,
                "Objective": objective,
                "SelectionMetric": selectionMetric,
                "ConstraintMet": constraintMet,
            }
        )
        predictionsByC[C] = {
            "gate": heldOutGate,
            "probabilities": mixture,
            "predictions": mixturePredictions,
        }
    selection = pd.DataFrame(rows).sort_values("C").reset_index(drop=True)
    eligible = selection.loc[selection["ConstraintMet"]]
    if eligible.empty:
        chosenRow = selection.sort_values(["CleanF1", "C"], ascending=[False, True]).iloc[0]
        constraintMet = False
    else:
        chosenRow = eligible.sort_values(["Objective", "C"], ascending=[False, True]).iloc[0]
        constraintMet = True
    chosenC = float(chosenRow["C"])
    chosen = predictionsByC[chosenC]
    routingMask = fitMask & np.isfinite(chosen["gate"])
    routingPrediction = (chosen["gate"][routingMask] >= 0.5).astype(int)
    routingTruth = targets[routingMask]
    diagnostics = {
        "GateROCAUC": float(roc_auc_score(routingTruth, chosen["gate"][routingMask])),
        "GateBalancedAccuracy": float(balanced_accuracy_score(routingTruth, routingPrediction)),
        "GatePrecision": float(precision_score(routingTruth, routingPrediction, zero_division=0)),
        "GateRecall": float(recall_score(routingTruth, routingPrediction, zero_division=0)),
        "GateF1": float(f1_score(routingTruth, routingPrediction, zero_division=0)),
        "ConfusionMatrix": confusion_matrix(routingTruth, routingPrediction, labels=[0, 1]),
    }
    return {
        "table": selection,
        "chosenC": chosenC,
        "constraintMet": constraintMet,
        "outOfFoldGate": chosen["gate"],
        "outOfFoldProbabilities": chosen["probabilities"],
        "outOfFoldPredictions": chosen["predictions"],
        "diagnostics": diagnostics,
    }
