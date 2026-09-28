# coordinates the verified paper reproduction while reusing every fitted model
from __future__ import annotations

from itertools import combinations
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from scipy.stats import percentileofscore

from .cascadeEngine import (
    bruteForceSoF,
    cascade,
    selectFromTopKAccuracy,
    selectOrder,
    softVoting,
    topKFromFolds,
    tuneParameter,
)
from .classifiers import buildBOSClassifiers
from .config import loadConfig
from .data import loadDataset, normaliseDatasetKey, saveSplitManifest, splitDataset
from .evaluation import alignProbabilities, classificationMetrics
from .experiments import modelSpecifications, prepareRepresentations
from .paths import TABLES_DIR, ensureOutputDirectories
from .publishedResults import (
    PAPER_DATASET_ORDER,
    PAPER_DDL_ACCURACY,
    PAPER_FIGURE3B,
    PAPER_HEURISTICS,
    PAPER_MODEL_NAMES,
    PAPER_POOLS,
    PAPER_SHORT_NAMES,
    PAPER_TABLE1,
    PAPER_TABLE2,
)


METHODS = list(PAPER_TABLE1)
STRATEGIES = {"MACE Top-k": "topK", "MACE PF": "pf", "MACE SoF": "sof"}


# map source labels to the probability-column positions used by the cascade
def classPositions(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(classes)}
    return np.asarray([lookup[label] for label in labels], dtype=int)


# fit one pool member once and return aligned test probabilities
def fitProbabilities(
    specification: Any,
    trainInputs: dict[str, np.ndarray],
    yTrain: np.ndarray,
    testInputs: dict[str, np.ndarray],
    classes: np.ndarray,
) -> tuple[Any, np.ndarray]:
    estimator = clone(specification.estimator)
    estimator.fit(trainInputs[specification.domain], yTrain)
    raw = estimator.predict_proba(testInputs[specification.domain])
    return estimator, alignProbabilities(raw, estimator.classes_, classes)


# cache aligned held-out probabilities for the shared five CV folds
def foldProbabilities(
    specification: Any,
    trainInputs: dict[str, np.ndarray],
    yTrain: np.ndarray,
    classes: np.ndarray,
    folds: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=False)
    cached = []
    for fitIndex, validationIndex in splitter.split(np.zeros(yTrain.size), yTrain):
        estimator = clone(specification.estimator)
        estimator.fit(trainInputs[specification.domain][fitIndex], yTrain[fitIndex])
        raw = estimator.predict_proba(trainInputs[specification.domain][validationIndex])
        probabilities = alignProbabilities(raw, estimator.classes_, classes)
        cached.append((probabilities, classPositions(yTrain[validationIndex], classes)))
    return cached


# return task metrics as report-ready percentages
def percentageMetrics(
    yTrue: np.ndarray,
    predictionPositions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    predictions = classes[predictionPositions]
    metrics = classificationMetrics(yTrue, predictions, probabilities, classes)
    return {
        "Accuracy": round(metrics["Accuracy"] * 100.0, 1),
        "Precision": round(metrics["MacroPrecision"] * 100.0, 1),
        "Recall": round(metrics["MacroRecall"] * 100.0, 1),
        "F1": round(metrics["MacroF1"] * 100.0, 1),
        "AUC": round(metrics["MacroROCAUC"] * 100.0, 1),
        "BalancedAccuracy": round(metrics["BalancedAccuracy"] * 100.0, 1),
        "WeightedF1": round(metrics["WeightedF1"] * 100.0, 1),
        "LogLoss": round(metrics["LogLoss"], 4),
    }


# prepare one faithful seed-100 split with reusable model-score caches
# saveManifest is false inside parallel workers because every split shares one manifest file per dataset
def preparePaperDataset(
    datasetName: str,
    seed: int = 100,
    splitStrategy: str = "stratified",
    constantMomentPolicy: str | None = None,
    ddlDistancePower: int = 2,
    ddlDistanceVariant: str = "restricted",
    saveManifest: bool = True,
    jobs: int = -1,
) -> dict[str, Any]:
    config = loadConfig()
    bundle = loadDataset(datasetName)
    split = splitDataset(bundle, config["testSize"], seed, splitStrategy)
    if saveManifest:
        saveSplitManifest(bundle, split)
    trainInputs, testInputs, yTrain, yTest = prepareRepresentations(
        bundle,
        split,
        constantMomentPolicy or config["constantMomentPolicy"],
    )
    specifications = modelSpecifications(
        bundle.key,
        seed=seed,
        ddlDistancePower=ddlDistancePower,
        ddlDistanceVariant=ddlDistanceVariant,
        jobs=jobs,
    )
    expectedPool = PAPER_POOLS[bundle.key]
    lookup = {specification.name: specification for specification in specifications}
    specifications = [lookup[name] for name in expectedPool]
    classes = bundle.classes

    # compute shared CV and test scores once per deterministic classifier
    foldCache = {
        specification.name: foldProbabilities(
            specification,
            trainInputs,
            yTrain,
            classes,
            config["selection"]["cvFolds"],
        )
        for specification in specifications
    }
    fittedModels, testCache = {}, {}
    for specification in specifications:
        estimator, probabilities = fitProbabilities(
            specification,
            trainInputs,
            yTrain,
            testInputs,
            classes,
        )
        fittedModels[specification.name] = estimator
        testCache[specification.name] = probabilities
    prepared = {
        "bundle": bundle,
        "split": split,
        "trainInputs": trainInputs,
        "testInputs": testInputs,
        "yTrain": yTrain,
        "yTest": yTest,
        "yTestPositions": classPositions(yTest, classes),
        "classes": classes,
        "specifications": specifications,
        "foldCache": foldCache,
        "fittedModels": fittedModels,
        "testCache": testCache,

        # record the protocol so a reused preparation can be checked against each request
        "configuration": {
            "datasetName": bundle.key,
            "seed": seed,
            "splitStrategy": splitStrategy,
            "constantMomentPolicy": constantMomentPolicy or config["constantMomentPolicy"],
            "ddlDistancePower": ddlDistancePower,
            "ddlDistanceVariant": ddlDistanceVariant,
        },
    }

    # fit the eight Figure 3b baselines once so reused preparations skip every model fit
    prepared["standaloneBOS"] = runStandalonePaperBaselines(prepared, jobs)
    return prepared


# evaluate the eight Figure 3b BOS baselines with released settings
def runStandalonePaperBaselines(prepared: dict[str, Any], jobs: int = -1) -> pd.DataFrame:
    config = loadConfig()
    rows = []
    for internalName, estimator in buildBOSClassifiers(config["randomSeed"], jobs=jobs).items():
        fitted = clone(estimator).fit(prepared["trainInputs"]["bos"], prepared["yTrain"])
        raw = fitted.predict_proba(prepared["testInputs"]["bos"])
        probabilities = alignProbabilities(raw, fitted.classes_, prepared["classes"])
        predictions = classPositions(fitted.predict(prepared["testInputs"]["bos"]), prepared["classes"])
        paperName = PAPER_MODEL_NAMES[internalName]
        rows.append(
            {
                "Dataset": prepared["bundle"].displayName,
                "Classifier": paperName,
                "PaperAccuracy": PAPER_FIGURE3B[paperName][prepared["bundle"].key],
            **percentageMetrics(
                    prepared["yTest"],
                    predictions,
                    probabilities,
                    prepared["classes"],
                ),
            }
        )
    table = pd.DataFrame(rows)
    table["AccuracyDifference"] = table["Accuracy"] - table["PaperAccuracy"]
    return table


# reshape Figure 3b values into one paper-versus-mine evidence table
def buildFigureThreeBComparison(standaloneTable: pd.DataFrame) -> pd.DataFrame:
    rows = []
    classifierOrder = list(PAPER_FIGURE3B)
    datasetDisplays = {"urban": "UrbanObservatory", "swissex": "Swissex", "thingspeak": "ThingSpeakEU"}
    for classifier in classifierOrder:
        row = {"Classifier": classifier}
        for datasetName in PAPER_DATASET_ORDER:
            displayName = PAPER_SHORT_NAMES[datasetName]
            matched = standaloneTable.loc[
                (standaloneTable["Classifier"] == classifier)
                & (standaloneTable["Dataset"] == datasetDisplays[datasetName])
            ]
            mine = float(matched.iloc[0]["Accuracy"])
            paper = PAPER_FIGURE3B[classifier][datasetName]
            row[f"{displayName} Paper"] = paper
            row[f"{displayName} Mine"] = mine
            row[f"{displayName} Difference"] = round(mine - paper, 1)
        rows.append(row)
    return pd.DataFrame(rows)


# reproduce the complete main comparison for one dataset
def runPaperDataset(
    datasetName: str,
    includeBruteForce: bool = True,
    seed: int = 100,
    splitStrategy: str = "stratified",
    constantMomentPolicy: str | None = None,
    ddlDistancePower: int = 2,
    ddlDistanceVariant: str = "restricted",
    percentageRule: str = "floor",
    earlyExit: bool = False,
    prepared: dict[str, Any] | None = None,
    saveManifest: bool = True,
    jobs: int = -1,
) -> dict[str, Any]:
    config = loadConfig()

    # a supplied preparation skips data preparation and model fitting and reruns only selection onwards
    requested = {
        "datasetName": normaliseDatasetKey(datasetName),
        "seed": seed,
        "splitStrategy": splitStrategy,
        "constantMomentPolicy": constantMomentPolicy or config["constantMomentPolicy"],
        "ddlDistancePower": ddlDistancePower,
        "ddlDistanceVariant": ddlDistanceVariant,
    }
    if prepared is None:
        prepared = preparePaperDataset(
            datasetName,
            seed,
            splitStrategy,
            constantMomentPolicy,
            ddlDistancePower,
            ddlDistanceVariant,
            saveManifest,
            jobs,
        )
    elif prepared.get("configuration") != requested:
        raise ValueError(
            f"Prepared dataset {prepared.get('configuration')} does not match the requested protocol {requested}"
        )
    pool = PAPER_POOLS[prepared["bundle"].key]
    classCount = prepared["classes"].size
    topKCurves = [topKFromFolds(prepared["foldCache"][name], classCount) for name in pool]
    selectedIndices, rawTopKValues = selectFromTopKAccuracy(topKCurves, classCount)
    order, topKValues = selectOrder(selectedIndices, pool, rawTopKValues)
    bestSigma, sigmaCurve = tuneParameter(
        order,
        prepared["foldCache"],
        classCount,
        "sof",
        config["selection"]["sigmaValues"],
        percentageRule,
        earlyExit,
    )
    bestPercentage, percentageCurve = tuneParameter(
        order,
        prepared["foldCache"],
        classCount,
        "pf",
        config["selection"]["qValues"],
        percentageRule,
        earlyExit,
    )

    results: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    standaloneRows = []
    for modelName in pool:
        modelProbabilities = prepared["testCache"][modelName]
        modelPredictions = modelProbabilities.argmax(axis=1)
        standaloneRows.append(
            {
                "Dataset": prepared["bundle"].displayName,
                "Classifier": PAPER_MODEL_NAMES[modelName],
                **percentageMetrics(
                    prepared["yTest"],
                    modelPredictions,
                    modelProbabilities,
                    prepared["classes"],
                ),
            }
        )
    standalone = pd.DataFrame(standaloneRows)
    paperMetricNames = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    bestStandalone = {metric: float(standalone[metric].max()) for metric in paperMetricNames}
    bestStandalone["BalancedAccuracy"] = float(standalone["BalancedAccuracy"].max())
    bestStandalone["WeightedF1"] = float(standalone["WeightedF1"].max())
    bestStandalone["LogLoss"] = float(standalone["LogLoss"].min())
    results["Best standalone"] = bestStandalone
    standaloneOrigins = {
        "Dataset": prepared["bundle"].displayName,
        "AccuracyClassifier": standalone.loc[standalone["Accuracy"].idxmax(), "Classifier"],
        "F1Classifier": standalone.loc[standalone["F1"].idxmax(), "Classifier"],
        "AUCClassifier": standalone.loc[standalone["AUC"].idxmax(), "Classifier"],
    }

    votingPredictions, votingProbabilities = softVoting(
        [prepared["testCache"][name] for name in pool]
    )
    results["Soft voting"] = percentageMetrics(
        prepared["yTest"],
        votingPredictions,
        votingProbabilities,
        prepared["classes"],
    )
    predictions["Soft voting"] = votingPredictions
    probabilities["Soft voting"] = votingProbabilities

    strategyParameters = {
        "MACE Top-k": topKValues,
        "MACE PF": bestPercentage,
        "MACE SoF": bestSigma,
    }
    for method, criterion in STRATEGIES.items():
        methodPredictions, methodProbabilities = cascade(
            [prepared["testCache"][name] for name in order],
            classCount,
            criterion,
            strategyParameters[method],
            percentageRule,
            earlyExit,
        )
        results[method] = percentageMetrics(
            prepared["yTest"],
            methodPredictions,
            methodProbabilities,
            prepared["classes"],
        )
        predictions[method] = methodPredictions
        probabilities[method] = methodProbabilities

    oracle = None
    if includeBruteForce:
        oracle = bruteForceSoF(
            prepared["testCache"],
            pool,
            classCount,
            prepared["yTestPositions"],
            config["selection"]["sigmaValues"],
        )
        results["MACE brute-force (SoF)"] = percentageMetrics(
            prepared["yTest"],
            oracle["predictions"],
            oracle["scores"],
            prepared["classes"],
        )
        predictions["MACE brute-force (SoF)"] = oracle["predictions"]
        probabilities["MACE brute-force (SoF)"] = oracle["scores"]

    resultRows = []
    for method, metrics in results.items():
        resultRows.append({"Dataset": prepared["bundle"].displayName, "Method": method, **metrics})
    metricsTable = pd.DataFrame(resultRows)

    curveRows = []
    for modelName, curve in zip(pool, topKCurves):
        curveRows.append(
            {
                "Dataset": prepared["bundle"].displayName,
                "Classifier": PAPER_MODEL_NAMES[modelName],
                **{f"k={k}": round(value, 1) for k, value in enumerate(curve, start=1)},
            }
        )
    tuningRows = [
        {
            "Dataset": prepared["bundle"].displayName,
            "Strategy": "SoF",
            "Parameter": parameter,
            "CVAccuracy": round(value, 2),
        }
        for parameter, value in zip(config["selection"]["sigmaValues"], sigmaCurve)
    ]
    tuningRows.extend(
        {
            "Dataset": prepared["bundle"].displayName,
            "Strategy": "PF",
            "Parameter": parameter,
            "CVAccuracy": round(value, 2),
        }
        for parameter, value in zip(config["selection"]["qValues"], percentageCurve)
    )

    sameRows = []
    finalModel = order[-1]
    finalPredictions = prepared["testCache"][finalModel].argmax(axis=1)
    for method in STRATEGIES:
        sameRows.append(
            {
                "Dataset": prepared["bundle"].displayName,
                "Method": method,
                "FinalClassifier": PAPER_MODEL_NAMES[finalModel],
                "IdenticalPredictions": round(float(np.mean(predictions[method] == finalPredictions) * 100.0), 1),
                "SamplesChanged": int(np.sum(predictions[method] != finalPredictions)),
            }
        )

    votingRows = []
    subsetAccuracies, subsetF1 = [], []
    for size in range(2, len(pool) + 1):
        for subset in combinations(pool, size):
            subsetPrediction, subsetProbability = softVoting(
                [prepared["testCache"][name] for name in subset]
            )
            subsetMetrics = percentageMetrics(
                prepared["yTest"],
                subsetPrediction,
                subsetProbability,
                prepared["classes"],
            )
            subsetAccuracies.append(subsetMetrics["Accuracy"])
            subsetF1.append(subsetMetrics["F1"])
    shortName = PAPER_SHORT_NAMES[prepared["bundle"].key]
    votingRows.append(
        {
            "Dataset": prepared["bundle"].displayName,
            "SubPools": len(subsetAccuracies),
            "PaperAccuracy": PAPER_TABLE1["Soft voting"][prepared["bundle"].key],
            "FullPoolAccuracy": results["Soft voting"]["Accuracy"],
            "MinimumAccuracy": min(subsetAccuracies),
            "MedianAccuracy": float(np.median(subsetAccuracies)),
            "MaximumAccuracy": max(subsetAccuracies),
            "PaperF1": PAPER_TABLE2["Soft voting"][prepared["bundle"].key],
            "FullPoolF1": results["Soft voting"]["F1"],
            "MinimumF1": min(subsetF1),
            "MedianF1": float(np.median(subsetF1)),
            "MaximumF1": max(subsetF1),
        }
    )

    heuristic = {
        "Dataset": prepared["bundle"].displayName,
        "ShortName": shortName,
        "PaperOrder": " > ".join(PAPER_MODEL_NAMES[name] for name in PAPER_HEURISTICS[prepared["bundle"].key]["order"]),
        "ReproducedOrder": " > ".join(PAPER_MODEL_NAMES[name] for name in order),
        "SameOrder": order == PAPER_HEURISTICS[prepared["bundle"].key]["order"],
        "PaperTopK": PAPER_HEURISTICS[prepared["bundle"].key]["topK"],
        "ReproducedTopK": str(topKValues),
        "BestSigma": bestSigma,
        "BestPercentage": bestPercentage,
        "BruteForceOrder": " > ".join(PAPER_MODEL_NAMES[name] for name in oracle["order"]) if oracle else "Not executed",
        "BruteForceSigma": oracle["sigma"] if oracle else np.nan,
        "BruteForceConfigurations": oracle["configurationsSearched"] if oracle else 0,
        "BruteForceSelectionProtocol": oracle["selectionProtocol"] if oracle else "Not executed",
        "BruteForceValidForRecommendation": oracle["validForRecommendation"] if oracle else False,
    }
    return {
        "prepared": prepared,
        "standaloneBOS": prepared["standaloneBOS"],
        "standalonePool": standalone,
        "standaloneOrigins": pd.DataFrame([standaloneOrigins]),
        "topKCurves": pd.DataFrame(curveRows),
        "tuningCurves": pd.DataFrame(tuningRows),
        "heuristic": pd.DataFrame([heuristic]),
        "metrics": metricsTable,
        "predictions": predictions,
        "probabilities": probabilities,
        "maceVsFinal": pd.DataFrame(sameRows),
        "votingSensitivity": pd.DataFrame(votingRows),
        "oracle": oracle,
        "configuration": {
            "order": order,
            "topK": topKValues,
            "sigma": bestSigma,
            "percentage": bestPercentage,
            "percentageRule": percentageRule,
            "earlyExit": earlyExit,
            "seed": seed,
            "splitStrategy": splitStrategy,
            "constantMomentPolicy": constantMomentPolicy or config["constantMomentPolicy"],
            "ddlDistancePower": ddlDistancePower,
            "ddlDistanceVariant": ddlDistanceVariant,
        },
    }


# build complete paper-versus-reproduced Tables I and II
def buildPublishedComparisons(results: dict[str, dict[str, Any]]) -> dict[str, pd.DataFrame]:
    tables = {}
    for tableName, paperValues, metric in [
        ("accuracy", PAPER_TABLE1, "Accuracy"),
        ("macroF1", PAPER_TABLE2, "F1"),
    ]:
        rows = []
        for method in METHODS:
            row = {"Method": method}
            for datasetName in PAPER_DATASET_ORDER:
                reproduced = results[datasetName]["metrics"].set_index("Method")
                paper = paperValues[method][datasetName]
                mine = float(reproduced.loc[method, metric]) if method in reproduced.index else np.nan
                shortName = PAPER_SHORT_NAMES[datasetName]
                row[f"{shortName} Paper"] = paper
                row[f"{shortName} Mine"] = mine
                row[f"{shortName} Difference"] = round(mine - paper, 1)
            rows.append(row)
        tables[tableName] = pd.DataFrame(rows)
    return tables


# tune PF and SoF for one cascade order and return the test accuracy of each heuristic
def _heuristicAccuracies(
    order: list[str],
    topKValues: list[int],
    prepared: dict[str, Any],
    classCount: int,
    config: dict[str, Any],
) -> dict[str, float]:
    sigma, _ = tuneParameter(
        order,
        prepared["foldCache"],
        classCount,
        "sof",
        config["selection"]["sigmaValues"],
    )
    percentage, _ = tuneParameter(
        order,
        prepared["foldCache"],
        classCount,
        "pf",
        config["selection"]["qValues"],
    )
    accuracies = {}
    for strategyName, criterion, parameter in [
        ("Top-k", "topK", topKValues),
        ("PF", "pf", percentage),
        ("SoF", "sof", sigma),
    ]:
        prediction, _ = cascade(
            [prepared["testCache"][name] for name in order],
            classCount,
            criterion,
            parameter,
        )
        accuracies[strategyName] = float(np.mean(prediction == prepared["yTestPositions"]) * 100.0)
    return accuracies


# reproduce the 26-pool heuristic comparisons from Figures 5 and 6
# also returns the pools whose released order repeats a classifier beside the Eq. 6 order
def runHeuristicPoolAnalysis(datasetResult: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = datasetResult["prepared"]
    pool = PAPER_POOLS[prepared["bundle"].key]
    classCount = prepared["classes"].size
    oracle = datasetResult["oracle"]
    if oracle is None:
        raise ValueError("The 26-pool analysis requires the paper oracle records")
    oracleTable = pd.DataFrame(oracle["records"])
    topKLookup = {
        name: topKFromFolds(prepared["foldCache"][name], classCount)
        for name in pool
    }
    config = loadConfig()
    rows, repeatRows = [], []
    for poolSize in range(2, len(pool) + 1):
        for poolSubset in combinations(pool, poolSize):
            curves = [topKLookup[name] for name in poolSubset]
            selectedIndices, rawTopKValues = selectFromTopKAccuracy(curves, classCount)
            order, topKValues = selectOrder(selectedIndices, list(poolSubset), rawTopKValues)
            verbatimOrder, verbatimTopK = selectOrder(
                selectedIndices,
                list(poolSubset),
                rawTopKValues,
                allowRepeats=True,
            )
            repeatedClassifierRemoved = verbatimOrder != order
            accuracies = _heuristicAccuracies(order, topKValues, prepared, classCount, config)
            subsetSet = set(poolSubset)
            oracleDistribution = oracleTable.loc[
                oracleTable["Order"].apply(lambda value: set(value.split(" > ")).issubset(subsetSet)),
                "Accuracy",
            ].to_numpy(float)
            oracleBest = float(oracleDistribution.max())
            inputPool = " | ".join(PAPER_MODEL_NAMES[name] for name in poolSubset)
            for strategyName, accuracy in accuracies.items():
                normalisedAccuracy = accuracy / oracleBest

                # the oracle searches only SoF configurations so only SoF is bounded by it
                if strategyName == "SoF" and normalisedAccuracy > 1.0 + 1.0e-9:
                    raise AssertionError(
                        f"SoF normalised accuracy {normalisedAccuracy:.6f} exceeds the oracle for {inputPool}"
                    )
                rows.append(
                    {
                        "Dataset": prepared["bundle"].displayName,
                        "InputPool": inputPool,
                        "Heuristic": strategyName,
                        "SelectedOrder": " > ".join(PAPER_MODEL_NAMES[name] for name in order),
                        "RepeatedClassifierRemoved": repeatedClassifierRemoved,
                        "Accuracy": accuracy,
                        "OracleBestAccuracy": oracleBest,
                        "NormalisedAccuracy": normalisedAccuracy,
                        "OraclePercentile": percentileofscore(
                            oracleDistribution,
                            accuracy,
                            kind="weak",
                        ),
                    }
                )
            if repeatedClassifierRemoved:
                verbatimAccuracies = _heuristicAccuracies(verbatimOrder, verbatimTopK, prepared, classCount, config)
                repeatRows.append(
                    {
                        "Dataset": prepared["bundle"].displayName,
                        "InputPool": inputPool,
                        "SelectedOrder": " > ".join(PAPER_MODEL_NAMES[name] for name in order),
                        "VerbatimOrder": " > ".join(PAPER_MODEL_NAMES[name] for name in verbatimOrder),
                        "SelectedTopK": str(topKValues),
                        "VerbatimTopK": str(verbatimTopK),
                        **{f"Selected{name.replace('-', '')}Accuracy": value for name, value in accuracies.items()},
                        **{f"Verbatim{name.replace('-', '')}Accuracy": value for name, value in verbatimAccuracies.items()},
                    }
                )
    repeatColumns = [
        "Dataset",
        "InputPool",
        "SelectedOrder",
        "VerbatimOrder",
        "SelectedTopK",
        "VerbatimTopK",
        "SelectedTopkAccuracy",
        "SelectedPFAccuracy",
        "SelectedSoFAccuracy",
        "VerbatimTopkAccuracy",
        "VerbatimPFAccuracy",
        "VerbatimSoFAccuracy",
    ]
    return pd.DataFrame(rows), pd.DataFrame(repeatRows, columns=repeatColumns)


# run all faithful experiments and save every report-ready table
def runFullPaperReproduction(
    includeBruteForce: bool = True,
    includePoolAnalysis: bool = False,
) -> dict[str, Any]:
    ensureOutputDirectories()
    results, runtimeRows = {}, []
    for datasetName in PAPER_DATASET_ORDER:
        started = time.perf_counter()
        results[datasetName] = runPaperDataset(datasetName, includeBruteForce)
        runtimeRows.append(
            {
                "Dataset": results[datasetName]["prepared"]["bundle"].displayName,
                "RuntimeSeconds": round(time.perf_counter() - started, 2),
                "BruteForceIncluded": includeBruteForce,
            }
        )
    comparisons = buildPublishedComparisons(results)
    standaloneBOS = pd.concat(
        [results[name]["standaloneBOS"] for name in PAPER_DATASET_ORDER],
        ignore_index=True,
    )
    combined = {
        "standaloneBOS": standaloneBOS,
        "figureThreeBComparison": buildFigureThreeBComparison(standaloneBOS),
        "standalonePool": pd.concat([results[name]["standalonePool"] for name in PAPER_DATASET_ORDER], ignore_index=True),
        "standaloneOrigins": pd.concat([results[name]["standaloneOrigins"] for name in PAPER_DATASET_ORDER], ignore_index=True),
        "topKCurves": pd.concat([results[name]["topKCurves"] for name in PAPER_DATASET_ORDER], ignore_index=True),
        "tuningCurves": pd.concat([results[name]["tuningCurves"] for name in PAPER_DATASET_ORDER], ignore_index=True),
        "heuristics": pd.concat([results[name]["heuristic"] for name in PAPER_DATASET_ORDER], ignore_index=True),
        "allMetrics": pd.concat([results[name]["metrics"] for name in PAPER_DATASET_ORDER], ignore_index=True),
        "maceVsFinal": pd.concat([results[name]["maceVsFinal"] for name in PAPER_DATASET_ORDER], ignore_index=True),
        "votingSensitivity": pd.concat([results[name]["votingSensitivity"] for name in PAPER_DATASET_ORDER], ignore_index=True),
        "table1Accuracy": comparisons["accuracy"],
        "table2MacroF1": comparisons["macroF1"],
        "runtime": pd.DataFrame(runtimeRows),
    }
    if includePoolAnalysis:
        poolTables, repeatTables = zip(*[runHeuristicPoolAnalysis(results[name]) for name in PAPER_DATASET_ORDER])
        combined["heuristicPoolAnalysis"] = pd.concat(poolTables, ignore_index=True)

        # saved as CSV and Markdown by the shared table loop below
        combined["heuristicRepeatSensitivity"] = pd.concat(repeatTables, ignore_index=True)
    for name, table in combined.items():
        table.to_csv(TABLES_DIR / f"{name}.csv", index=False)
        (TABLES_DIR / f"{name}.md").write_text(
            table.to_markdown(index=False, floatfmt=".2f"),
            encoding="utf-8",
        )
    return {"datasets": results, **combined}


# explain DDL-NLP performance for seen and unseen test names
def ddlNLPDiagnostic(prepared: dict[str, Any]) -> pd.DataFrame:
    if prepared["bundle"].key != "thingspeak":
        raise ValueError("DDL-NLP is only used with ThingSpeak")
    probabilities = prepared["testCache"]["DDL-NLP"]
    predictions = probabilities.argmax(axis=1)
    trainNames = {str(value).strip().lower() for value in prepared["trainInputs"]["metadata"]}
    seen = np.asarray(
        [str(value).strip().lower() in trainNames for value in prepared["testInputs"]["metadata"]]
    )
    metrics = percentageMetrics(
        prepared["yTest"],
        predictions,
        probabilities,
        prepared["classes"],
    )
    return pd.DataFrame(
        [
            {
                "Classifier": "DDL-NLP",
                "PaperAccuracy": PAPER_DDL_ACCURACY,
                **metrics,
                "TestNamesSeenInTrain": round(float(seen.mean() * 100.0), 1),
                "AccuracyOnSeenNames": round(float(np.mean(predictions[seen] == prepared["yTestPositions"][seen]) * 100.0), 1),
                "AccuracyOnUnseenNames": round(float(np.mean(predictions[~seen] == prepared["yTestPositions"][~seen]) * 100.0), 1),
            }
        ]
    )
