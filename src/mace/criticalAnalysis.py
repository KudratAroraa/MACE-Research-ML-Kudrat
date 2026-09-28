# runs leakage sensitivity robustness and training-only selection experiments
from __future__ import annotations

import warnings
from collections.abc import Callable
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

from .additionalExperiments import saveTable
from .cascadeEngine import (
    bruteForceSoF,
    cascade,
    selectFromTopKAccuracy,
    selectOrder,
    topKFromFolds,
    tuneParameter,
)
from .config import loadConfig
from .data import loadDataset, splitDataset
from .evaluation import alignProbabilities
from .experiments import modelSpecifications, prepareRepresentations
from .paths import TABLES_DIR
from .publishedResults import PAPER_DATASET_ORDER, PAPER_MODEL_NAMES, PAPER_POOLS
from .reproductionPipeline import (
    classPositions,
    fitProbabilities,
    foldProbabilities,
    percentageMetrics,
    preparePaperDataset,
    runPaperDataset,
    runStandalonePaperBaselines,
)
from .robustness import corruptMetadata, injectLabelNoise, multiclassBrierScore


# return every method instead of selecting only the strongest row
def _methodRows(result: dict, experiment: str, seed: int) -> list[dict]:
    rows = []
    for _, metricRow in result["metrics"].iterrows():
        rows.append(
            {
                "Dataset": result["prepared"]["bundle"].displayName,
                "Experiment": experiment,
                "Seed": seed,
                **metricRow.drop(labels="Dataset").to_dict(),
            }
        )
    return rows


# restore the sequential task order after parallel execution while keeping method order inside each task
# joblib already returns results in submission order so sorting by task position makes that guarantee explicit
def _orderedRows(completed: list[list[dict]]) -> list[dict]:
    ordered = sorted(enumerate(completed), key=lambda pair: pair[0])
    return [row for _, taskRows in ordered for row in taskRows]


# run one worker job with the known unscaled-feature and rare-class warnings hidden
# worker processes do not inherit notebook or script warning filters so each job sets its own
def _quietly(function: Callable[..., Any], *arguments: Any) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=LinAlgWarning)
        warnings.filterwarnings("ignore", message="The least populated class")
        return function(*arguments)


# run one repeated-split job with single-threaded forests inside a parallel worker
def _repeatedSplitRows(datasetName: str, seed: int) -> list[dict]:
    result = runPaperDataset(datasetName, includeBruteForce=False, seed=seed, saveManifest=False, jobs=1)
    return _methodRows(result, "RepeatedStratifiedSplit", seed)


# evaluate every valid method over repeated stratified seeds with independent jobs run in parallel
def runRepeatedSplits(seeds: list[int], jobs: int = -1) -> tuple[pd.DataFrame, pd.DataFrame]:
    taskKeys = [(datasetName, seed) for datasetName in PAPER_DATASET_ORDER for seed in seeds]
    completed = Parallel(n_jobs=jobs)(
        delayed(_quietly)(_repeatedSplitRows, datasetName, seed) for datasetName, seed in taskKeys
    )
    detailed = pd.DataFrame(_orderedRows(completed)).reset_index(drop=True)
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC", "BalancedAccuracy", "WeightedF1"]
    summary = detailed.groupby(["Dataset", "Method"])[metrics].agg(["mean", "std", "min", "max"])
    saveTable(detailed, "repeatedSplitResults")
    summary.to_csv(TABLES_DIR / "repeatedSplitSummary.csv")
    (TABLES_DIR / "repeatedSplitSummary.md").write_text(
        summary.to_markdown(floatfmt=".2f"),
        encoding="utf-8",
    )
    return detailed, summary


# run one split-protocol job with single-threaded forests inside a parallel worker
def _leakageRows(datasetName: str, strategy: str) -> list[dict]:
    result = runPaperDataset(
        datasetName,
        includeBruteForce=False,
        splitStrategy=strategy,
        saveManifest=False,
        jobs=1,
    )
    return _methodRows(result, f"Split_{strategy}", 100)


# compare row duplicate-group and ThingSpeak channel-grouped protocols with independent jobs run in parallel
def runLeakageAwareSplits(jobs: int = -1) -> pd.DataFrame:
    taskKeys = []
    for datasetName in PAPER_DATASET_ORDER:
        strategies = ["stratified", "groupAware"]
        if datasetName == "thingspeak":
            strategies.append("channelAware")
        taskKeys.extend((datasetName, strategy) for strategy in strategies)
    completed = Parallel(n_jobs=jobs)(
        delayed(_quietly)(_leakageRows, datasetName, strategy) for datasetName, strategy in taskKeys
    )
    rows = pd.DataFrame(_orderedRows(completed)).reset_index(drop=True)
    return saveTable(rows, "leakageAwareSplitResults")


# prepare the faithful seed-100 baseline of every dataset once for reuse across critical analyses
def prepareBaselines() -> dict[str, dict]:
    return {datasetName: preparePaperDataset(datasetName) for datasetName in PAPER_DATASET_ORDER}


# copy the reusable caches so a variant can replace models without touching the baseline
def _copyPrepared(prepared: dict) -> dict:
    return {
        **prepared,
        "configuration": dict(prepared["configuration"]),
        "specifications": list(prepared["specifications"]),
        "foldCache": {name: list(folds) for name, folds in prepared["foldCache"].items()},
        "testCache": {name: values.copy() for name, values in prepared["testCache"].items()},
        "fittedModels": dict(prepared["fittedModels"]),
    }


# refit only the named pool members of a copied preparation and replace their cached scores
def _refitMembers(variant: dict, specifications: list, names: list[str]) -> None:
    config = loadConfig()
    lookup = {specification.name: specification for specification in specifications}
    for name in names:
        specification = lookup[name]
        variant["foldCache"][name] = foldProbabilities(
            specification,
            variant["trainInputs"],
            variant["yTrain"],
            variant["classes"],
            config["selection"]["cvFolds"],
        )
        variant["fittedModels"][name], variant["testCache"][name] = fitProbabilities(
            specification,
            variant["trainInputs"],
            variant["yTrain"],
            variant["testInputs"],
            variant["classes"],
        )
    variant["specifications"] = [
        lookup[specification.name] if specification.name in names else specification
        for specification in variant["specifications"]
    ]


# build the preparation for one sensitivity variant by reusing every baseline cache the variant leaves unchanged
def _sensitivityPrepared(baseline: dict, options: dict) -> dict:
    if not {"ddlDistancePower", "ddlDistanceVariant", "constantMomentPolicy"} & set(options):
        return baseline
    variant = _copyPrepared(baseline)
    variant["configuration"].update(options)
    configuration = variant["configuration"]
    specifications = modelSpecifications(
        configuration["datasetName"],
        seed=configuration["seed"],
        ddlDistancePower=configuration["ddlDistancePower"],
        ddlDistanceVariant=configuration["ddlDistanceVariant"],
    )

    # only DDL-NLP reads stream names so only its scores change with the distance definition
    if "constantMomentPolicy" not in options:
        _refitMembers(variant, specifications, ["DDL-NLP"])
        return variant

    # only BOS models read the moment features so 1NN-ED and DDL-NLP keep their baseline scores
    variant["trainInputs"], variant["testInputs"], _, _ = prepareRepresentations(
        variant["bundle"],
        variant["split"],
        configuration["constantMomentPolicy"],
    )
    bosMembers = [specification.name for specification in variant["specifications"] if specification.domain == "bos"]
    _refitMembers(variant, specifications, bosMembers)
    variant["standaloneBOS"] = runStandalonePaperBaselines(variant)
    return variant


# change one ambiguous implementation choice at a time and retain every method
# each dataset is prepared once and variants refit only the models their change affects
def runPaperCodeSensitivity(baselinePrepared: dict[str, dict] | None = None) -> pd.DataFrame:
    variants = [
        ("LegacyBaseline", {}),
        ("PFCeiling", {"percentageRule": "ceiling"}),
        ("EarlyExit", {"earlyExit": True}),
        ("DDLPowerOne", {"ddlDistancePower": 1}),
        ("DDLUnrestricted", {"ddlDistanceVariant": "unrestricted"}),
        ("TrainingMedianMoments", {"constantMomentPolicy": "medianTraining"}),
    ]
    baselinePrepared = prepareBaselines() if baselinePrepared is None else baselinePrepared
    rows = []
    for datasetName in PAPER_DATASET_ORDER:
        for variantName, options in variants:
            if datasetName != "thingspeak" and variantName in {"DDLPowerOne", "DDLUnrestricted"}:
                continue
            prepared = _sensitivityPrepared(baselinePrepared[datasetName], options)
            result = runPaperDataset(datasetName, includeBruteForce=False, prepared=prepared, **options)
            rows.extend(_methodRows(result, variantName, 100))
    return saveTable(pd.DataFrame(rows), "paperCodeSensitivity")


# stress the reproduced ThingSpeak configuration at deployment time
# a supplied baseline preparation is reused so no model is refitted
def runMetadataRobustness(
    rates: list[float],
    modes: list[str] | None = None,
    baselinePrepared: dict[str, dict] | None = None,
) -> pd.DataFrame:
    prepared = None if baselinePrepared is None else baselinePrepared["thingspeak"]
    result = runPaperDataset("thingspeak", includeBruteForce=False, prepared=prepared)
    prepared = result["prepared"]
    configuration = result["configuration"]
    order = configuration["order"]
    modes = modes or ["missing", "delete", "replace", "case"]
    rows = []
    for mode in modes:
        for rate in rates:
            cache = {name: values.copy() for name, values in prepared["testCache"].items()}
            corrupted = corruptMetadata(prepared["testInputs"]["metadata"], rate, seed=100, mode=mode)
            model = prepared["fittedModels"]["DDL-NLP"]
            raw = model.predict_proba(corrupted)
            cache["DDL-NLP"] = alignProbabilities(raw, model.classes_, prepared["classes"])
            prediction, probabilities = cascade(
                [cache[name] for name in order],
                prepared["classes"].size,
                "sof",
                configuration["sigma"],
            )
            rows.append(
                {
                    "Dataset": prepared["bundle"].displayName,
                    "Mode": mode,
                    "CorruptionRate": rate,
                    **percentageMetrics(
                        prepared["yTest"],
                        prediction,
                        probabilities,
                        prepared["classes"],
                    ),
                }
            )
    return saveTable(pd.DataFrame(rows), "thingspeakMetadataRobustness")


# select and fit a candidate subset once using shared CV folds
def _cachedSelection(
    specifications: list,
    trainInputs: dict[str, np.ndarray],
    yTrain: np.ndarray,
    testInputs: dict[str, np.ndarray],
    classes: np.ndarray,
) -> tuple[list[str], float, dict[str, np.ndarray]]:
    foldCache = {
        specification.name: foldProbabilities(specification, trainInputs, yTrain, classes, 5)
        for specification in specifications
    }
    curves = [topKFromFolds(foldCache[specification.name], classes.size) for specification in specifications]
    selectedIndices, topKValues = selectFromTopKAccuracy(curves, classes.size)
    order, _ = selectOrder(selectedIndices, [specification.name for specification in specifications], topKValues)
    sigma, _ = tuneParameter(order, foldCache, classes.size, "sof", [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    testCache = {}
    for specification in specifications:
        _, testCache[specification.name] = fitProbabilities(
            specification,
            trainInputs,
            yTrain,
            testInputs,
            classes,
        )
    return order, sigma, testCache


# retune the cascade for one dataset and noise rate with single-threaded forests inside a parallel worker
def _labelNoiseRows(datasetName: str, rate: float) -> list[dict]:
    bundle = loadDataset(datasetName)
    split = splitDataset(bundle, 0.30, 100, "stratified")
    trainInputs, testInputs, yTrain, yTest = prepareRepresentations(bundle, split, "zero")
    noisyLabels = injectLabelNoise(yTrain, rate, seed=100)
    specifications = modelSpecifications(datasetName, seed=100, jobs=1)
    order, sigma, testCache = _cachedSelection(
        specifications,
        trainInputs,
        noisyLabels,
        testInputs,
        bundle.classes,
    )
    prediction, probabilities = cascade(
        [testCache[name] for name in order],
        bundle.classes.size,
        "sof",
        sigma,
    )
    return [
        {
            "Dataset": bundle.displayName,
            "NoiseRate": rate,
            "Order": " > ".join(PAPER_MODEL_NAMES[name] for name in order),
            "Sigma": sigma,
            **percentageMetrics(yTest, prediction, probabilities, bundle.classes),
        }
    ]


# retune the cascade after corrupting training labels only with independent jobs run in parallel
def runLabelNoise(rates: list[float], jobs: int = -1) -> pd.DataFrame:
    taskKeys = [(datasetName, rate) for datasetName in PAPER_DATASET_ORDER for rate in rates]
    completed = Parallel(n_jobs=jobs)(
        delayed(_quietly)(_labelNoiseRows, datasetName, rate) for datasetName, rate in taskKeys
    )
    rows = pd.DataFrame(_orderedRows(completed)).reset_index(drop=True)
    return saveTable(rows, "trainingLabelNoise")


# retune each available information-domain subset inside training data
def runDomainAblation() -> pd.DataFrame:
    rows = []
    for datasetName in PAPER_DATASET_ORDER:
        bundle = loadDataset(datasetName)
        split = splitDataset(bundle, 0.30, 100, "stratified")
        trainInputs, testInputs, yTrain, yTest = prepareRepresentations(bundle, split, "zero")
        specifications = modelSpecifications(datasetName, seed=100)
        availableDomains = list(dict.fromkeys(specification.domain for specification in specifications))
        domainSets = []
        for size in range(1, len(availableDomains) + 1):
            domainSets.extend(combinations(availableDomains, size))
        for domainSet in domainSets:
            selected = [specification for specification in specifications if specification.domain in domainSet]
            order, sigma, testCache = _cachedSelection(
                selected,
                trainInputs,
                yTrain,
                testInputs,
                bundle.classes,
            )
            prediction, probabilities = cascade(
                [testCache[name] for name in order],
                bundle.classes.size,
                "sof",
                sigma,
            )
            rows.append(
                {
                    "Dataset": bundle.displayName,
                    "Domains": " + ".join(domainSet),
                    "Order": " > ".join(PAPER_MODEL_NAMES[name] for name in order),
                    "Sigma": sigma,
                    **percentageMetrics(yTest, prediction, probabilities, bundle.classes),
                }
            )
    return saveTable(pd.DataFrame(rows), "informationDomainAblation")


# select the exhaustive SoF configuration on validation then test it once
def runTrainingOnlyBruteForce() -> pd.DataFrame:
    rows = []
    sigmaValues = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for datasetName in PAPER_DATASET_ORDER:
        bundle = loadDataset(datasetName)
        split = splitDataset(bundle, 0.30, 100, "stratified")
        trainInputs, testInputs, yTrain, yTest = prepareRepresentations(bundle, split, "zero")
        fitIndex, validationIndex = train_test_split(
            np.arange(yTrain.size),
            test_size=0.25,
            random_state=100,
            stratify=yTrain,
        )
        fitInputs = {name: values[fitIndex] for name, values in trainInputs.items()}
        validationInputs = {name: values[validationIndex] for name, values in trainInputs.items()}
        specifications = modelSpecifications(datasetName, seed=100)
        validationCache = {}
        testCache = {}
        for specification in specifications:
            _, validationCache[specification.name] = fitProbabilities(
                specification,
                fitInputs,
                yTrain[fitIndex],
                validationInputs,
                bundle.classes,
            )
            _, testCache[specification.name] = fitProbabilities(
                specification,
                trainInputs,
                yTrain,
                testInputs,
                bundle.classes,
            )
        pool = PAPER_POOLS[datasetName]
        validationOracle = bruteForceSoF(
            validationCache,
            pool,
            bundle.classes.size,
            classPositions(yTrain[validationIndex], bundle.classes),
            sigmaValues,
        )
        prediction, probabilities = cascade(
            [testCache[name] for name in validationOracle["order"]],
            bundle.classes.size,
            "sof",
            validationOracle["sigma"],
        )
        rows.append(
            {
                "Dataset": bundle.displayName,
                "SelectionProtocol": "TrainingOnly_ValidationSelected",
                "ValidForRecommendation": True,
                "Order": " > ".join(PAPER_MODEL_NAMES[name] for name in validationOracle["order"]),
                "Sigma": validationOracle["sigma"],
                "ConfigurationsSearched": validationOracle["configurationsSearched"],
                **percentageMetrics(yTest, prediction, probabilities, bundle.classes),
            }
        )
    return saveTable(pd.DataFrame(rows), "trainingOnlyBruteForce")


# compare probability quality across all valid final methods
# a supplied baseline preparation is reused so no model is refitted
def runCalibrationEvidence(baselinePrepared: dict[str, dict] | None = None) -> pd.DataFrame:
    rows = []
    for datasetName in PAPER_DATASET_ORDER:
        prepared = None if baselinePrepared is None else baselinePrepared[datasetName]
        result = runPaperDataset(datasetName, includeBruteForce=False, prepared=prepared)
        prepared = result["prepared"]
        for method, probabilities in result["probabilities"].items():
            prediction = result["predictions"][method]
            metrics = percentageMetrics(
                prepared["yTest"],
                prediction,
                probabilities,
                prepared["classes"],
            )
            rows.append(
                {
                    "Dataset": prepared["bundle"].displayName,
                    "Method": method,
                    **metrics,
                    "BrierScore": multiclassBrierScore(
                        prepared["yTest"],
                        probabilities,
                        prepared["classes"],
                    ),
                }
            )
    return saveTable(pd.DataFrame(rows), "calibrationAndImbalance")
