# implements dataset audit tuned BOS and time-series reproduction experiments
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .classifiers import buildBOSClassifiers
from .config import loadConfig
from .data import classDistribution, datasetAudit, loadDataset, saveSplitManifest, splitDataset
from .evaluation import alignProbabilities
from .evaluation import bootstrapMetricInterval, confusionTable, perClassMetrics
from .features import BOS_FEATURE_NAMES, extractBOSFeatures, zNormaliseSeries
from .paths import TABLES_DIR, ensureOutputDirectories
from .publishedResults import PAPER_DATASET_FACTS, PAPER_DATASET_ORDER, PAPER_FIGURE3A, PAPER_MODEL_NAMES
from .reproductionPipeline import classPositions, percentageMetrics
from .timeSeries import buildTimeSeriesComparators


# save one evidence table as CSV and Markdown
def saveTable(table: pd.DataFrame, name: str) -> pd.DataFrame:
    ensureOutputDirectories()
    table.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    (TABLES_DIR / f"{name}.md").write_text(
        table.to_markdown(index=False, floatfmt=".2f"),
        encoding="utf-8",
    )
    return table


# run the complete Exp1 to Exp4 dataset audit shared by notebook 1 and the command-line runner
# writes structure summary split quality class-distribution and BOS feature tables plus the split manifests
def runDataAudit() -> dict[str, pd.DataFrame]:
    config = loadConfig()
    structureRows = []
    paperCheckRows = []
    datasetRows = []
    splitRows = []
    qualityRows = []
    distributionRows = []
    featureRows = []
    for datasetName in PAPER_DATASET_ORDER:
        bundle = loadDataset(datasetName)
        split = splitDataset(bundle, config["testSize"], config["randomSeed"], config["splitStrategy"])
        saveSplitManifest(bundle, split)
        audit = datasetAudit(bundle, split)
        qualityRows.append(audit)
        counts = np.bincount(bundle.labels)
        structureRows.append(
            {
                "Dataset": bundle.displayName,
                "Observations": bundle.shape[0],
                "Time-series readings": bundle.shape[1],
                "Classes": bundle.classes.size,
                "Missing numerical values": int(pd.isna(bundle.series).sum()),
                "Unique IDs": pd.Series(bundle.ids).nunique(),
            }
        )

        # compare the loaded dimensions with Section IV-A of the paper
        facts = PAPER_DATASET_FACTS[datasetName]
        mineStreams, mineLength, mineClasses = bundle.series.shape[0], bundle.series.shape[1], bundle.classes.size
        paperCheckRows.append(
            {
                "Dataset": bundle.displayName,
                "PaperStreams": facts["streams"],
                "MineStreams": mineStreams,
                "PaperLength": facts["length"],
                "MineLength": mineLength,
                "PaperClasses": facts["classes"],
                "MineClasses": mineClasses,
                "HasMetadata": facts["metadata"],
                "Match": (mineStreams, mineLength, mineClasses)
                == (facts["streams"], facts["length"], facts["classes"]),
            }
        )
        datasetRows.append(
            {
                "Dataset": bundle.displayName,
                "Streams": bundle.series.shape[0],
                "Length": bundle.series.shape[1],
                "Classes": bundle.classes.size,
                "ExactConstantSeries": audit["exactConstantSeries"],
                "UndefinedMomentRows": audit["undefinedMomentRows"],
                "DuplicatedSeries": int(bundle.series.shape[0] - np.unique(bundle.series, axis=0).shape[0]),
                "MajorityClassBaseline": round(counts.max() / counts.sum() * 100.0, 1),
            }
        )
        splitRows.append(
            {
                "Dataset": bundle.displayName,
                "TrainingRows": split.trainIndex.size,
                "TestingRows": split.testIndex.size,
                "TestShare": round(split.testIndex.size / bundle.series.shape[0] * 100.0, 1),
                "Seed": split.seed,
                "Strategy": split.strategy,
            }
        )
        distribution = classDistribution(bundle)
        distribution.insert(0, "Dataset", bundle.displayName)
        distributionRows.append(distribution)

        # keep feature ranges so scale-sensitive classifiers such as SVM and LR can be explained
        features = extractBOSFeatures(bundle.series, config["constantMomentPolicy"])
        summary = features.agg(["mean", "std", "min", "max"]).T.reset_index(names="Feature")
        summary.insert(0, "Dataset", bundle.displayName)
        featureRows.append(summary)
    outputs = {
        "datasetStructure": pd.DataFrame(structureRows),
        "datasetPaperCheck": pd.DataFrame(paperCheckRows),
        "datasetSummary": pd.DataFrame(datasetRows),
        "splitSummary": pd.DataFrame(splitRows),
        "datasetQualityAudit": pd.DataFrame(qualityRows),
        "classDistribution": pd.concat(distributionRows, ignore_index=True),
        "featureSummary": pd.concat(featureRows, ignore_index=True),
        "featureOrder": pd.DataFrame(
            {"Position": np.arange(1, len(BOS_FEATURE_NAMES) + 1), "Feature": BOS_FEATURE_NAMES}
        ),
    }
    for name, table in outputs.items():
        saveTable(table, name)
    return outputs


# test the paper's unreported optimisation claim with stated training-only grids
def runTunedBOS(datasetResults: dict[str, dict[str, Any]]) -> pd.DataFrame:
    grids = {
        "Ridge": {"estimator__alpha": [0.1, 1.0, 10.0]},
        "SVM": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.1]},
        "kNN": {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
        "LogisticRegression": {"C": [0.1, 1.0, 10.0]},
        "DecisionTree": {"max_depth": [None, 5, 10, 20], "min_samples_leaf": [1, 2, 5]},
        "GaussianNB": {"var_smoothing": [1.0e-11, 1.0e-9, 1.0e-7]},
        "RandomForest": {"max_depth": [None, 20], "max_features": ["sqrt", None]},
        "GradientBoosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    }
    rows = []
    for datasetName in PAPER_DATASET_ORDER:
        prepared = datasetResults[datasetName]["prepared"]
        for modelName, estimator in buildBOSClassifiers(100).items():
            search = GridSearchCV(
                clone(estimator),
                grids[modelName],
                scoring="accuracy",
                cv=StratifiedKFold(5, shuffle=False),
                n_jobs=-1,
                refit=True,
            )
            started = time.perf_counter()
            search.fit(prepared["trainInputs"]["bos"], prepared["yTrain"])
            fitSeconds = time.perf_counter() - started
            fitted = search.best_estimator_
            raw = fitted.predict_proba(prepared["testInputs"]["bos"])
            probabilities = alignProbabilities(raw, fitted.classes_, prepared["classes"])
            predictions = classPositions(fitted.predict(prepared["testInputs"]["bos"]), prepared["classes"])
            rows.append(
                {
                    "Dataset": prepared["bundle"].displayName,
                    "Classifier": PAPER_MODEL_NAMES[modelName],
                    "BestParameters": json.dumps(search.best_params_, sort_keys=True),
                    "MeanCVAccuracy": search.best_score_ * 100.0,
                    **percentageMetrics(
                        prepared["yTest"],
                        predictions,
                        probabilities,
                        prepared["classes"],
                    ),
                    "FitSeconds": fitSeconds,
                }
            )
    return saveTable(pd.DataFrame(rows), "tunedBosMetrics")


# compare labelled sensitivity variants with the published value of their main method
SENSITIVITY_PAPER_KEYS = {"1NN-DTW-band10": "1NN-DTW"}
SENSITIVITY_STATUS = {"1NN-DTW-band10": "Sensitivity (10% band)"}


# label each Figure 3a row as an exact method a proxy or a sensitivity variant
def _timeSeriesStatus(modelName: str) -> str:
    if modelName in SENSITIVITY_STATUS:
        return SENSITIVITY_STATUS[modelName]
    return "Proxy" if "Proxy" in modelName else "Exact available method"


# evaluate Figure 3a methods with unavailable methods clearly labelled as proxies
def runTimeSeriesBaselines(
    datasetResults: dict[str, dict[str, Any]],
    includeDTW: bool = True,
    includeProxies: bool = True,
) -> pd.DataFrame:
    rows = []
    for datasetName in PAPER_DATASET_ORDER:
        prepared = datasetResults[datasetName]["prepared"]
        comparators = buildTimeSeriesComparators(100)
        if not includeDTW:
            comparators = {name: model for name, model in comparators.items() if not name.startswith("1NN-DTW")}
        if not includeProxies:
            comparators = {name: model for name, model in comparators.items() if "Proxy" not in name}
        for representation in ["raw", "znorm"]:
            trainSeries = prepared["trainInputs"]["series"]
            testSeries = prepared["testInputs"]["series"]
            if representation == "znorm":
                trainSeries = zNormaliseSeries(trainSeries)
                testSeries = zNormaliseSeries(testSeries)
            for modelName, estimator in comparators.items():
                started = time.perf_counter()
                fitted = clone(estimator).fit(trainSeries, prepared["yTrain"])
                fitSeconds = time.perf_counter() - started
                started = time.perf_counter()
                raw = fitted.predict_proba(testSeries)
                probabilities = alignProbabilities(raw, fitted.classes_, prepared["classes"])

                # take predictions from the same scores so each distance matrix is computed once
                predictions = classPositions(fitted.classes_[np.argmax(raw, axis=1)], prepared["classes"])
                predictionSeconds = time.perf_counter() - started
                paperKey = SENSITIVITY_PAPER_KEYS.get(modelName, modelName.replace("-Proxy", ""))
                published = PAPER_FIGURE3A.get((paperKey, representation), {}).get(datasetName, np.nan)
                rows.append(
                    {
                        "Dataset": prepared["bundle"].displayName,
                        "Method": modelName,
                        "Representation": representation,
                        "Status": _timeSeriesStatus(modelName),
                        "PaperAccuracy": published,
                        **percentageMetrics(
                            prepared["yTest"],
                            predictions,
                            probabilities,
                            prepared["classes"],
                        ),
                        "FitSeconds": fitSeconds,
                        "PredictionSeconds": predictionSeconds,
                    }
                )
    return saveTable(pd.DataFrame(rows), "timeSeriesMetrics")


# save per-class confusion and bootstrap evidence for the best valid method
def runErrorAnalysis(datasetResults: dict[str, dict[str, Any]]) -> dict[str, pd.DataFrame]:
    perClassRows = []
    confusionRows = []
    intervalRows = []
    validMethods = ["Soft voting", "MACE Top-k", "MACE PF", "MACE SoF"]
    for datasetName in PAPER_DATASET_ORDER:
        result = datasetResults[datasetName]
        prepared = result["prepared"]
        valid = result["metrics"].loc[result["metrics"]["Method"].isin(validMethods)]
        bestMethod = valid.sort_values(["F1", "Accuracy"], ascending=False).iloc[0]["Method"]
        predictionPositions = result["predictions"][bestMethod]
        predictions = prepared["classes"][predictionPositions]
        perClass = perClassMetrics(prepared["yTest"], predictions, prepared["classes"])
        perClass.insert(0, "Dataset", prepared["bundle"].displayName)
        perClass.insert(1, "Method", bestMethod)
        perClassRows.append(perClass)
        confusion = confusionTable(prepared["yTest"], predictions, prepared["classes"])
        confusion.insert(0, "ActualClass", confusion.index)
        confusion.insert(0, "Method", bestMethod)
        confusion.insert(0, "Dataset", prepared["bundle"].displayName)
        confusionRows.append(confusion.reset_index(drop=True))
        intervalRows.append(
            {
                "Dataset": prepared["bundle"].displayName,
                "Method": bestMethod,
                **bootstrapMetricInterval(
                    prepared["yTest"],
                    predictions,
                    metric="f1_macro",
                    iterations=1000,
                    confidence=0.95,
                    seed=100,
                ),
            }
        )
    outputs = {
        "perClassMetrics": pd.concat(perClassRows, ignore_index=True),
        "confusionMatrices": pd.concat(confusionRows, ignore_index=True),
        "macroF1BootstrapIntervals": pd.DataFrame(intervalRows),
    }
    for name, table in outputs.items():
        saveTable(table, name)
    return outputs
