# runs the isolated Part 2 experiments from Exp22 to Exp30
from __future__ import annotations

import pickle
import time
import warnings
from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

from .evaluation import perClassMetrics
from .plotting import plotSeedSpread
from .proposedConfig import (
    PART2_TABLES_DIR,
    PROPOSED_CONFIG,
    ensurePart2Directories,
)
from .proposedPlotting import (
    plotAblation,
    plotAccuracyCalibrationTradeoff,
    plotDifferenceForest,
    plotGateCoefficients,
    plotGateFeatureDistributions,
    plotGateSelection,
    plotMetricBars,
    plotReliabilityDiagram,
    plotRobustnessPanels,
    plotRoutingShare,
    plotSubsetComparison,
    plotThreeWayComparison,
    savePart2Figure,
)
from .proposedSolution import (
    METHOD_NAMES,
    ReliabilityGatedMace,
    SoftStackingReference,
    evaluatePredictions,
    predictCaseFoldedMissingNameRule,
    predictMaceSof,
    predictMissingNameRule,
    predictRandomForest,
    predictSoftVoting,
)
from .proposedStatistics import (
    areaUnderRobustnessCurve,
    mcnemarExact,
    pairedBootstrapDifference,
    pairedWilcoxon,
    seedDifferenceInterval,
)
from .publishedResults import PAPER_TABLE1, PAPER_TABLE2
from .reliabilityGate import (
    CONFIDENCE_FEATURES,
    NAME_DISTANCE_FEATURES,
    RELIABILITY_FEATURES,
    caseFoldCollisionAudit,
)
from .reproductionPipeline import percentageMetrics, preparePaperDataset, runPaperDataset
from .robustness import corruptMetadata, multiclassBrierScore


PART1_TABLES_DIR = PART2_TABLES_DIR.parent
REQUIRED_METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUC", "BalancedAccuracy"]
CASE_FOLDED_RULE = "Case-folded MACE + missing-name rule"


# hide the warnings Part 2 triggers by design in runners and in joblib workers that do not inherit notebook filters
def _silenceKnownWarnings() -> None:
    # logistic regression is fitted on unscaled features as in Part 1
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # ridge in the Part 1 pool is ill-conditioned on the unscaled BOS features
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    # the smallest ThingSpeak class has only 4 streams so it cannot fill every CV fold
    warnings.filterwarnings("ignore", message="The least populated class", category=UserWarning)
    # subset metrics are computed on subsets where some rare classes are absent
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
    # subset metrics are computed on subsets where some rare classes are absent
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# retain prepared datasets and fitted Part 2 models across experiments in one execution
@dataclass
class Part2Context:
    prepared: dict[tuple[int, str], dict[str, Any]] = field(default_factory=dict)
    preparationSeconds: dict[tuple[int, str], float] = field(default_factory=dict)
    maceExperts: dict[tuple[int, str, bool], dict[str, Any]] = field(default_factory=dict)
    rgModels: dict[tuple[int, str, str], ReliabilityGatedMace] = field(default_factory=dict)
    stackingModels: dict[tuple[int, str], SoftStackingReference] = field(default_factory=dict)

    # prepare one split once without writing a Part 1 manifest
    def getPrepared(self, seed: int = 100, strategy: str = "stratified", jobs: int = -1) -> dict[str, Any]:
        key = (seed, strategy)
        if key not in self.prepared:
            started = time.perf_counter()
            self.prepared[key] = preparePaperDataset(
                "thingspeak",
                seed=seed,
                splitStrategy=strategy,
                saveManifest=False,
                jobs=jobs,
            )
            self.preparationSeconds[key] = time.perf_counter() - started
        return self.prepared[key]

    # build one original or case-folded MACE expert once per split
    def getMaceExpert(
        self,
        seed: int = 100,
        strategy: str = "stratified",
        caseFold: bool = False,
        jobs: int = -1,
    ) -> dict[str, Any]:
        from .reliabilityGate import buildMaceExpert

        key = (seed, strategy, caseFold)
        if key not in self.maceExperts:
            self.maceExperts[key] = buildMaceExpert(
                self.getPrepared(seed, strategy, jobs),
                caseFold,
            )
        return self.maceExperts[key]

    # fit one named RG-MACE variant once for a split
    def getRgModel(
        self,
        seed: int = 100,
        strategy: str = "stratified",
        variant: str = "full",
        jobs: int = -1,
        **options: Any,
    ) -> ReliabilityGatedMace:
        key = (seed, strategy, variant)
        if key not in self.rgModels:
            settings = {
                "caseFold": PROPOSED_CONFIG["caseFold"],
                "corruptionShare": PROPOSED_CONFIG["corruptionShare"],
                "modes": PROPOSED_CONFIG["corruptionModes"],
                "cGrid": PROPOSED_CONFIG["gateCGrid"],
                "innerFolds": PROPOSED_CONFIG["innerFolds"],
                "cleanToleranceF1": PROPOSED_CONFIG["cleanToleranceF1"],
                "seed": seed,
            }
            settings.update(options)
            self.rgModels[key] = ReliabilityGatedMace(**settings).fit(
                self.getPrepared(seed, strategy, jobs)
            )
            self.maceExperts[(seed, strategy, settings["caseFold"])] = self.rgModels[key].maceExpert_
        return self.rgModels[key]

    # fit the generic stacking comparator once for a split
    def getStacking(self, seed: int = 100, strategy: str = "stratified", jobs: int = -1) -> SoftStackingReference:
        key = (seed, strategy)
        if key not in self.stackingModels:
            self.stackingModels[key] = SoftStackingReference(seed=seed).fit(
                self.getPrepared(seed, strategy, jobs)
            )
        return self.stackingModels[key]


# save one isolated Part 2 evidence table as both CSV and Markdown
def savePart2Table(table: pd.DataFrame, name: str) -> pd.DataFrame:
    ensurePart2Directories()
    table.to_csv(PART2_TABLES_DIR / f"{name}.csv", index=False)
    (PART2_TABLES_DIR / f"{name}.md").write_text(
        table.to_markdown(index=False, floatfmt=".3f"),
        encoding="utf-8",
    )
    return table


# close and save one figure only inside the Part 2 image folder
def _saveAndClose(figure: plt.Figure, name: str) -> None:
    savePart2Figure(figure, name)
    plt.close(figure)


# evaluate a prediction object on a subset while retaining the global class order
def _subsetMetrics(prepared: dict[str, Any], result: dict[str, Any], mask: np.ndarray) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)

    # subset AUC uses every global class so scikit-learn returns NaN when any class is absent from the subset
    metrics = percentageMetrics(
        prepared["yTest"][mask],
        np.asarray(result["predictions"])[mask],
        np.asarray(result["probabilities"])[mask],
        prepared["classes"],
    )
    metrics["BrierScore"] = multiclassBrierScore(
        prepared["yTest"][mask],
        np.asarray(result["probabilities"])[mask],
        prepared["classes"],
    )
    return metrics


# label how the gate C was chosen and how far each C falls short of the clean-F1 constraint without changing the selection
# a positive shortfall means the clean-F1 constraint was missed
def _annotatedGateSelection(model: ReliabilityGatedMace) -> pd.DataFrame:
    selection = model.selection_["table"].copy()
    selection["Chosen"] = selection["C"] == model.chosenC_
    rule = (
        "Constrained"
        if model.selection_["constraintMet"]
        else "Pre-specified fallback: no C met the clean-F1 constraint; highest clean F1 chosen"
    )
    selection["SelectionRule"] = np.where(selection["Chosen"], rule, "")
    selection["ShortfallFromConstraint"] = selection["MinimumAllowedCleanF1"] - selection["CleanF1"]
    return selection


# convert a soft RG-MACE prediction into the matching hard-routing ablation
def _hardGateResult(softResult: dict[str, Any]) -> dict[str, np.ndarray]:
    useMace = np.asarray(softResult["gate"]) >= 0.5
    probabilities = np.asarray(softResult["rfProbabilities"]).copy()
    probabilities[useMace] = np.asarray(softResult["maceProbabilities"])[useMace]
    return {"predictions": probabilities.argmax(axis=1), "probabilities": probabilities}


# match Part 1 Exp8 by comparing stripped lowercase training and test names
def _unseenNameMask(prepared: dict[str, Any]) -> np.ndarray:
    trainingNames = {
        str(name).strip().lower()
        for name in prepared["trainInputs"]["metadata"]
    }
    return np.asarray(
        [str(name).strip().lower() not in trainingNames for name in prepared["testInputs"]["metadata"]],
        dtype=bool,
    )


# evaluate every clean or corrupted comparison method from shared fitted objects
def _methodResults(
    context: Part2Context,
    prepared: dict[str, Any],
    names: np.ndarray,
    seed: int = 100,
    strategy: str = "stratified",
    includeStacking: bool = True,
) -> dict[str, dict[str, Any]]:
    model = context.getRgModel(seed, strategy)
    originalExpert = context.getMaceExpert(seed, strategy, caseFold=False)
    rg = model.predict(prepared, names)
    results = {
        "RF": predictRandomForest(prepared),
        "Soft voting": predictSoftVoting(prepared, names, maceExpert=originalExpert),
        "MACE SoF": predictMaceSof(prepared, names, caseFold=False, maceExpert=originalExpert),
        "MACE SoF + missing-name rule": predictMissingNameRule(prepared, names, maceExpert=originalExpert),
        CASE_FOLDED_RULE: predictCaseFoldedMissingNameRule(prepared, names, maceExpert=model.maceExpert_),
        "RG-MACE (hard gate)": _hardGateResult(rg),
        "RG-MACE": rg,
    }
    if includeStacking:
        results["Soft stacking"] = context.getStacking(seed, strategy).predict(names)
    return results


# extract the strongest published Part 1 evidence without recomputing it
def runExp22Motivation(context: Part2Context | None = None) -> dict[str, pd.DataFrame]:
    _silenceKnownWarnings()
    robustness = pd.read_csv(PART1_TABLES_DIR / "thingspeakMetadataRobustness.csv")
    diagnostic = pd.read_csv(PART1_TABLES_DIR / "ddlNLPDiagnostic.csv").iloc[0]
    calibration = pd.read_csv(PART1_TABLES_DIR / "calibrationAndImbalance.csv")
    leakage = pd.read_csv(PART1_TABLES_DIR / "leakageAwareSplitResults.csv")
    standalone = pd.read_csv(PART1_TABLES_DIR / "standalonePool.csv")
    missing = robustness.loc[robustness["Mode"] == "missing"].set_index("CorruptionRate")
    case = robustness.loc[robustness["Mode"] == "case"].set_index("CorruptionRate")
    maceCalibration = calibration.loc[
        (calibration["Dataset"] == "ThingSpeakEU") & (calibration["Method"] == "MACE SoF")
    ].iloc[0]
    votingCalibration = calibration.loc[
        (calibration["Dataset"] == "ThingSpeakEU") & (calibration["Method"] == "Soft voting")
    ].iloc[0]
    channel = leakage.loc[
        (leakage["Dataset"] == "ThingSpeakEU")
        & (leakage["Experiment"] == "Split_channelAware")
        & (leakage["Method"] == "MACE SoF")
    ].iloc[0]
    rfAccuracy = float(
        standalone.loc[
            (standalone["Dataset"] == "ThingSpeakEU")
            & (standalone["Classifier"] == "RF"),
            "Accuracy",
        ].iloc[0]
    )
    evidence = pd.DataFrame(
        [
            {"Evidence": "Clean MACE SoF accuracy", "Value": float(missing.loc[0.0, "Accuracy"]), "Unit": "%"},
            {"Evidence": "Random Forest accuracy", "Value": rfAccuracy, "Unit": "%"},
            {"Evidence": "MACE accuracy with 25% names missing", "Value": float(missing.loc[0.25, "Accuracy"]), "Unit": "%"},
            {"Evidence": "MACE accuracy with 50% names missing", "Value": float(missing.loc[0.5, "Accuracy"]), "Unit": "%"},
            {"Evidence": "MACE accuracy with all names missing", "Value": float(missing.loc[1.0, "Accuracy"]), "Unit": "%"},
            {"Evidence": "MACE accuracy after complete case swap", "Value": float(case.loc[1.0, "Accuracy"]), "Unit": "%"},
            {"Evidence": "DDL-NLP accuracy on seen names", "Value": float(diagnostic["AccuracyOnSeenNames"]), "Unit": "%"},
            {"Evidence": "DDL-NLP accuracy on unseen names", "Value": float(diagnostic["AccuracyOnUnseenNames"]), "Unit": "%"},
            {"Evidence": "MACE SoF log loss", "Value": float(maceCalibration["LogLoss"]), "Unit": "loss"},
            {"Evidence": "Soft voting log loss", "Value": float(votingCalibration["LogLoss"]), "Unit": "loss"},
            {"Evidence": "MACE accuracy on channel-grouped split", "Value": float(channel["Accuracy"]), "Unit": "%"},
        ]
    )
    curve = missing.reset_index()[["CorruptionRate", "Accuracy"]].copy()
    curve["RFAccuracy"] = rfAccuracy
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.plot(curve["CorruptionRate"], curve["Accuracy"], marker="o", label="MACE SoF")
    axis.axhline(rfAccuracy, linestyle="--", color="#7B8794", label="RF")
    axis.set(
        title="Part 1 failure case: missing ThingSpeak names",
        xlabel="Share of test names missing",
        ylabel="Accuracy (%)",
    )
    axis.legend()
    figure.tight_layout()
    _saveAndClose(figure, "part1FailureEvidence")
    return {
        "failureEvidence": savePart2Table(evidence, "exp22FailureEvidence"),
        "missingNameCurve": savePart2Table(curve, "exp22MissingNameCurve"),
    }


# inspect the OOF reliability data and verify its leakage controls before gate fitting
def runExp23GateDataset(context: Part2Context | None = None) -> dict[str, pd.DataFrame]:
    _silenceKnownWarnings()
    context = context or Part2Context()
    prepared = context.getPrepared()
    model = context.getRgModel()
    data = model.gateDataset_
    cleanRows = ~data["isCorrupted"]
    gateGlobalRows = set(prepared["split"].trainIndex[data["sourceRow"]].tolist())
    testRows = set(prepared["split"].testIndex.tolist())
    checks = pd.DataFrame(
        [
            {"Check": "Every training row has one clean OOF row", "Result": bool(np.array_equal(np.sort(data["sourceRow"][cleanRows]), np.arange(prepared["yTrain"].size)))},
            {"Check": "No source test index appears in gate groups", "Result": gateGlobalRows.isdisjoint(testRows)},
            {"Check": "Original metadata unchanged", "Result": data["originalMetadataUnchanged"]},
            {"Check": "All reliability features finite", "Result": bool(np.isfinite(data["features"].to_numpy(float)).all())},
            {"Check": "Feature order matches the pre-registered list", "Result": list(data["features"].columns) == RELIABILITY_FEATURES},
        ]
    )
    counts = pd.DataFrame(
        {
            "RowType": pd.Series(data["corruptionMode"]).value_counts().index,
            "Rows": pd.Series(data["corruptionMode"]).value_counts().values,
        }
    )
    summaryRows = []
    for label, mask in {
        "Overall": np.ones(data["targets"].size, dtype=bool),
        "Seen clean names": cleanRows & (data["features"]["exactNameSeen"].to_numpy() == 1),
        "Unseen clean names": cleanRows & (data["features"]["exactNameSeen"].to_numpy() == 0),
        **{
            f"Corruption: {mode}": data["corruptionMode"] == mode
            for mode in PROPOSED_CONFIG["corruptionModes"]
        },
    }.items():
        usable = mask & data["fitMask"]
        summaryRows.append(
            {
                "Group": label,
                "Rows": int(mask.sum()),
                "TiedRowsExcludedFromFit": int(np.sum(mask & ~data["fitMask"])),
                "ExpertDisagreement": float(np.mean(data["maceProbabilities"][mask].argmax(axis=1) != data["rfProbabilities"][mask].argmax(axis=1)) * 100.0) if np.any(mask) else np.nan,
                "MaceWins": float(np.mean(data["targets"][usable] == 1) * 100.0) if np.any(usable) else np.nan,
                "RfWins": float(np.mean(data["targets"][usable] == 0) * 100.0) if np.any(usable) else np.nan,
            }
        )
    summary = pd.DataFrame(summaryRows)
    featureSummary = data["features"].describe().T.reset_index(names="Feature")
    correlation = data["features"].corr()
    correlationTable = correlation.reset_index(names="Feature")
    pairs = []
    for leftIndex, left in enumerate(correlation.columns):
        for right in correlation.columns[leftIndex + 1 :]:
            value = float(correlation.loc[left, right])
            if abs(value) > 0.9:
                pairs.append({"FeatureA": left, "FeatureB": right, "Correlation": value})
    highCorrelation = pd.DataFrame(pairs, columns=["FeatureA", "FeatureB", "Correlation"])
    caseAudit = caseFoldCollisionAudit(prepared["trainInputs"]["metadata"], prepared["yTrain"])
    figure = plotGateFeatureDistributions(
        data,
        ["nameMissing", "nearestNameDistance", "nameDistanceGap", "ddlMargin", "rfMargin", "expertsAgree"],
        "Reliability evidence by the better expert",
    )
    _saveAndClose(figure, "gateFeatureDistributions")
    return {
        "checks": savePart2Table(checks, "exp23LeakageChecks"),
        "rowCounts": savePart2Table(counts, "exp23GateRowCounts"),
        "expertSummary": savePart2Table(summary, "exp23ExpertSummary"),
        "featureSummary": savePart2Table(featureSummary, "exp23FeatureSummary"),
        "correlationMatrix": savePart2Table(correlationTable, "exp23FeatureCorrelationMatrix"),
        "highCorrelation": savePart2Table(highCorrelation, "exp23HighFeatureCorrelations"),
        "caseFoldAudit": savePart2Table(caseAudit, "exp23CaseFoldCollisionAudit"),
    }


# select the gate regularisation and report routing diagnostics rather than final test performance
def runExp24GateSelection(context: Part2Context | None = None) -> dict[str, pd.DataFrame]:
    _silenceKnownWarnings()
    context = context or Part2Context()
    model = context.getRgModel()
    selection = _annotatedGateSelection(model)
    diagnostics = model.selection_["diagnostics"]
    diagnosticTable = pd.DataFrame(
        [
            {
                "ChosenC": model.chosenC_,
                "ConstraintMet": model.selection_["constraintMet"],
                "GateROCAUC": diagnostics["GateROCAUC"],
                "GateBalancedAccuracy": diagnostics["GateBalancedAccuracy"],
                "GatePrecision": diagnostics["GatePrecision"],
                "GateRecall": diagnostics["GateRecall"],
                "GateF1": diagnostics["GateF1"],
                "Interpretation": "Routing quality only, not final classification performance",
            }
        ]
    )
    confusion = pd.DataFrame(
        diagnostics["ConfusionMatrix"],
        index=["Actual_RF", "Actual_MACE"],
        columns=["Predicted_RF", "Predicted_MACE"],
    ).reset_index(names="Actual")
    _saveAndClose(plotGateSelection(selection, "Training-only gate selection"), "gateSelection")
    _saveAndClose(plotGateCoefficients(model.gateCoefficients_, "RG-MACE gate coefficients"), "gateCoefficients")
    return {
        "selection": savePart2Table(selection, "exp24GateSelection"),
        "diagnostics": savePart2Table(diagnosticTable, "exp24GateDiagnostics"),
        "confusion": savePart2Table(confusion, "exp24GateConfusion"),
        "coefficients": savePart2Table(model.gateCoefficients_, "exp24GateCoefficients"),
    }


# compare the proposed method with the paper result and every valid clean-data baseline
def runExp25CleanComparison(context: Part2Context | None = None) -> dict[str, pd.DataFrame]:
    _silenceKnownWarnings()
    context = context or Part2Context()
    rows = []
    for datasetName in ["urban", "swissex"]:
        prepared = preparePaperDataset(datasetName, saveManifest=False)
        reproduced = runPaperDataset(
            datasetName,
            includeBruteForce=False,
            prepared=prepared,
            saveManifest=False,
        )
        metricLookup = reproduced["metrics"].set_index("Method")
        rfRow = reproduced["standalonePool"].loc[
            reproduced["standalonePool"]["Classifier"] == "RF"
        ].iloc[0]
        rows.append({"Dataset": prepared["bundle"].displayName, "Method": "RF", **rfRow.drop(labels=["Dataset", "Classifier"]).to_dict(), "Status": "Evaluated"})
        for method in ["Soft voting", "MACE SoF"]:
            rows.append({"Dataset": prepared["bundle"].displayName, "Method": method, **metricLookup.loc[method].to_dict(), "Status": "Evaluated"})
        for method in ["MACE SoF + missing-name rule", CASE_FOLDED_RULE, "Soft stacking", "RG-MACE (hard gate)", "RG-MACE"]:
            rows.append({"Dataset": prepared["bundle"].displayName, "Method": method, **{metric: np.nan for metric in ["Accuracy", "Precision", "Recall", "F1", "AUC", "BalancedAccuracy", "WeightedF1", "LogLoss", "BrierScore"]}, "Status": "Not applicable (single information domain)"})

    prepared = context.getPrepared()
    names = prepared["testInputs"]["metadata"]
    results = _methodResults(context, prepared, names)
    for method in METHOD_NAMES:
        if method in results:
            rows.append({"Dataset": prepared["bundle"].displayName, "Method": method, **evaluatePredictions(prepared, results[method]["predictions"], results[method]["probabilities"]), "Status": "Evaluated"})
    metrics = pd.DataFrame(rows)
    paperComparison = pd.DataFrame(
        [
            {"Method": "Paper MACE SoF", "Accuracy": PAPER_TABLE1["MACE SoF"]["thingspeak"], "F1": PAPER_TABLE2["MACE SoF"]["thingspeak"]},
            {"Method": "Reproduced MACE SoF", **{metric: evaluatePredictions(prepared, results["MACE SoF"]["predictions"], results["MACE SoF"]["probabilities"])[metric] for metric in ["Accuracy", "F1"]}},
            {"Method": "RG-MACE", **{metric: evaluatePredictions(prepared, results["RG-MACE"]["predictions"], results["RG-MACE"]["probabilities"])[metric] for metric in ["Accuracy", "F1"]}},
        ]
    )
    paperComparison["DifferenceFromPaperAccuracy"] = paperComparison["Accuracy"] - PAPER_TABLE1["MACE SoF"]["thingspeak"]
    paperComparison["DifferenceFromPaperF1"] = paperComparison["F1"] - PAPER_TABLE2["MACE SoF"]["thingspeak"]
    model = context.getRgModel()
    rg = results["RG-MACE"]
    maceCorrect = prepared["classes"][results["MACE SoF"]["predictions"]] == prepared["yTest"]
    rgCorrect = prepared["classes"][rg["predictions"]] == prepared["yTest"]
    changed = rg["predictions"] != results["MACE SoF"]["predictions"]
    routing = pd.DataFrame(
        [
            {
                "MeanGate": float(np.mean(rg["gate"])),
                "RfRoutingShare": float(np.mean(rg["gate"] < 0.5) * 100.0),
                "PredictionsDifferentFromMace": int(changed.sum()),
                "MaceErrorsCorrected": int(np.sum(changed & ~maceCorrect & rgCorrect)),
                "MaceCorrectPredictionsBroken": int(np.sum(changed & maceCorrect & ~rgCorrect)),
                "ChosenC": model.chosenC_,
                "SelectedOrder": " > ".join(model.maceExpert_["order"]),
                "SelectedSigma": model.maceExpert_["sigma"],
            }
        ]
    )
    thingSpeakMetrics = metrics.loc[metrics["Dataset"] == "ThingSpeakEU"]
    _saveAndClose(plotThreeWayComparison(paperComparison, "ThingSpeak: paper, reproduction and RG-MACE"), "cleanThreeWayComparison")
    _saveAndClose(
        plotMetricBars(
            thingSpeakMetrics,
            ["RF", "Soft voting", "MACE SoF", "Soft stacking", "RG-MACE"],
            REQUIRED_METRICS,
            "Clean ThingSpeak performance",
        ),
        "cleanMetricComparison",
    )
    return {
        "metrics": savePart2Table(metrics, "exp25CleanMetrics"),
        "paperComparison": savePart2Table(paperComparison, "exp25PaperComparison"),
        "routing": savePart2Table(routing, "exp25Routing"),
    }


# evaluate one seed on naturally seen and unseen names without pooling dependent rows for inference
def _naturalSubsetRows(seed: int, jobs: int = 1) -> list[dict[str, Any]]:
    _silenceKnownWarnings()
    context = Part2Context()
    prepared = context.getPrepared(seed, "stratified", jobs)
    names = np.asarray(prepared["testInputs"]["metadata"], dtype=str)
    seen = ~_unseenNameMask(prepared)
    results = _methodResults(context, prepared, names, seed, includeStacking=False)
    rows = []
    for subset, mask in [("Seen", seen), ("Unseen", ~seen)]:
        for method in ["RF", "MACE SoF", "MACE SoF + missing-name rule", CASE_FOLDED_RULE, "RG-MACE"]:
            rows.append(
                {
                    "Seed": seed,
                    "Subset": subset,
                    "Rows": int(mask.sum()),
                    "Method": method,
                    **_subsetMetrics(prepared, results[method], mask),
                }
            )
    return rows


# refit the full pipeline for each leakage-aware split and test clean and missing-name conditions
def _splitRows(strategy: str, jobs: int = 1) -> list[dict[str, Any]]:
    _silenceKnownWarnings()
    context = Part2Context()
    prepared = context.getPrepared(100, strategy, jobs)
    cleanNames = prepared["testInputs"]["metadata"]
    missingNames = corruptMetadata(cleanNames, 0.25, seed=100, mode="missing")
    rows = []
    for condition, names in [("Clean", cleanNames), ("Missing25", missingNames)]:
        results = _methodResults(context, prepared, names, 100, strategy, includeStacking=False)
        for method in ["RF", "MACE SoF", "MACE SoF + missing-name rule", CASE_FOLDED_RULE, "RG-MACE"]:
            rows.append(
                {
                    "SplitStrategy": strategy,
                    "Condition": condition,
                    "Method": method,
                    **evaluatePredictions(prepared, results[method]["predictions"], results[method]["probabilities"]),
                }
            )
    return rows


# test the primary natural reliability failure and leakage-controlled split protocols
def runExp26NaturalFailure(context: Part2Context | None = None, jobs: int = -1) -> dict[str, pd.DataFrame]:
    _silenceKnownWarnings()
    seeds = list(PROPOSED_CONFIG["repeatedSeeds"])
    completed = Parallel(n_jobs=jobs)(delayed(_naturalSubsetRows)(seed, 1) for seed in seeds)
    subsetRows = pd.DataFrame([row for task in completed for row in task]).sort_values(["Seed", "Subset", "Method"])
    subsetSummary = subsetRows.groupby(["Subset", "Method"], as_index=False).agg(
        Rows=("Rows", "sum"),
        AccuracyMean=("Accuracy", "mean"),
        AccuracyStd=("Accuracy", "std"),
        F1Mean=("F1", "mean"),
        F1Std=("F1", "std"),
    )
    subsetStatisticsRows = []
    for subset in ["Seen", "Unseen"]:
        current = subsetRows.loc[subsetRows["Subset"] == subset]
        for baseline in ["MACE SoF", "MACE SoF + missing-name rule", CASE_FOLDED_RULE]:
            for metric in ["Accuracy", "F1"]:
                pivot = current.pivot(index="Seed", columns="Method", values=metric).sort_index()
                differences = pivot["RG-MACE"].to_numpy() - pivot[baseline].to_numpy()
                lower, upper = seedDifferenceInterval(differences, PROPOSED_CONFIG["confidenceLevel"])
                signed = pairedWilcoxon(pivot[baseline].to_numpy(), pivot["RG-MACE"].to_numpy())
                subsetStatisticsRows.append(
                    {
                        "Subset": subset,
                        "Comparison": f"RG-MACE - {baseline}",
                        "Metric": metric,
                        "MeanDifference": float(differences.mean()),
                        "Lower95": lower,
                        "Upper95": upper,
                        "WilcoxonPValue": signed["PValue"],
                    }
                )
    subsetStatistics = pd.DataFrame(subsetStatisticsRows)
    splitTasks = Parallel(n_jobs=jobs)(
        delayed(_splitRows)(strategy, 1) for strategy in PROPOSED_CONFIG["splitStrategies"]
    )
    splitRows = pd.DataFrame([row for task in splitTasks for row in task]).sort_values(
        ["SplitStrategy", "Condition", "Method"]
    )
    seed100 = subsetRows.loc[subsetRows["Seed"] == 100]
    _saveAndClose(plotSubsetComparison(seed100, "Natural seen and unseen ThingSpeak names"), "naturalNameSubsets")
    return {
        "subsetsBySeed": savePart2Table(subsetRows, "exp26NaturalSubsetsBySeed"),
        "subsetSummary": savePart2Table(subsetSummary, "exp26NaturalSubsetSummary"),
        "subsetStatistics": savePart2Table(subsetStatistics, "exp26NaturalSubsetStatistics"),
        "splitResults": savePart2Table(splitRows, "exp26LeakageAwareSplits"),
    }


# test every metadata failure mode over fixed rates and corruption seeds
def runExp27Robustness(context: Part2Context | None = None) -> dict[str, pd.DataFrame]:
    _silenceKnownWarnings()
    context = context or Part2Context()
    prepared = context.getPrepared()
    original = prepared["testInputs"]["metadata"]
    rows = []
    for mode in PROPOSED_CONFIG["corruptionModes"]:
        for rate in PROPOSED_CONFIG["robustnessRates"]:
            for corruptionSeed in PROPOSED_CONFIG["corruptionSeeds"]:
                names = corruptMetadata(original, rate, corruptionSeed, mode)
                results = _methodResults(context, prepared, names)
                for method, result in results.items():
                    metrics = evaluatePredictions(prepared, result["predictions"], result["probabilities"])
                    rows.append(
                        {
                            "Mode": mode,
                            "CorruptionRate": rate,
                            "CorruptionSeed": corruptionSeed,
                            "Method": method,
                            **metrics,
                            "RfRoutingShare": float(np.mean(result["gate"] < 0.5) * 100.0) if method == "RG-MACE" else np.nan,
                        }
                    )
    detailed = pd.DataFrame(rows)
    numeric = ["Accuracy", "Precision", "Recall", "F1", "AUC", "BalancedAccuracy", "WeightedF1", "LogLoss", "BrierScore", "RfRoutingShare"]
    summary = detailed.groupby(["Mode", "CorruptionRate", "Method"], as_index=False)[numeric].agg(["mean", "std"])
    summary.columns = ["_".join(column).rstrip("_") for column in summary.columns]
    aucRows = []
    meanRows = detailed.groupby(["Mode", "CorruptionRate", "Method"], as_index=False)[["Accuracy", "F1"]].mean()
    for (mode, method), group in meanRows.groupby(["Mode", "Method"]):
        for rangeName, rateRange in [("Primary_0_50", PROPOSED_CONFIG["primaryRateRange"]), ("Secondary_0_100", [0.0, 1.0])]:
            aucRows.append(
                {
                    "Mode": mode,
                    "Method": method,
                    "Range": rangeName,
                    "AccuracyAUC": areaUnderRobustnessCurve(group["CorruptionRate"], group["Accuracy"], rateRange),
                    "F1AUC": areaUnderRobustnessCurve(group["CorruptionRate"], group["F1"], rateRange),
                }
            )
    aucTable = pd.DataFrame(aucRows)
    plotRows = meanRows.copy()
    routing = detailed.loc[detailed["Method"] == "RG-MACE"].groupby(["Mode", "CorruptionRate"], as_index=False)["RfRoutingShare"].mean()
    _saveAndClose(plotRobustnessPanels(plotRows, "Accuracy", "Accuracy under ThingSpeak metadata corruption"), "metadataRobustnessAccuracy")
    _saveAndClose(
        plotRobustnessPanels(plotRows, "Accuracy", "Accuracy under ThingSpeak metadata corruption (zoomed per panel)", zoom=True),
        "metadataRobustnessAccuracyZoom",
    )
    _saveAndClose(plotRobustnessPanels(plotRows, "F1", "Macro F1 under ThingSpeak metadata corruption"), "metadataRobustnessF1")
    _saveAndClose(plotRoutingShare(routing, "Learned routing under metadata corruption"), "metadataRoutingShare")
    return {
        "detailed": savePart2Table(detailed, "exp27RobustnessDetailed"),
        "summary": savePart2Table(summary, "exp27RobustnessSummary"),
        "areas": savePart2Table(aucTable, "exp27RobustnessAreas"),
    }


# refit one seed and evaluate clean and realistically missing metadata
# also return how that seed's gate C was selected so constraint failures stay visible across splits
def _repeatedSeedRows(seed: int, jobs: int = 1) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _silenceKnownWarnings()
    context = Part2Context()
    prepared = context.getPrepared(seed, "stratified", jobs)
    clean = prepared["testInputs"]["metadata"]
    missing = corruptMetadata(clean, 0.25, seed=100, mode="missing")
    rows = []
    for condition, names in [("Clean", clean), ("Missing25", missing)]:
        results = _methodResults(context, prepared, names, seed, includeStacking=True)
        for method in ["RF", "Soft voting", "MACE SoF", "MACE SoF + missing-name rule", CASE_FOLDED_RULE, "Soft stacking", "RG-MACE"]:
            rows.append(
                {
                    "Seed": seed,
                    "Condition": condition,
                    "Method": method,
                    **evaluatePredictions(prepared, results[method]["predictions"], results[method]["probabilities"]),
                }
            )
    chosen = _annotatedGateSelection(context.getRgModel(seed, "stratified")).query("Chosen").iloc[0]
    gateRecord = {
        "Seed": seed,
        "ChosenC": float(chosen["C"]),
        "ConstraintMet": bool(context.getRgModel(seed, "stratified").selection_["constraintMet"]),
        "CleanF1": float(chosen["CleanF1"]),
        "MinimumAllowedCleanF1": float(chosen["MinimumAllowedCleanF1"]),
        "ShortfallFromConstraint": float(chosen["ShortfallFromConstraint"]),
        "SelectionRule": chosen["SelectionRule"],
    }
    return rows, gateRecord


# measure paired split-to-split robustness without treating test rows as independent seeds
def runExp28RepeatedSplits(context: Part2Context | None = None, jobs: int = -1) -> dict[str, pd.DataFrame]:
    _silenceKnownWarnings()
    seeds = list(PROPOSED_CONFIG["repeatedSeeds"])
    completed = Parallel(n_jobs=jobs)(delayed(_repeatedSeedRows)(seed, 1) for seed in seeds)
    detailed = pd.DataFrame([row for rows, _ in completed for row in rows]).sort_values(
        ["Seed", "Condition", "Method"]
    )
    gateSelection = pd.DataFrame([record for _, record in completed]).sort_values("Seed").reset_index(drop=True)
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC", "BalancedAccuracy", "WeightedF1", "LogLoss", "BrierScore"]
    summary = detailed.groupby(["Condition", "Method"], as_index=False)[metrics].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(column).rstrip("_") for column in summary.columns]
    summary["GateConstraintMetSeeds"] = f"{int(gateSelection['ConstraintMet'].sum())} of {len(seeds)}"
    statisticalRows = []
    for condition in ["Clean", "Missing25"]:
        conditionRows = detailed.loc[detailed["Condition"] == condition]

        # for LogLoss and BrierScore a negative difference favours RG-MACE
        for baseline in ["MACE SoF", "MACE SoF + missing-name rule", CASE_FOLDED_RULE, "Soft stacking"]:
            for metric in ["Accuracy", "F1", "AUC", "LogLoss", "BrierScore"]:
                pivot = conditionRows.pivot(index="Seed", columns="Method", values=metric).sort_index()
                differences = pivot["RG-MACE"].to_numpy() - pivot[baseline].to_numpy()
                lower, upper = seedDifferenceInterval(differences, PROPOSED_CONFIG["confidenceLevel"])
                signed = pairedWilcoxon(pivot[baseline].to_numpy(), pivot["RG-MACE"].to_numpy())
                statisticalRows.append(
                    {
                        "Condition": condition,
                        "Comparison": f"RG-MACE - {baseline}",
                        "Metric": metric,
                        "MeanDifference": float(differences.mean()),
                        "Lower95": lower,
                        "Upper95": upper,
                        "WilcoxonStatistic": signed["Statistic"],
                        "WilcoxonPValue": signed["PValue"],
                        "MedianDifference": signed["MedianDifference"],
                    }
                )
    statistics = pd.DataFrame(statisticalRows)
    _saveAndClose(
        plotAccuracyCalibrationTradeoff(
            detailed.loc[detailed["Condition"] == "Clean"],
            "Accuracy and log loss on clean data across seven splits",
        ),
        "accuracyCalibrationTradeoff",
    )
    for condition, figureName in [("Clean", "repeatedCleanAccuracy"), ("Missing25", "repeatedMissingAccuracy")]:
        figureData = detailed.loc[detailed["Condition"] == condition].copy()
        figureData["Dataset"] = condition
        _saveAndClose(
            plotSeedSpread(figureData, "Accuracy", f"{condition} accuracy across seven splits"),
            figureName,
        )
    return {
        "detailed": savePart2Table(detailed, "exp28RepeatedSplitDetailed"),
        "summary": savePart2Table(summary, "exp28RepeatedSplitSummary"),
        "statistics": savePart2Table(statistics, "exp28RepeatedSplitStatistics"),
        "gateSelectionBySeed": savePart2Table(gateSelection, "exp28GateSelectionBySeed"),
    }


# return one fixed Part 2 ablation model and its report label
def _ablationModels(context: Part2Context) -> list[tuple[str, Any, str]]:
    return [
        ("MACE SoF", context.getMaceExpert(caseFold=False), "mace"),
        ("MACE SoF + case folding", context.getMaceExpert(caseFold=True), "maceCaseFold"),
        ("MACE SoF + missing-name rule", context.getMaceExpert(caseFold=False), "rule"),
        (CASE_FOLDED_RULE, context.getRgModel(variant="full").maceExpert_, "caseFoldRule"),
        ("RG-MACE (hard gate)", context.getRgModel(variant="full"), "hard"),
        ("RG-MACE, clean-trained gate", context.getRgModel(variant="clean", useCorruption=False), "rg"),
        ("RG-MACE, no class balancing", context.getRgModel(variant="unbalanced", useClassBalance=False), "rg"),
        ("RG-MACE, no name-distance features", context.getRgModel(variant="noName", featureSubset=CONFIDENCE_FEATURES), "rg"),
        ("RG-MACE, no confidence features", context.getRgModel(variant="noConfidence", featureSubset=NAME_DISTANCE_FEATURES), "rg"),
        ("RG-MACE, case-sensitive", context.getRgModel(variant="caseSensitive", caseFold=False), "rg"),
        ("RG-MACE, accuracy-selected C", context.getRgModel(variant="accuracySelected", selectionMetric="Accuracy"), "rg"),
        ("RG-MACE", context.getRgModel(variant="full"), "rg"),
    ]


# predict one ablation variant on one metadata array
def _ablationPrediction(
    prepared: dict[str, Any],
    model: Any,
    kind: str,
    names: np.ndarray,
) -> dict[str, Any]:
    if kind == "mace":
        return predictMaceSof(prepared, names, caseFold=False, maceExpert=model)
    if kind == "maceCaseFold":
        return predictMaceSof(prepared, names, caseFold=True, maceExpert=model)
    if kind == "rule":
        return predictMissingNameRule(prepared, names, maceExpert=model)
    if kind == "caseFoldRule":
        return predictCaseFoldedMissingNameRule(prepared, names, maceExpert=model)
    result = model.predict(prepared, names)
    return _hardGateResult(result) if kind == "hard" else result


# isolate the effect of each proposed component and the amount of corruption augmentation
def runExp29Ablation(context: Part2Context | None = None) -> dict[str, pd.DataFrame]:
    _silenceKnownWarnings()
    context = context or Part2Context()
    prepared = context.getPrepared()
    cleanNames = prepared["testInputs"]["metadata"]
    unseen = _unseenNameMask(prepared)
    rows = []
    for variant, model, kind in _ablationModels(context):
        clean = _ablationPrediction(prepared, model, kind, cleanNames)
        cleanMetrics = evaluatePredictions(prepared, clean["predictions"], clean["probabilities"])
        unseenMetrics = _subsetMetrics(prepared, clean, unseen)
        missingNames = np.full(cleanNames.size, "", dtype=str)
        missing = _ablationPrediction(prepared, model, kind, missingNames)
        missingMetrics = evaluatePredictions(prepared, missing["predictions"], missing["probabilities"])
        record = {
            "Variant": variant,
            "CleanAccuracy": cleanMetrics["Accuracy"],
            "CleanF1": cleanMetrics["F1"],
            "UnseenAccuracy": unseenMetrics["Accuracy"],
            "UnseenF1": unseenMetrics["F1"],
            "Missing100Accuracy": missingMetrics["Accuracy"],
        }
        for mode in PROPOSED_CONFIG["corruptionModes"]:
            rateValues = []
            for rate in [0.0, 0.1, 0.25, 0.5]:
                scores = []
                for corruptionSeed in PROPOSED_CONFIG["corruptionSeeds"]:
                    names = corruptMetadata(cleanNames, rate, corruptionSeed, mode)
                    result = _ablationPrediction(prepared, model, kind, names)
                    scores.append(evaluatePredictions(prepared, result["predictions"], result["probabilities"])["Accuracy"])
                rateValues.append(float(np.mean(scores)))
            record[f"{mode.capitalize()}AccuracyAUC"] = areaUnderRobustnessCurve(
                np.asarray([0.0, 0.1, 0.25, 0.5]),
                np.asarray(rateValues),
                PROPOSED_CONFIG["primaryRateRange"],
            )
        rows.append(record)
    ablation = pd.DataFrame(rows)
    sensitivityRows = []
    for share in PROPOSED_CONFIG["corruptionShareSensitivity"]:
        model = context.getRgModel(variant=f"share{share}", corruptionShare=share, useCorruption=share > 0.0)
        clean = model.predict(prepared, cleanNames)
        cleanMetrics = evaluatePredictions(prepared, clean["predictions"], clean["probabilities"])
        unseenMetrics = _subsetMetrics(prepared, clean, unseen)
        missingScores = []
        for rate in [0.0, 0.1, 0.25, 0.5]:
            scores = []
            for corruptionSeed in PROPOSED_CONFIG["corruptionSeeds"]:
                names = corruptMetadata(cleanNames, rate, corruptionSeed, "missing")
                result = model.predict(prepared, names)
                scores.append(evaluatePredictions(prepared, result["predictions"], result["probabilities"])["Accuracy"])
            missingScores.append(float(np.mean(scores)))
        sensitivityRows.append(
            {
                "CorruptionShare": share,
                "CleanAccuracy": cleanMetrics["Accuracy"],
                "CleanF1": cleanMetrics["F1"],
                "UnseenAccuracy": unseenMetrics["Accuracy"],
                "MissingAccuracyAUC": areaUnderRobustnessCurve(
                    np.asarray([0.0, 0.1, 0.25, 0.5]),
                    np.asarray(missingScores),
                    PROPOSED_CONFIG["primaryRateRange"],
                ),
            }
        )
    sensitivity = pd.DataFrame(sensitivityRows)
    _saveAndClose(
        plotAblation(
            ablation,
            [
                "CleanAccuracy",
                "CleanF1",
                "UnseenAccuracy",
                "Missing100Accuracy",
                *[f"{mode.capitalize()}AccuracyAUC" for mode in PROPOSED_CONFIG["corruptionModes"]],
            ],
            "RG-MACE ablation study (colour scaled within each column)",
        ),
        "ablationStudy",
    )
    return {
        "ablation": savePart2Table(ablation, "exp29Ablation"),
        "corruptionShare": savePart2Table(sensitivity, "exp29CorruptionShareSensitivity"),
    }


# create compact corrected and broken error-transition tables
def _errorTransitions(
    yTrue: np.ndarray,
    baselinePredictions: np.ndarray,
    proposedPredictions: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(
        {"TrueClass": yTrue, "MacePrediction": baselinePredictions, "RgMacePrediction": proposedPredictions}
    )
    corrected = frame.loc[(frame["MacePrediction"] != frame["TrueClass"]) & (frame["RgMacePrediction"] == frame["TrueClass"])]
    broken = frame.loc[(frame["MacePrediction"] == frame["TrueClass"]) & (frame["RgMacePrediction"] != frame["TrueClass"])]
    columns = ["TrueClass", "MacePrediction", "RgMacePrediction"]
    correctedCounts = corrected.value_counts(columns).head(10).reset_index(name="Count")
    brokenCounts = broken.value_counts(columns).head(10).reset_index(name="Count")
    return correctedCounts, brokenCounts


# convert internal class positions to source labels for statistical metric functions
def _labelResult(prepared: dict[str, Any], result: dict[str, Any], mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    predictions = prepared["classes"][np.asarray(result["predictions"], dtype=int)]
    probabilities = np.asarray(result["probabilities"], dtype=float)
    if mask is not None:
        predictions = predictions[mask]
        probabilities = probabilities[mask]
    return {"predictions": predictions, "probabilities": probabilities, "classes": prepared["classes"]}


# provide class-level errors paired tests calibration and computational cost evidence
def runExp30EvidenceAndCost(context: Part2Context | None = None) -> dict[str, pd.DataFrame]:
    _silenceKnownWarnings()
    context = context or Part2Context()
    prepared = context.getPrepared()
    model = context.getRgModel()
    names = prepared["testInputs"]["metadata"]
    clean = _methodResults(context, prepared, names, includeStacking=False)
    missingNames = corruptMetadata(names, 0.25, 100, "missing")
    missing = _methodResults(context, prepared, missingNames, includeStacking=False)
    perClassRows = []
    for method in ["MACE SoF", "RG-MACE"]:
        table = perClassMetrics(prepared["yTest"], prepared["classes"][clean[method]["predictions"]], prepared["classes"])
        table.insert(0, "Method", method)
        table["RareClass"] = table["Support"] <= 5
        perClassRows.append(table)
    perClass = pd.concat(perClassRows, ignore_index=True)
    corrected, broken = _errorTransitions(
        prepared["yTest"],
        prepared["classes"][clean["MACE SoF"]["predictions"]],
        prepared["classes"][clean["RG-MACE"]["predictions"]],
    )
    mcnemarRows = []
    for condition, results in [("Clean", clean), ("Missing25", missing)]:
        for baseline in ["MACE SoF", "MACE SoF + missing-name rule"]:
            evidence = mcnemarExact(
                prepared["yTest"],
                prepared["classes"][results[baseline]["predictions"]],
                prepared["classes"][results["RG-MACE"]["predictions"]],
            )
            mcnemarRows.append({"Condition": condition, "Comparison": f"RG-MACE vs {baseline}", **evidence})
    mcnemar = pd.DataFrame(mcnemarRows)
    unseen = _unseenNameMask(prepared)
    bootstrapRows = []
    comparisons = [
        ("Clean", prepared["yTest"], _labelResult(prepared, clean["MACE SoF"]), _labelResult(prepared, clean["RG-MACE"]), "RG-MACE - MACE SoF"),
        ("Unseen", prepared["yTest"][unseen], _labelResult(prepared, clean["MACE SoF"], unseen), _labelResult(prepared, clean["RG-MACE"], unseen), "RG-MACE - MACE SoF"),
        ("Missing25", prepared["yTest"], _labelResult(prepared, missing["MACE SoF"]), _labelResult(prepared, missing["RG-MACE"]), "RG-MACE - MACE SoF"),
        ("Unseen", prepared["yTest"][unseen], _labelResult(prepared, clean["MACE SoF + missing-name rule"], unseen), _labelResult(prepared, clean["RG-MACE"], unseen), "RG-MACE - missing-name rule"),
        ("Clean", prepared["yTest"], _labelResult(prepared, clean[CASE_FOLDED_RULE]), _labelResult(prepared, clean["RG-MACE"]), f"RG-MACE - {CASE_FOLDED_RULE}"),
        ("Missing25", prepared["yTest"], _labelResult(prepared, missing[CASE_FOLDED_RULE]), _labelResult(prepared, missing["RG-MACE"]), f"RG-MACE - {CASE_FOLDED_RULE}"),
    ]
    for condition, truth, first, second, comparison in comparisons:
        for metric in ["Accuracy", "F1", "LogLoss"]:
            evidence = pairedBootstrapDifference(
                truth,
                first,
                second,
                metric,
                PROPOSED_CONFIG["bootstrapIterations"],
                100,
                stratify=True,
            )
            bootstrapRows.append({"Condition": condition, "Comparison": comparison, "Metric": metric, **evidence})
    bootstrap = pd.DataFrame(bootstrapRows)
    calibrationRows = []
    for method in ["MACE SoF", "RG-MACE"]:
        metrics = evaluatePredictions(prepared, clean[method]["predictions"], clean[method]["probabilities"])
        calibrationRows.append({"Method": method, "LogLoss": metrics["LogLoss"], "BrierScore": metrics["BrierScore"]})
    calibration = pd.DataFrame(calibrationRows)
    originalExpert = context.getMaceExpert(caseFold=False)
    coldOriginalExpert = {**originalExpert, "fullDdlModel": deepcopy(originalExpert["fullDdlModel"])}
    coldOriginalExpert["fullDdlModel"].distanceCache_ = {}
    started = time.perf_counter()
    predictMaceSof(prepared, names, caseFold=False, maceExpert=coldOriginalExpert)
    macePredictionSeconds = time.perf_counter() - started
    coldRgModel = copy(model)
    coldRgModel.maceExpert_ = {
        **model.maceExpert_,
        "fullDdlModel": deepcopy(model.maceExpert_["fullDdlModel"]),
    }
    coldRgModel.maceExpert_["fullDdlModel"].distanceCache_ = {}
    rgResult = coldRgModel.predict(prepared, names)
    cost = pd.DataFrame(
        [
            {"Component": "Part 1 preparation and OOF pool scores", "Seconds": context.preparationSeconds[(100, "stratified")], "MillisecondsPerStream": np.nan, "ModelBytes": np.nan},
            {"Component": "MACE expert fitting and reselection", "Seconds": model.maceExpertSeconds_, "MillisecondsPerStream": np.nan, "ModelBytes": np.nan},
            {"Component": "OOF gate dataset construction", "Seconds": model.gateDatasetSeconds_, "MillisecondsPerStream": np.nan, "ModelBytes": np.nan},
            {"Component": "Gate selection and final fitting", "Seconds": model.gateSelectionAndFitSeconds_, "MillisecondsPerStream": np.nan, "ModelBytes": len(pickle.dumps(model.gate_))},
            {"Component": "MACE SoF prediction", "Seconds": macePredictionSeconds, "MillisecondsPerStream": macePredictionSeconds * 1000.0 / prepared["yTest"].size, "ModelBytes": np.nan},
            {"Component": "RG-MACE prediction", "Seconds": rgResult["predictionSeconds"], "MillisecondsPerStream": rgResult["predictionSeconds"] * 1000.0 / prepared["yTest"].size, "ModelBytes": np.nan},
        ]
    )
    forest = bootstrap.copy()
    forest["Comparison"] = forest["Condition"] + ": " + forest["Comparison"] + " (" + forest["Metric"] + ")"
    _saveAndClose(plotDifferenceForest(forest, "Paired RG-MACE performance differences"), "pairedDifferenceForest")
    _saveAndClose(
        plotReliabilityDiagram(
            {
                method: {**_labelResult(prepared, clean[method]), "yTrue": prepared["yTest"]}
                for method in ["MACE SoF", "RG-MACE"]
            },
            "Top-label calibration on clean ThingSpeak data",
        ),
        "reliabilityDiagram",
    )
    return {
        "perClass": savePart2Table(perClass, "exp30PerClassMetrics"),
        "correctedErrors": savePart2Table(corrected, "exp30CorrectedErrors"),
        "brokenErrors": savePart2Table(broken, "exp30BrokenErrors"),
        "mcnemar": savePart2Table(mcnemar, "exp30Mcnemar"),
        "bootstrap": savePart2Table(bootstrap, "exp30BootstrapDifferences"),
        "calibration": savePart2Table(calibration, "exp30Calibration"),
        "cost": savePart2Table(cost, "exp30Cost"),
    }


# compute every pre-registered success decision from saved experiment tables without changing its target
def runPart2SuccessCriteria(allTables: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    _silenceKnownWarnings()
    clean = allTables["Exp25"]["metrics"]
    cleanTs = clean.loc[clean["Dataset"] == "ThingSpeakEU"].set_index("Method")
    repeated = allTables["Exp28"]["statistics"]
    robust = allTables["Exp27"]["detailed"]
    areas = allTables["Exp27"]["areas"]
    naturalStatistics = allTables["Exp26"]["subsetStatistics"]
    splits = allTables["Exp26"]["splitResults"]
    cost = allTables["Exp30"]["cost"].set_index("Component")
    cleanDifference = repeated.loc[
        (repeated["Condition"] == "Clean")
        & (repeated["Comparison"] == "RG-MACE - MACE SoF")
        & (repeated["Metric"] == "Accuracy")
    ].iloc[0]
    missing25 = robust.loc[
        (robust["Mode"] == "missing") & (robust["CorruptionRate"] == 0.25)
    ].groupby("Method")["Accuracy"].mean()
    missing100 = robust.loc[
        (robust["Mode"] == "missing") & (robust["CorruptionRate"] == 1.0)
    ].groupby("Method")["Accuracy"].mean()
    primaryAreas = areas.loc[areas["Range"] == "Primary_0_50"]
    areaPass = all(
        primaryAreas.loc[(primaryAreas["Mode"] == mode) & (primaryAreas["Method"] == "RG-MACE"), "AccuracyAUC"].iloc[0]
        > primaryAreas.loc[(primaryAreas["Mode"] == mode) & (primaryAreas["Method"] == "MACE SoF"), "AccuracyAUC"].iloc[0]
        for mode in PROPOSED_CONFIG["corruptionModes"]
    )
    ruleAreaPass = all(
        primaryAreas.loc[(primaryAreas["Mode"] == mode) & (primaryAreas["Method"] == "RG-MACE"), "AccuracyAUC"].iloc[0]
        > primaryAreas.loc[(primaryAreas["Mode"] == mode) & (primaryAreas["Method"] == "MACE SoF + missing-name rule"), "AccuracyAUC"].iloc[0]
        for mode in ["case", "delete", "replace"]
    )
    # report the smallest per-mode area difference and every mode's difference without changing the pass rule
    def _areaDifferences(comparator: str, modes: list[str]) -> dict[str, float]:
        return {
            mode: float(
                primaryAreas.loc[(primaryAreas["Mode"] == mode) & (primaryAreas["Method"] == "RG-MACE"), "AccuracyAUC"].iloc[0]
                - primaryAreas.loc[(primaryAreas["Mode"] == mode) & (primaryAreas["Method"] == comparator), "AccuracyAUC"].iloc[0]
            )
            for mode in modes
        }

    maceAreaDifferences = _areaDifferences("MACE SoF", list(PROPOSED_CONFIG["corruptionModes"]))
    ruleAreaDifferences = _areaDifferences("MACE SoF + missing-name rule", ["case", "delete", "replace"])
    missingRuleDifference = (
        primaryAreas.loc[(primaryAreas["Mode"] == "missing") & (primaryAreas["Method"] == "RG-MACE"), "AccuracyAUC"].iloc[0]
        - primaryAreas.loc[(primaryAreas["Mode"] == "missing") & (primaryAreas["Method"] == "MACE SoF + missing-name rule"), "AccuracyAUC"].iloc[0]
    )
    unseenAccuracy = naturalStatistics.loc[
        (naturalStatistics["Subset"] == "Unseen")
        & (naturalStatistics["Comparison"] == "RG-MACE - MACE SoF")
        & (naturalStatistics["Metric"] == "Accuracy")
    ].iloc[0]
    unseenF1 = naturalStatistics.loc[
        (naturalStatistics["Subset"] == "Unseen")
        & (naturalStatistics["Comparison"] == "RG-MACE - MACE SoF")
        & (naturalStatistics["Metric"] == "F1")
    ].iloc[0]
    unseenRuleAccuracy = naturalStatistics.loc[
        (naturalStatistics["Subset"] == "Unseen")
        & (naturalStatistics["Comparison"] == "RG-MACE - MACE SoF + missing-name rule")
        & (naturalStatistics["Metric"] == "Accuracy")
    ].iloc[0]
    grouped = splits.loc[(splits["SplitStrategy"] == "channelAware")].pivot(index="Condition", columns="Method", values="Accuracy")
    rgPrediction = float(cost.loc["RG-MACE prediction", "MillisecondsPerStream"])
    macePrediction = float(cost.loc["MACE SoF prediction", "MillisecondsPerStream"])
    rows = [
        {"Criterion": "Clean accuracy at seed 100", "Target": ">= 88.0%", "ObservedValue": cleanTs.loc["RG-MACE", "Accuracy"], "Met": cleanTs.loc["RG-MACE", "Accuracy"] >= 88.0},
        {"Criterion": "Clean macro F1 at seed 100", "Target": ">= 78.9%", "ObservedValue": cleanTs.loc["RG-MACE", "F1"], "Met": cleanTs.loc["RG-MACE", "F1"] >= 78.9},
        {"Criterion": "Seven-seed clean non-inferiority", "Target": "Lower 95% bound > -1 point", "ObservedValue": cleanDifference["Lower95"], "Met": cleanDifference["Lower95"] > -1.0},
        {"Criterion": "Clean log loss", "Target": "Better than MACE SoF", "ObservedValue": cleanTs.loc["RG-MACE", "LogLoss"] - cleanTs.loc["MACE SoF", "LogLoss"], "Met": cleanTs.loc["RG-MACE", "LogLoss"] < cleanTs.loc["MACE SoF", "LogLoss"]},
        {"Criterion": "Clean Brier score", "Target": "Better than MACE SoF", "ObservedValue": cleanTs.loc["RG-MACE", "BrierScore"] - cleanTs.loc["MACE SoF", "BrierScore"], "Met": cleanTs.loc["RG-MACE", "BrierScore"] < cleanTs.loc["MACE SoF", "BrierScore"]},
        {"Criterion": "Unseen-name accuracy", "Target": "Lower paired seed CI above 0", "ObservedValue": unseenAccuracy["Lower95"], "Met": unseenAccuracy["Lower95"] > 0.0},
        {"Criterion": "Unseen-name macro F1", "Target": "Lower paired seed CI above 0", "ObservedValue": unseenF1["Lower95"], "Met": unseenF1["Lower95"] > 0.0},
        {"Criterion": "25% missing names", "Target": ">= MACE SoF + 5 points", "ObservedValue": missing25["RG-MACE"] - missing25["MACE SoF"], "Met": missing25["RG-MACE"] - missing25["MACE SoF"] >= 5.0},
        {"Criterion": "100% missing names", "Target": ">= 74.0%", "ObservedValue": missing100["RG-MACE"], "Met": missing100["RG-MACE"] >= 74.0},
        {"Criterion": "Primary corruption AUC", "Target": "Higher for all four modes", "ObservedValue": min(maceAreaDifferences.values()), "Met": areaPass, "Detail": "; ".join(f"{mode}: {value:+.2f}" for mode, value in maceAreaDifferences.items())},
        {"Criterion": "Learned gate on unseen names", "Target": "Lower paired seed CI above missing-name rule", "ObservedValue": unseenRuleAccuracy["Lower95"], "Met": unseenRuleAccuracy["Lower95"] > 0.0},
        {"Criterion": "Learned gate on non-missing corruption", "Target": "Higher AUC than rule for case, delete and replace", "ObservedValue": min(ruleAreaDifferences.values()), "Met": ruleAreaPass, "Detail": "; ".join(f"{mode}: {value:+.2f}" for mode, value in ruleAreaDifferences.items())},
        {"Criterion": "Learned gate on missing corruption", "Target": "Not worse than missing-name rule", "ObservedValue": missingRuleDifference, "Met": missingRuleDifference >= 0.0},
        {"Criterion": "Channel split clean", "Target": "No more than 1 point below MACE", "ObservedValue": grouped.loc["Clean", "RG-MACE"] - grouped.loc["Clean", "MACE SoF"], "Met": grouped.loc["Clean", "RG-MACE"] - grouped.loc["Clean", "MACE SoF"] >= -1.0},
        {"Criterion": "Channel split 25% missing", "Target": "Higher than MACE SoF", "ObservedValue": grouped.loc["Missing25", "RG-MACE"] - grouped.loc["Missing25", "MACE SoF"], "Met": grouped.loc["Missing25", "RG-MACE"] > grouped.loc["Missing25", "MACE SoF"]},
        {"Criterion": "Prediction overhead", "Target": "Under 10%", "ObservedValue": (rgPrediction / macePrediction - 1.0) * 100.0, "Met": rgPrediction <= 1.10 * macePrediction},
    ]
    criteria = pd.DataFrame(rows)
    criteria["Detail"] = criteria["Detail"].fillna("")
    return savePart2Table(criteria, "part2SuccessCriteria")
