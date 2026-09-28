# provides the single reproducible entry point for every Part 1 experiment from Exp1 to Exp21
from __future__ import annotations

import argparse
import json
import platform
import re
import time
import warnings
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning

from mace.additionalExperiments import runDataAudit, runErrorAnalysis, runTimeSeriesBaselines, runTunedBOS, saveTable
from mace.config import CRITICAL_ANALYSIS_CONFIG
from mace.criticalAnalysis import (
    prepareBaselines,
    runCalibrationEvidence,
    runDomainAblation,
    runLabelNoise,
    runLeakageAwareSplits,
    runMetadataRobustness,
    runPaperCodeSensitivity,
    runRepeatedSplits,
    runTrainingOnlyBruteForce,
)
from mace.paths import LOGS_DIR, ensureOutputDirectories
from mace.plotting import (
    applyPlotStyle,
    plotHeuristicDistributions,
    plotPaperComparison,
    plotRobustness,
    plotSeedSpread,
    plotTopKPanels,
    saveFigure,
)
from mace.publishedResults import PAPER_DATASET_ORDER, PAPER_SHORT_NAMES
from mace.reproductionPipeline import ddlNLPDiagnostic, runFullPaperReproduction


# global Part 1 labels shared by the notebooks documentation and this runner
EXPERIMENTS = {
    "Exp1": "Dataset Structure and Dimensions",
    "Exp2": "Class Distribution",
    "Exp3": "Constant Series, Missing Moments and Duplicate Audit",
    "Exp4": "Bag-of-Summaries Feature Verification",
    "Exp5": "Standalone Bag-of-Summaries Classifiers",
    "Exp6": "Training-Only BOS Hyperparameter Optimisation",
    "Exp7": "Time-Series Methods from Figure 3(a)",
    "Exp8": "DDL-NLP Metadata Classifier",
    "Exp9": "Five-Fold Top-k Accuracy Curves",
    "Exp10": "Classifier Ordering and Cascade Parameter Tuning",
    "Exp11": "Main MACE Reproduction of Tables I and II",
    "Exp12": "Heuristic Evaluation Across 26 Classifier Pools",
    "Exp13": "Error Analysis, Uncertainty and Runtime",
    "Exp14": "Repeated-Split Robustness",
    "Exp15": "Duplicate-Aware and Channel-Grouped Evaluation",
    "Exp16": "Class Balance and Probability Quality",
    "Exp17": "ThingSpeak Metadata Robustness",
    "Exp18": "Information-Domain Ablation",
    "Exp19": "Paper-to-Code Sensitivity Analysis",
    "Exp20": "Training-Label Noise Analysis",
    "Exp21": "Training-Only Exhaustive SoF Selection",
}
FIRST_RESERVED_EXPERIMENT = 22
AUDIT_EXPERIMENTS = {"Exp1", "Exp2", "Exp3", "Exp4"}
CORE_EXPERIMENTS = {"Exp5", "Exp6", "Exp7", "Exp8", "Exp9", "Exp10", "Exp11", "Exp12", "Exp13"}
POOL_ANALYSIS_EXPERIMENTS = {"Exp12"}
QUICK_EXPERIMENTS = {"Exp11"}
DATASET_KEYS = [("UrbObs", "Urban Observatory"), ("Swissex", "Swissex"), ("TS", "ThingSpeak")]
LABEL_PATTERN = re.compile(r"^(?:exp|e)?0*(\d+)$")


# convert concise command-line aliases such as e05 or exp5 to the global experiment labels
def normaliseExperiment(value: str) -> str:
    cleaned = value.strip().lower().replace("_", "").replace("-", "")
    match = LABEL_PATTERN.match(cleaned)
    if match is None:
        raise argparse.ArgumentTypeError(f"Unknown experiment label: {value}")
    number = int(match.group(1))
    if number >= FIRST_RESERVED_EXPERIMENT:
        raise argparse.ArgumentTypeError(f"{value} is reserved for the Part 2 proposed solution")
    label = f"Exp{number}"
    if label not in EXPERIMENTS:
        raise argparse.ArgumentTypeError(f"Unknown experiment label: {value}")
    return label


# read quick and selective execution options
def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the MACE Part 1 research workflow from Exp1 to Exp21",
        epilog="Exp1 to Exp4 audit the datasets. Exp5 to Exp13 reproduce the paper. "
        "Exp14 to Exp21 form the critical analysis. Exp22 onward is reserved for Part 2.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--quick",
        action="store_true",
        help="produce the main Tables I and II evidence only which corresponds to Exp11",
    )
    modes.add_argument(
        "--only",
        nargs="+",
        type=normaliseExperiment,
        metavar="LABEL",
        help="run selected experiments from Exp1 to Exp21 such as Exp5 Exp11 or e05 e11",
    )
    return parser.parse_args()


# resolve the requested experiment set without changing the protocol
def selectedExperiments(arguments: argparse.Namespace) -> set[str]:
    if arguments.quick:
        return set(QUICK_EXPERIMENTS)
    if arguments.only:
        return set(arguments.only)
    return set(EXPERIMENTS)


# order labels numerically so Exp10 follows Exp9 in logs
def sortedLabels(labels: set[str]) -> list[str]:
    return sorted(labels, key=lambda label: int(label[3:]))


# save one figure and release its memory immediately
def saveAndClose(figure: plt.Figure, name: str) -> None:
    saveFigure(figure, name)
    plt.close(figure)


# reshape Figure 3(a) into the same direct paper-versus-mine layout and plot the exact methods
def saveFigureThreeAComparison(timeSeriesTable: pd.DataFrame) -> pd.DataFrame:
    exactOrProxy = timeSeriesTable.copy()
    exactOrProxy["MethodLabel"] = exactOrProxy["Method"] + " " + exactOrProxy["Representation"]
    rows = []
    for methodLabel, group in exactOrProxy.groupby("MethodLabel", sort=False):
        row = {"Method": methodLabel}
        for datasetName in PAPER_DATASET_ORDER:
            shortName = PAPER_SHORT_NAMES[datasetName]
            displayName = {"urban": "UrbanObservatory", "swissex": "Swissex", "thingspeak": "ThingSpeakEU"}[datasetName]
            matched = group.loc[group["Dataset"] == displayName]
            if matched.empty:
                continue
            row[f"{shortName} Paper"] = float(matched.iloc[0]["PaperAccuracy"])
            row[f"{shortName} Mine"] = float(matched.iloc[0]["Accuracy"])
            row[f"{shortName} Difference"] = round(row[f"{shortName} Mine"] - row[f"{shortName} Paper"], 1)
        rows.append(row)
    comparison = saveTable(pd.DataFrame(rows), "figure3aComparison")

    # plot only the exact methods with the same labels and order as notebook 2 because proxies are not comparable
    figureLabels = {
        "1NN-ED raw": "1NN-ED (raw)",
        "1NN-ED znorm": "1NN-ED (z-norm)",
        "1NN-DTW raw": "1NN-DTW (raw)",
        "1NN-DTW znorm": "1NN-DTW (z-norm)",
    }
    exactRows = comparison.loc[comparison["Method"].isin(figureLabels)].set_index("Method")
    exactRows = exactRows.reindex([label for label in figureLabels if label in exactRows.index]).reset_index()
    exactRows["Method"] = exactRows["Method"].map(figureLabels)
    figure = plotPaperComparison(exactRows, DATASET_KEYS, "Figure 3(a): paper and mine", "Accuracy (%)")
    saveAndClose(figure, "figure3aPaperVsMine")
    return comparison


# exp5 saves the Figure 3(b) paper-versus-mine comparison
def runExp5(results: dict[str, Any]) -> None:
    figure = plotPaperComparison(
        results["figureThreeBComparison"], DATASET_KEYS, "Figure 3(b): paper and mine", "Accuracy (%)"
    )
    saveAndClose(figure, "figure3bPaperVsMine")


# exp6 tests the optimisation claim with training-only grids
def runExp6(results: dict[str, Any]) -> None:
    runTunedBOS(results["datasets"])


# exp7 evaluates the Figure 3(a) time-series methods and their comparison figure
def runExp7(results: dict[str, Any]) -> None:
    saveFigureThreeAComparison(runTimeSeriesBaselines(results["datasets"]))


# exp8 separates exact-name DDL-NLP evidence from unseen-name generalisation
def runExp8(results: dict[str, Any]) -> None:
    saveTable(ddlNLPDiagnostic(results["datasets"]["thingspeak"]["prepared"]), "ddlNLPDiagnostic")


# exp9 saves the Figure 4 top-k accuracy panels
def runExp9(results: dict[str, Any]) -> None:
    saveAndClose(plotTopKPanels(results["topKCurves"], "Figure 4: five-fold CV top-k accuracy"), "topKAccuracyCurves")


# exp11 saves the Table I and Table II paper-versus-mine figures
def runExp11(results: dict[str, Any]) -> None:
    accuracyFigure = plotPaperComparison(
        results["table1Accuracy"], DATASET_KEYS, "Table I accuracy: paper and mine", "Accuracy (%)"
    )
    f1Figure = plotPaperComparison(
        results["table2MacroF1"], DATASET_KEYS, "Table II macro F1: paper and mine", "Macro F1 (%)"
    )
    saveAndClose(accuracyFigure, "table1PaperVsMine")
    saveAndClose(f1Figure, "table2PaperVsMine")


# exp12 saves the Figure 5 and Figure 6 heuristic distributions
def runExp12(results: dict[str, Any]) -> None:
    normalisedFigure = plotHeuristicDistributions(
        results["heuristicPoolAnalysis"],
        "NormalisedAccuracy",
        "Figure 5: heuristic accuracy relative to the paper oracle",
        "Normalised accuracy",
    )
    percentileFigure = plotHeuristicDistributions(
        results["heuristicPoolAnalysis"],
        "OraclePercentile",
        "Figure 6: heuristic percentile within exhaustive configurations",
        "Oracle percentile",
    )
    saveAndClose(normalisedFigure, "normalisedAccuracyDistributions")
    saveAndClose(percentileFigure, "oraclePercentileDistributions")


# exp13 saves per-class confusion and bootstrap evidence for the best valid method
def runExp13(results: dict[str, Any]) -> None:
    runErrorAnalysis(results["datasets"])


# exp14 repeats the valid protocol over seven seeds and saves the same accuracy spread figure as notebook 3
def runExp14() -> None:
    detailed, _ = runRepeatedSplits(CRITICAL_ANALYSIS_CONFIG["randomSeeds"])
    figure = plotSeedSpread(detailed, "Accuracy", "Accuracy across seven stratified splits")
    saveAndClose(figure, "repeatedSplitAccuracy")


# exp17 measures ThingSpeak metadata robustness and saves the same accuracy figure as notebook 3
def runExp17(baselinePrepared: dict[str, Any]) -> None:
    metadata = runMetadataRobustness(
        CRITICAL_ANALYSIS_CONFIG["metadataCorruptionRates"],
        baselinePrepared=baselinePrepared,
    )
    figure = plotRobustness(
        metadata,
        "CorruptionRate",
        "Accuracy",
        "Mode",
        "ThingSpeak accuracy under metadata corruption",
        xLabel="Share of test names corrupted",
        yLabel="Accuracy (%)",
    )
    saveAndClose(figure, "metadataRobustness")


# exp20 measures training-label noise sensitivity and saves the same accuracy figure as notebook 3
def runExp20() -> None:
    noise = runLabelNoise(CRITICAL_ANALYSIS_CONFIG["labelNoiseRates"])
    figure = plotRobustness(
        noise,
        "NoiseRate",
        "Accuracy",
        "Dataset",
        "Accuracy under training-label noise",
        xLabel="Training-label noise rate",
        yLabel="Accuracy (%)",
    )
    saveAndClose(figure, "labelNoise")


# exp10 has no extra step because its ordering and tuning tables come from the shared reproduction
CORE_STEPS: dict[str, Callable[[dict[str, Any]], None]] = {
    "Exp5": runExp5,
    "Exp6": runExp6,
    "Exp7": runExp7,
    "Exp8": runExp8,
    "Exp9": runExp9,
    "Exp11": runExp11,
    "Exp12": runExp12,
    "Exp13": runExp13,
}
# each critical step receives a provider of the shared seed-100 baselines used by Exp16 Exp17 and Exp19
CRITICAL_STEPS: dict[str, Callable[[Callable[[], dict[str, Any]]], Any]] = {
    "Exp14": lambda baselines: runExp14(),
    "Exp15": lambda baselines: runLeakageAwareSplits(),
    "Exp16": lambda baselines: runCalibrationEvidence(baselines()),
    "Exp17": lambda baselines: runExp17(baselines()),
    "Exp18": lambda baselines: runDomainAblation(),
    "Exp19": lambda baselines: runPaperCodeSensitivity(baselines()),
    "Exp20": lambda baselines: runExp20(),
    "Exp21": lambda baselines: runTrainingOnlyBruteForce(),
}


# capture exact library versions beside the generated results
def environmentRecord() -> dict[str, str]:
    packages = ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "seaborn", "pyxDamerauLevenshtein", "numba"]
    record = {"Python": platform.python_version()}
    for package in packages:
        try:
            record[package] = version(package)
        except PackageNotFoundError:
            record[package] = "not installed"
    return record


# execute selected experiments and save a machine-readable timing log
def main() -> None:
    arguments = parseArguments()
    selected = selectedExperiments(arguments)
    # every shared Exp5 to Exp13 table carries oracle columns so the oracle always joins the core run
    includePaperOracle = bool(selected & CORE_EXPERIMENTS)
    includePoolAnalysis = bool(selected & POOL_ANALYSIS_EXPERIMENTS)

    # logistic regression is fitted on unscaled features to match the authors' released code,
    # so it does not converge on ThingSpeak; this is reported in the Exp5 findings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # ridge on unscaled features is ill-conditioned and ThingSpeak's rarest class has only 3 training streams
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    warnings.filterwarnings("ignore", message="The least populated class in y", category=UserWarning)
    applyPlotStyle()
    ensureOutputDirectories()
    timings: dict[str, float] = {}
    executionGroups: dict[str, list[str]] = {}
    workflowStarted = time.perf_counter()

    # one shared audit produces every Exp1 to Exp4 table
    auditSelected = selected & AUDIT_EXPERIMENTS
    if auditSelected:
        started = time.perf_counter()
        runDataAudit()
        timings["DatasetAudit"] = time.perf_counter() - started
        executionGroups["DatasetAudit"] = sortedLabels(auditSelected)

    # one cached reproduction serves every selected Exp5 to Exp13 step
    results: dict[str, Any] | None = None
    coreSelected = selected & CORE_EXPERIMENTS
    if coreSelected:
        started = time.perf_counter()
        results = runFullPaperReproduction(
            includeBruteForce=includePaperOracle,
            includePoolAnalysis=includePoolAnalysis,
        )
        timings["CoreReproduction"] = time.perf_counter() - started
        executionGroups["CoreReproduction"] = sortedLabels(coreSelected)
        for label in sortedLabels(coreSelected & set(CORE_STEPS)):
            started = time.perf_counter()
            CORE_STEPS[label](results)
            timings[label] = time.perf_counter() - started

    baselineCache: dict[str, Any] = {}

    # prepare the seed-100 baselines at most once for Exp16 Exp17 and Exp19 and reuse the core preparation when it ran
    def sharedBaselines() -> dict[str, Any]:
        if not baselineCache:
            if results is not None:
                baselineCache.update({name: results["datasets"][name]["prepared"] for name in PAPER_DATASET_ORDER})
            else:
                baselineCache.update(prepareBaselines())
        return baselineCache

    # each critical-analysis experiment performs its own training-only selection
    for label in sortedLabels(selected & set(CRITICAL_STEPS)):
        started = time.perf_counter()
        CRITICAL_STEPS[label](sharedBaselines)
        timings[label] = time.perf_counter() - started

    log = {
        "selectedExperiments": sortedLabels(selected),
        "experimentTitles": {label: EXPERIMENTS[label] for label in sortedLabels(selected)},
        "mode": "quick" if arguments.quick else "only" if arguments.only else "full",
        "paperOracleIncluded": includePaperOracle,
        "poolAnalysisIncluded": includePoolAnalysis,
        "executionGroups": executionGroups,
        "environment": environmentRecord(),
        "timingsSeconds": {name: round(seconds, 3) for name, seconds in timings.items()},
        "totalSeconds": round(time.perf_counter() - workflowStarted, 3),
    }
    (LOGS_DIR / "runLog.json").write_text(json.dumps(log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
