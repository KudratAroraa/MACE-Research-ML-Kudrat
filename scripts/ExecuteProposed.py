# provides the reproducible command-line entry point for Part 2 experiments Exp22 to Exp30
from __future__ import annotations

import argparse
import json
import platform
import re
import time
from importlib.metadata import PackageNotFoundError, version

from mace.plotting import applyPlotStyle
from mace.proposedConfig import PART2_LOGS_DIR, ensurePart2Directories
from mace.proposedExperiments import (
    Part2Context,
    runExp22Motivation,
    runExp23GateDataset,
    runExp24GateSelection,
    runExp25CleanComparison,
    runExp26NaturalFailure,
    runExp27Robustness,
    runExp28RepeatedSplits,
    runExp29Ablation,
    runExp30EvidenceAndCost,
    runPart2SuccessCriteria,
)


EXPERIMENTS = {
    "Exp22": "Failure Case and Research Motivation",
    "Exp23": "Out-of-Fold Reliability Dataset",
    "Exp24": "Gate Training and Selection",
    "Exp25": "Clean Test Comparison",
    "Exp26": "Natural Reliability and Leakage-Aware Splits",
    "Exp27": "Metadata Robustness",
    "Exp28": "Repeated-Split Robustness",
    "Exp29": "Ablation Study",
    "Exp30": "Error Analysis, Statistical Evidence and Cost",
}
LABEL_PATTERN = re.compile(r"^(?:exp|e)?0*(\d+)$")


# accept concise aliases while keeping the global Part 2 labels unambiguous
def normaliseExperiment(value: str) -> str:
    cleaned = value.strip().lower().replace("_", "").replace("-", "")
    match = LABEL_PATTERN.match(cleaned)
    if match is None:
        raise argparse.ArgumentTypeError(f"Unknown experiment label: {value}")
    label = f"Exp{int(match.group(1))}"
    if label not in EXPERIMENTS:
        raise argparse.ArgumentTypeError(f"Part 2 supports Exp22 to Exp30, not {value}")
    return label


# read full quick and selective execution modes
def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the MACE Part 2 workflow from Exp22 to Exp30")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--quick", action="store_true", help="run the clean seed-100 comparison in Exp25 only")
    modes.add_argument(
        "--only",
        nargs="+",
        type=normaliseExperiment,
        metavar="LABEL",
        help="run selected experiments such as Exp22 Exp27 or e22 e27",
    )
    return parser.parse_args()


# resolve requested labels in numerical order
def selectedExperiments(arguments: argparse.Namespace) -> list[str]:
    if arguments.quick:
        return ["Exp25"]
    if arguments.only:
        return sorted(set(arguments.only), key=lambda label: int(label[3:]))
    return list(EXPERIMENTS)


# report installed library versions without introducing any new dependency
def packageVersions() -> dict[str, str]:
    output = {}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn", "joblib"]:
        try:
            output[package] = version(package)
        except PackageNotFoundError:
            output[package] = "not installed"
    return output


# execute selected runners through one shared in-memory context and write the run provenance log
def main() -> None:
    arguments = parseArguments()
    labels = selectedExperiments(arguments)
    ensurePart2Directories()
    applyPlotStyle()
    context = Part2Context()
    runners = {
        "Exp22": runExp22Motivation,
        "Exp23": runExp23GateDataset,
        "Exp24": runExp24GateSelection,
        "Exp25": runExp25CleanComparison,
        "Exp26": runExp26NaturalFailure,
        "Exp27": runExp27Robustness,
        "Exp28": runExp28RepeatedSplits,
        "Exp29": runExp29Ablation,
        "Exp30": runExp30EvidenceAndCost,
    }
    started = time.perf_counter()
    timings = {}
    tables = {}
    for label in labels:
        experimentStarted = time.perf_counter()
        tables[label] = runners[label](context)
        timings[label] = time.perf_counter() - experimentStarted
    if set(EXPERIMENTS).issubset(tables):
        tables["SuccessCriteria"] = {"criteria": runPart2SuccessCriteria(tables)}
    log = {
        "selectedExperiments": labels,
        "experimentTitles": {label: EXPERIMENTS[label] for label in labels},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packageVersions(),
        "experimentSeconds": timings,
        "totalSeconds": time.perf_counter() - started,
    }
    (PART2_LOGS_DIR / "proposedRunLog.json").write_text(
        json.dumps(log, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
