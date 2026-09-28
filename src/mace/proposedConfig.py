# stores every pre-registered Part 2 setting and isolated output path
from pathlib import Path

from .paths import RESULTS_DIR


# keep design choices fixed before test evaluation and tune only the gate regularisation on training data
PROPOSED_CONFIG = {
    "dataset": "thingspeak",
    "seed": 100,
    "repeatedSeeds": [7, 21, 42, 77, 100, 123, 202],
    "splitStrategies": ["stratified", "groupAware", "channelAware"],
    "innerFolds": 5,
    "caseFold": True,
    "corruptionModes": ["missing", "case", "delete", "replace"],
    "corruptionShare": 0.5,
    "corruptionShareSensitivity": [0.0, 0.25, 0.5, 1.0],
    "gateCGrid": [0.01, 0.1, 1.0, 10.0],
    "cleanToleranceF1": 0.5,
    "robustnessRates": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    "primaryRateRange": [0.0, 0.5],
    "corruptionSeeds": [100, 101, 102],
    "bootstrapIterations": 2000,
    "confidenceLevel": 0.95,
}

PART2_TABLES_DIR = RESULTS_DIR / "tables" / "part2"
PART2_FIGURES_DIR = RESULTS_DIR / "figures" / "part2"
PART2_LOGS_DIR = RESULTS_DIR / "logs"


# create only the isolated Part 2 output directories
def ensurePart2Directories() -> None:
    for directory in (PART2_TABLES_DIR, PART2_FIGURES_DIR, PART2_LOGS_DIR):
        Path(directory).mkdir(parents=True, exist_ok=True)
