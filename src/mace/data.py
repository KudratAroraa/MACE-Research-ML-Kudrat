# loads the three open IoT datasets and creates leakage-aware deterministic splits
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .config import loadConfig
from .paths import DATA_DIR, METRICS_DIR, ensureOutputDirectories


DATASET_ALIASES = {
    "thingspeak": "thingspeak",
    "thingspeakeu": "thingspeak",
    "swissex": "swissex",
    "urban": "urban",
    "urbanobservatory": "urban",
}


# hold raw time series, labels and metadata without mixing representations
@dataclass(frozen=True)
class DatasetBundle:
    key: str
    displayName: str
    ids: np.ndarray
    series: np.ndarray
    labels: np.ndarray
    metadata: np.ndarray
    sourcePath: Path
    channels: np.ndarray | None = None

    @property
    def classes(self) -> np.ndarray:
        return np.unique(self.labels)

    @property
    def shape(self) -> tuple[int, int]:
        return self.series.shape


# hold aligned train and test indices for every input representation
@dataclass(frozen=True)
class DatasetSplit:
    trainIndex: np.ndarray
    testIndex: np.ndarray
    seed: int
    strategy: str


# convert display names and short names to one internal dataset key
def normaliseDatasetKey(name: str) -> str:
    key = name.lower().replace("_", "").replace("-", "").replace(" ", "")
    if key not in DATASET_ALIASES:
        raise KeyError(f"Unknown dataset: {name}")
    return DATASET_ALIASES[key]


# load one headerless source CSV using its configured time-series length
def loadDataset(name: str, dataDir: Path | None = None) -> DatasetBundle:
    config = loadConfig()
    key = normaliseDatasetKey(name)
    settings = config["datasets"][key]
    sourcePath = (dataDir or DATA_DIR) / settings["file"]
    frame = pd.read_csv(sourcePath, header=None, dtype=str, keep_default_na=False)
    seriesLength = int(settings["seriesLength"])
    seriesStart = int(settings["seriesStartIndex"])
    labelIndex = int(settings["labelIndex"])
    metadataIndex = int(settings["metadataIndex"])
    expectedColumns = seriesStart + seriesLength
    if frame.shape[1] != expectedColumns:
        raise ValueError(
            f"{settings['displayName']} has {frame.shape[1]} columns but {expectedColumns} were expected"
        )

    # parse numeric fields only after the file shape has been validated
    ids = frame.iloc[:, 0].astype(str).to_numpy()
    metadata = frame.iloc[:, metadataIndex].astype(str).replace("null", "").fillna("").to_numpy()
    channels = frame.iloc[:, 2].astype(str).to_numpy()
    labels = pd.to_numeric(frame.iloc[:, labelIndex], errors="raise").to_numpy(int)
    # use correctly rounded Python float parsing because pd.to_numeric can differ in the last bit
    series = frame.iloc[:, seriesStart : seriesStart + seriesLength].to_numpy(dtype=str).astype(float)
    if not np.isfinite(series).all():
        raise ValueError(f"{settings['displayName']} contains non-finite time-series values")
    if pd.Index(ids).has_duplicates:
        raise ValueError(f"{settings['displayName']} contains duplicate observation identifiers")
    return DatasetBundle(key, settings["displayName"], ids, series, labels, metadata, sourcePath, channels)


# choose the grouped stratified fold closest to the requested test share
def _bestStratifiedGroupFold(
    labels: np.ndarray,
    groups: np.ndarray,
    testSize: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitCount = max(2, int(round(1.0 / testSize)))
    splitter = StratifiedGroupKFold(n_splits=splitCount, shuffle=True, random_state=seed)
    index = np.arange(labels.size)
    overall = pd.Series(labels).value_counts(normalize=True).sort_index()
    candidates = []
    for trainIndex, testIndex in splitter.split(index, labels, groups):
        testDistribution = (
            pd.Series(labels[testIndex]).value_counts(normalize=True).reindex(overall.index, fill_value=0.0)
        )
        sizeDifference = abs(testIndex.size / labels.size - testSize)
        distributionDifference = float(np.abs(testDistribution - overall).mean())
        candidates.append((sizeDifference + distributionDifference, trainIndex, testIndex))
    _, trainIndex, testIndex = min(candidates, key=lambda item: item[0])
    return np.asarray(trainIndex), np.asarray(testIndex)


# create stable content fingerprints for duplicate-aware analysis
def seriesFingerprints(series: np.ndarray, precision: int = 12) -> np.ndarray:
    rounded = np.round(np.asarray(series, dtype=float), precision)
    return np.array([hashlib.sha256(row.tobytes()).hexdigest() for row in rounded])


# create a standard stratified split or a duplicate-group-aware split
def splitDataset(
    bundle: DatasetBundle,
    testSize: float = 0.30,
    seed: int = 100,
    strategy: str = "stratified",
) -> DatasetSplit:
    index = np.arange(bundle.series.shape[0])
    if strategy == "stratified":
        trainIndex, testIndex = train_test_split(
            index,
            test_size=testSize,
            random_state=seed,
            stratify=bundle.labels,
        )
    elif strategy == "groupAware":
        groups = seriesFingerprints(bundle.series)
        trainIndex, testIndex = _bestStratifiedGroupFold(bundle.labels, groups, testSize, seed)
    elif strategy == "channelAware":
        if bundle.channels is None:
            raise ValueError("channelAware splitting requires channel identifiers")
        trainIndex, testIndex = _bestStratifiedGroupFold(bundle.labels, bundle.channels, testSize, seed)
    else:
        raise ValueError("strategy must be 'stratified', 'groupAware' or 'channelAware'")
    return DatasetSplit(np.asarray(trainIndex), np.asarray(testIndex), seed, strategy)


# save the exact observation identifiers used by a deterministic split
def saveSplitManifest(bundle: DatasetBundle, split: DatasetSplit) -> Path:
    ensureOutputDirectories()
    assignment = np.full(bundle.ids.shape[0], "test", dtype=object)
    assignment[split.trainIndex] = "train"
    manifest = pd.DataFrame(
        {
            "dataset": bundle.displayName,
            "id": bundle.ids,
            "label": bundle.labels,
            "partition": assignment,
            "seed": split.seed,
            "strategy": split.strategy,
        }
    )
    path = METRICS_DIR / f"{bundle.key}SplitManifest.csv"
    manifest.to_csv(path, index=False)
    return path


# summarise dimensions, imbalance, constant series and duplicate risks
def datasetAudit(bundle: DatasetBundle, split: DatasetSplit | None = None) -> dict[str, object]:
    from .features import extractBOSFeatures

    fingerprints = seriesFingerprints(bundle.series)
    duplicateMask = pd.Series(fingerprints).duplicated(keep=False).to_numpy()
    rawFeatures = extractBOSFeatures(bundle.series, "nan")
    undefinedMoments = ~np.isfinite(rawFeatures[["Kurtosis", "Skewness"]].to_numpy())
    conflictingGroups = 0
    for fingerprint in np.unique(fingerprints[duplicateMask]):
        groupLabels = np.unique(bundle.labels[fingerprints == fingerprint])
        conflictingGroups += int(groupLabels.size > 1)
    audit: dict[str, object] = {
        "dataset": bundle.displayName,
        "observations": int(bundle.series.shape[0]),
        "seriesLength": int(bundle.series.shape[1]),
        "classes": int(bundle.classes.size),
        "exactConstantSeries": int(np.sum(np.ptp(bundle.series, axis=1) == 0)),
        "zeroStdSeries": int(np.sum(np.std(bundle.series, axis=1) == 0)),
        "undefinedMomentRows": int(np.sum(undefinedMoments.any(axis=1))),
        "duplicateGroups": int(pd.Series(fingerprints).value_counts().gt(1).sum()),
        "conflictingDuplicateGroups": int(conflictingGroups),
        "largestClassShare": float(pd.Series(bundle.labels).value_counts(normalize=True).max()),
    }
    if split is not None:
        trainGroups = set(fingerprints[split.trainIndex])
        testGroups = set(fingerprints[split.testIndex])
        audit["crossPartitionDuplicateGroups"] = len(trainGroups.intersection(testGroups))
    return audit


# return class counts and shares in a report-ready table
def classDistribution(bundle: DatasetBundle) -> pd.DataFrame:
    counts = pd.Series(bundle.labels).value_counts().sort_index()
    return pd.DataFrame(
        {
            "Class": counts.index,
            "Count": counts.values,
            "Share": counts.values / counts.values.sum(),
        }
    )
