# prepares aligned input representations and ordered candidate-model specifications
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .classifiers import buildCandidatePool
from .data import DatasetBundle, DatasetSplit
from .features import extractBOSFeatures


# describe one classifier and the input representation it consumes
@dataclass(frozen=True)
class ModelSpec:
    name: str
    domain: str
    estimator: Any


# create aligned train and test representations for every information domain
def prepareRepresentations(
    bundle: DatasetBundle,
    split: DatasetSplit,
    constantMomentPolicy: str = "zero",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    extractionPolicy = "nan" if constantMomentPolicy == "medianTraining" else constantMomentPolicy

    # copy because pandas 3 returns a read-only view and the median policy fills missing moments in place
    bos = extractBOSFeatures(bundle.series, extractionPolicy).to_numpy(copy=True)
    if constantMomentPolicy == "medianTraining":
        trainingMedian = np.nanmedian(bos[split.trainIndex], axis=0)
        missingRows, missingColumns = np.where(~np.isfinite(bos))
        bos[missingRows, missingColumns] = trainingMedian[missingColumns]
    trainInputs = {
        "series": bundle.series[split.trainIndex],
        "bos": bos[split.trainIndex],
        "metadata": bundle.metadata[split.trainIndex],
    }
    testInputs = {
        "series": bundle.series[split.testIndex],
        "bos": bos[split.testIndex],
        "metadata": bundle.metadata[split.testIndex],
    }
    return trainInputs, testInputs, bundle.labels[split.trainIndex], bundle.labels[split.testIndex]


# convert the dataset candidate pool into ordered MACE model specifications
def modelSpecifications(
    datasetKey: str,
    order: list[str] | None = None,
    seed: int = 100,
    ddlDistancePower: int = 2,
    ddlDistanceVariant: str = "restricted",
    jobs: int = -1,
) -> list[ModelSpec]:
    pool = buildCandidatePool(datasetKey, seed, ddlDistancePower, ddlDistanceVariant, jobs)
    selectedOrder = order or list(pool)
    return [ModelSpec(name, pool[name][0], pool[name][1]) for name in selectedOrder]
