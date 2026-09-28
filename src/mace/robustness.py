# provides controlled stress tests for metadata, labels and repeated data splits
from __future__ import annotations

import numpy as np


# apply reproducible character-level corruption to a selected share of stream names
def corruptMetadata(
    metadata: np.ndarray,
    rate: float,
    seed: int = 100,
    mode: str = "delete",
) -> np.ndarray:
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be between zero and one")
    rng = np.random.default_rng(seed)
    output = np.asarray(metadata, dtype=str).copy()
    selected = rng.choice(output.size, size=int(round(rate * output.size)), replace=False)
    alphabet = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    for index in selected:
        value = output[index]
        if mode == "missing":
            output[index] = ""
        elif mode == "case":
            output[index] = value.swapcase()
        elif mode == "delete" and value:
            position = rng.integers(0, len(value))
            output[index] = value[:position] + value[position + 1 :]
        elif mode == "replace" and value:
            position = rng.integers(0, len(value))
            output[index] = value[:position] + rng.choice(alphabet) + value[position + 1 :]
        elif mode not in {"missing", "case", "delete", "replace"}:
            raise ValueError("Unsupported metadata corruption mode")
    return output


# replace selected training labels with a different observed class
def injectLabelNoise(labels: np.ndarray, rate: float, seed: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = np.asarray(labels).copy()
    classes = np.unique(noisy)
    selected = rng.choice(noisy.size, size=int(round(rate * noisy.size)), replace=False)
    for index in selected:
        alternatives = classes[classes != noisy[index]]
        noisy[index] = rng.choice(alternatives)
    return noisy


# calculate the mean multiclass Brier score for probability calibration analysis
def multiclassBrierScore(yTrue: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    classLookup = {label: index for index, label in enumerate(classes)}
    observed = np.zeros_like(probabilities)
    for rowIndex, label in enumerate(yTrue):
        observed[rowIndex, classLookup[label]] = 1.0
    return float(np.mean(np.sum((probabilities - observed) ** 2, axis=1)))
