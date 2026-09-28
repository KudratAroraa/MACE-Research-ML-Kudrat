# provides paired statistical evidence for the Part 2 model comparisons
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, t, wilcoxon
from sklearn.metrics import accuracy_score, f1_score, log_loss


# calculate the exact paired McNemar comparison from discordant predictions
def mcnemarExact(yTrue: np.ndarray, predA: np.ndarray, predB: np.ndarray) -> dict[str, float]:
    truth = np.asarray(yTrue)
    correctA = np.asarray(predA) == truth
    correctB = np.asarray(predB) == truth
    b = int(np.sum(correctA & ~correctB))
    c = int(np.sum(~correctA & correctB))
    pValue = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    return {"AOnlyCorrect": b, "BOnlyCorrect": c, "Discordant": b + c, "PValue": pValue}


# evaluate one requested metric from predictions and optional class probabilities
def _metricValue(
    yTrue: np.ndarray,
    result: dict,
    metric: str,
    classes: np.ndarray,
) -> float:
    if metric == "Accuracy":
        return float(accuracy_score(yTrue, result["predictions"]) * 100.0)
    if metric == "F1":
        return float(
            f1_score(yTrue, result["predictions"], labels=classes, average="macro", zero_division=0) * 100.0
        )
    if metric == "LogLoss":
        return float(log_loss(yTrue, result["probabilities"], labels=classes))
    raise KeyError(f"Unsupported paired bootstrap metric: {metric}")


# resample within each class so rare-class representation is preserved in every paired replicate
def pairedBootstrapDifference(
    yTrue: np.ndarray,
    resultA: dict,
    resultB: dict,
    metric: str,
    iterations: int,
    seed: int,
    stratify: bool = True,
) -> dict[str, float]:
    truth = np.asarray(yTrue)
    classes = np.asarray(resultA.get("classes", np.unique(truth)))
    observed = _metricValue(truth, resultB, metric, classes) - _metricValue(truth, resultA, metric, classes)
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        if stratify:
            parts = []
            for label in classes:
                classIndex = np.flatnonzero(truth == label)
                parts.append(rng.choice(classIndex, classIndex.size, replace=True))
            sample = np.concatenate(parts)
            rng.shuffle(sample)
        else:
            sample = rng.choice(truth.size, truth.size, replace=True)
        sampledA = {
            "predictions": np.asarray(resultA["predictions"])[sample],
            "probabilities": np.asarray(resultA["probabilities"])[sample],
            "classes": classes,
        }
        sampledB = {
            "predictions": np.asarray(resultB["predictions"])[sample],
            "probabilities": np.asarray(resultB["probabilities"])[sample],
            "classes": classes,
        }
        differences[iteration] = _metricValue(truth[sample], sampledB, metric, classes) - _metricValue(
            truth[sample], sampledA, metric, classes
        )
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "ObservedDifference": float(observed),
        "Lower95": float(lower),
        "Upper95": float(upper),
        "ProbabilityDifferenceAtMostZero": float(np.mean(differences <= 0.0)),
    }


# compare paired seed-level values without treating rows from different splits as independent
def pairedWilcoxon(valuesA: np.ndarray, valuesB: np.ndarray) -> dict[str, float]:
    first = np.asarray(valuesA, dtype=float)
    second = np.asarray(valuesB, dtype=float)
    differences = second - first
    if np.allclose(differences, 0.0):
        return {"Statistic": 0.0, "PValue": 1.0, "MedianDifference": 0.0}
    result = wilcoxon(first, second, alternative="two-sided")
    return {
        "Statistic": float(result.statistic),
        "PValue": float(result.pvalue),
        "MedianDifference": float(np.median(differences)),
    }


# return a t interval for the mean paired difference across independent split seeds
def seedDifferenceInterval(
    differences: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    if values.size == 0:
        raise ValueError("At least one seed difference is required")
    if values.size == 1 or np.allclose(values, values[0]):
        return float(values.mean()), float(values.mean())
    mean = float(values.mean())
    standardError = float(values.std(ddof=1) / np.sqrt(values.size))
    critical = float(t.ppf((1.0 + confidence) / 2.0, values.size - 1))
    return mean - critical * standardError, mean + critical * standardError


# convert a robustness curve into average performance over a fixed rate interval
def areaUnderRobustnessCurve(
    rates: np.ndarray,
    values: np.ndarray,
    rateRange: tuple[float, float] | list[float],
) -> float:
    x = np.asarray(rates, dtype=float)
    y = np.asarray(values, dtype=float)
    lower, upper = map(float, rateRange)
    if upper <= lower:
        raise ValueError("The robustness interval must have positive width")
    order = np.argsort(x)
    x, y = x[order], y[order]
    inside = (x >= lower) & (x <= upper)
    selectedX = x[inside]
    selectedY = y[inside]
    if selectedX.size == 0:
        raise ValueError("No robustness observations fall inside the requested range")
    if selectedX[0] > lower:
        selectedX = np.insert(selectedX, 0, lower)
        selectedY = np.insert(selectedY, 0, np.interp(lower, x, y))
    if selectedX[-1] < upper:
        selectedX = np.append(selectedX, upper)
        selectedY = np.append(selectedY, np.interp(upper, x, y))
    return float(np.trapezoid(selectedY, selectedX) / (upper - lower))
