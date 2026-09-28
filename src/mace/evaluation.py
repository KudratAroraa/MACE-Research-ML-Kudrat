# calculates consistent multiclass metrics and uncertainty summaries
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)



# align model probability columns to a shared class order before combination
def alignProbabilities(
    probabilities: np.ndarray,
    sourceClasses: np.ndarray,
    targetClasses: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    aligned = np.zeros((probabilities.shape[0], len(targetClasses)), dtype=float)
    targetLookup = {label: index for index, label in enumerate(targetClasses)}
    for sourceIndex, label in enumerate(sourceClasses):
        if label in targetLookup:
            aligned[:, targetLookup[label]] = probabilities[:, sourceIndex]
    return aligned


# calculate the paper metrics plus imbalance-aware supporting measures
def classificationMetrics(
    yTrue: np.ndarray,
    yPred: np.ndarray,
    yProbability: np.ndarray | None = None,
    classes: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {
        "Accuracy": accuracy_score(yTrue, yPred),
        "BalancedAccuracy": balanced_accuracy_score(yTrue, yPred),
        "MacroPrecision": precision_score(yTrue, yPred, average="macro", zero_division=0),
        "MacroRecall": recall_score(yTrue, yPred, average="macro", zero_division=0),
        "MacroF1": f1_score(yTrue, yPred, average="macro", zero_division=0),
        "WeightedF1": f1_score(yTrue, yPred, average="weighted", zero_division=0),
    }
    if yProbability is not None:
        labels = np.asarray(classes if classes is not None else np.unique(yTrue))
        try:
            metrics["MacroROCAUC"] = roc_auc_score(
                yTrue,
                yProbability,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            metrics["MacroROCAUC"] = np.nan
        metrics["LogLoss"] = log_loss(yTrue, yProbability, labels=labels)
    return {name: float(value) for name, value in metrics.items()}


# return precision, recall, F1 and support for every class
def perClassMetrics(yTrue: np.ndarray, yPred: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        yTrue,
        yPred,
        labels=classes,
        zero_division=0,
    )
    return pd.DataFrame(
        {"Class": classes, "Precision": precision, "Recall": recall, "F1": f1, "Support": support}
    )


# return a labelled confusion matrix for reports and saved outputs
def confusionTable(yTrue: np.ndarray, yPred: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    matrix = confusion_matrix(yTrue, yPred, labels=classes)
    return pd.DataFrame(
        matrix,
        index=[f"Actual_{label}" for label in classes],
        columns=[f"Predicted_{label}" for label in classes],
    )


# estimate a nonparametric confidence interval for one prediction vector
def bootstrapMetricInterval(
    yTrue: np.ndarray,
    yPred: np.ndarray,
    metric: str = "f1_macro",
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 100,
) -> dict[str, float]:
    generators = {
        "accuracy": lambda a, b: accuracy_score(a, b),
        "f1_macro": lambda a, b: f1_score(a, b, average="macro", zero_division=0),
        "balanced_accuracy": lambda a, b: balanced_accuracy_score(a, b),
    }
    if metric not in generators:
        raise KeyError(f"Unsupported bootstrap metric: {metric}")
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(iterations):
        sample = rng.integers(0, len(yTrue), len(yTrue))
        scores.append(generators[metric](yTrue[sample], yPred[sample]))
    alpha = (1.0 - confidence) / 2.0
    return {
        "Mean": float(np.mean(scores)),
        "Lower": float(np.quantile(scores, alpha)),
        "Upper": float(np.quantile(scores, 1.0 - alpha)),
    }
