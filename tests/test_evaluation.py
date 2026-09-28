# validates multiclass metrics and per-class evidence tables
import numpy as np

from mace.evaluation import classificationMetrics, perClassMetrics


# check perfect predictions give perfect accuracy macro F1 and AUC
def test_perfect_predictions_produce_perfect_core_metrics() -> None:
    labels = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[labels]
    metrics = classificationMetrics(labels, labels, probabilities, np.array([0, 1, 2]))
    assert metrics["Accuracy"] == 1.0
    assert metrics["MacroF1"] == 1.0
    assert metrics["MacroROCAUC"] == 1.0


# check the per-class table has one row for every requested class
def test_per_class_table_contains_every_class() -> None:
    table = perClassMetrics(np.array([0, 1, 2]), np.array([0, 1, 1]), np.array([0, 1, 2]))
    assert table["Class"].tolist() == [0, 1, 2]
    assert table.shape[0] == 3
