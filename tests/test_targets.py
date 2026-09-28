# protects this project's reproduced seed-100 results and their agreement with the paper
import numpy as np
import pytest

from mace.publishedResults import (
    PAPER_DATASET_ORDER,
    PAPER_TABLE1,
    PAPER_TABLE2,
    REPRODUCED_HEURISTICS,
    REPRODUCED_TABLE1,
    REPRODUCED_TABLE2,
)
from mace.reproductionPipeline import ddlNLPDiagnostic


# check selected order Top-k sigma and q for every dataset
@pytest.mark.slow
def test_reproduced_heuristic_targets(paperResults) -> None:
    for datasetName in PAPER_DATASET_ORDER:
        row = paperResults["datasets"][datasetName]["heuristic"].iloc[0]
        expected = REPRODUCED_HEURISTICS[datasetName]
        expectedOrder = " > ".join(
            {"RandomForest": "RF", "GradientBoosting": "GBoost"}.get(name, name)
            for name in expected["order"]
        )
        assert row["ReproducedOrder"] == expectedOrder
        assert row["ReproducedTopK"] == str(expected["topK"])
        assert row["BestSigma"] == expected["sigma"]
        assert row["BestPercentage"] == expected["q"]


# check every Table I accuracy and Table II macro F1 against this project's reproduced values
@pytest.mark.slow
def test_reproduced_table_targets(paperResults) -> None:
    # a tolerance of 0.2 points allows one ThingSpeak test stream (1/637 = 0.16), the largest difference observed between operating systems
    tolerance = 0.2
    for datasetName in PAPER_DATASET_ORDER:
        metrics = paperResults["datasets"][datasetName]["metrics"].set_index("Method")
        for column, targets in [("Accuracy", REPRODUCED_TABLE1), ("F1", REPRODUCED_TABLE2)]:
            for method, datasetValues in targets.items():
                assert np.isclose(metrics.loc[method, column], datasetValues[datasetName], atol=tolerance), (
                    datasetName,
                    method,
                    column,
                )


# check the MACE rows stay close to the paper's Tables I and II on every dataset
@pytest.mark.slow
def test_mace_rows_close_to_paper(paperResults) -> None:
    for datasetName in PAPER_DATASET_ORDER:
        metrics = paperResults["datasets"][datasetName]["metrics"].set_index("Method")
        for method in ["MACE Top-k", "MACE PF", "MACE SoF"]:
            assert abs(metrics.loc[method, "Accuracy"] - PAPER_TABLE1[method][datasetName]) <= 1.5, (datasetName, method)
            assert abs(metrics.loc[method, "F1"] - PAPER_TABLE2[method][datasetName]) <= 1.2, (datasetName, method)


# check the exact published DDL-NLP standalone checkpoint
@pytest.mark.slow
def test_ddl_accuracy_target(paperResults) -> None:
    prepared = paperResults["datasets"]["thingspeak"]["prepared"]
    diagnostic = ddlNLPDiagnostic(prepared).iloc[0]
    assert np.isclose(diagnostic["Accuracy"], 73.2, atol=0.05)
