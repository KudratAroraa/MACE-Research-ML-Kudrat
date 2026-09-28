# verifies the leakage controls features targets and selection logic of the Part 2 gate
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from mace.ddlNLP import DDLNLPClassifier
from mace.proposedConfig import PROPOSED_CONFIG
from mace.reliabilityGate import (
    RELIABILITY_FEATURES,
    ReliabilityDDL,
    buildGateDataset,
    buildMaceExpert,
    foldIndices,
    outOfFoldMatrix,
    routingTargets,
    selectGate,
)
from mace.reproductionPipeline import classPositions, preparePaperDataset


# prepare the real ThingSpeak split once for all integration checks in this module
@pytest.fixture(scope="module")
def preparedThingSpeak():
    return preparePaperDataset("thingspeak", saveManifest=False)


# check the subclass is numerically identical when its added reliability behaviour is disabled
@pytest.mark.slow
def test_reliability_ddl_matches_parent(preparedThingSpeak) -> None:
    trainNames = preparedThingSpeak["trainInputs"]["metadata"]
    labels = preparedThingSpeak["yTrain"]
    testNames = preparedThingSpeak["testInputs"]["metadata"][:12]
    parent = DDLNLPClassifier().fit(trainNames, labels)
    proposed = ReliabilityDDL().fit(trainNames, labels)
    assert np.array_equal(parent.predict_proba(testNames), proposed.predict_proba(testNames))


# check the cached held-out probabilities restore every row and its correct label once
@pytest.mark.slow
def test_out_of_fold_matrix_has_exact_coverage(preparedThingSpeak) -> None:
    restored = outOfFoldMatrix(preparedThingSpeak, "RandomForest")
    assert restored.shape[0] == preparedThingSpeak["yTrain"].size
    for foldNumber, (_, heldOut) in enumerate(foldIndices(preparedThingSpeak)):
        storedLabels = preparedThingSpeak["foldCache"]["RandomForest"][foldNumber][1]
        expected = classPositions(preparedThingSpeak["yTrain"][heldOut], preparedThingSpeak["classes"])
        assert np.array_equal(storedLabels, expected)


# check copies remain grouped with originals and never include an untouched test observation
@pytest.mark.slow
def test_gate_dataset_grouping_and_partition(preparedThingSpeak) -> None:
    expert = buildMaceExpert(preparedThingSpeak, caseFold=True)
    original = preparedThingSpeak["trainInputs"]["metadata"].copy()
    data = buildGateDataset(
        preparedThingSpeak,
        expert,
        True,
        0.1,
        PROPOSED_CONFIG["corruptionModes"],
        100,
    )
    assert np.array_equal(original, preparedThingSpeak["trainInputs"]["metadata"])
    assert set(data["groups"]).issubset(set(range(preparedThingSpeak["yTrain"].size)))
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=100)
    splitTargets = np.where(data["fitMask"], data["targets"], 0)
    for fitIndex, heldOutIndex in splitter.split(np.zeros(len(splitTargets)), splitTargets, data["groups"]):
        assert set(data["groups"][fitIndex]).isdisjoint(set(data["groups"][heldOutIndex]))


# check every declared feature is finite and remains in its pre-registered order
@pytest.mark.slow
def test_reliability_features_are_finite_and_ordered(preparedThingSpeak) -> None:
    expert = buildMaceExpert(preparedThingSpeak, caseFold=True)
    data = buildGateDataset(preparedThingSpeak, expert, True, 0.05, ["missing"], 100)
    assert list(data["features"].columns) == RELIABILITY_FEATURES
    assert np.isfinite(data["features"].to_numpy(float)).all()


# check targets support-gap weights and tie exclusion on a hand-made probability example
def test_gate_target_and_weight_definition() -> None:
    mace = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
    rf = np.array([[0.6, 0.4], [0.2, 0.8], [0.5, 0.5]])
    routing = routingTargets(mace, rf, np.array([0, 1, 0]))
    assert np.array_equal(routing["targets"], np.array([1, 0, -1]))
    assert np.array_equal(routing["fitMask"], np.array([True, True, False]))
    assert np.allclose(routing["probabilityGap"], np.array([0.2, 0.1, 0.0]))
    assert routing["sampleWeights"][2] == 0.0


# create a deterministic synthetic gate dataset for the constraint and smaller-C tie rule
def _syntheticGateDataset() -> dict:
    rows = 40
    classes = np.array([0, 1])
    targets = np.tile([0, 1], rows // 2)
    truePositions = targets.copy()
    mace = np.tile([0.6, 0.4], (rows, 1))
    rf = np.tile([0.4, 0.6], (rows, 1))
    features = pd.DataFrame(np.zeros((rows, len(RELIABILITY_FEATURES))), columns=RELIABILITY_FEATURES)
    return {
        "features": features,
        "targets": targets,
        "fitMask": np.ones(rows, dtype=bool),
        "probabilityGap": np.ones(rows),
        "sampleWeights": np.ones(rows),
        "groups": np.arange(rows),
        "isCorrupted": np.arange(rows) >= rows // 2,
        "truePositions": truePositions,
        "maceProbabilities": mace,
        "rfProbabilities": rf,
        "classes": classes,
    }


# check selection records the constraint and resolves an objective tie with the smaller C
def test_select_gate_constraint_and_tie_rule() -> None:
    selected = selectGate(_syntheticGateDataset(), [0.1, 1.0], 2, 100.0, 100)
    assert selected["constraintMet"]
    assert selected["chosenC"] == 0.1
