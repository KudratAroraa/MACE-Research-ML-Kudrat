# verifies Part 2 fusion fallbacks determinism 
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from mace.proposedSolution import (
    ReliabilityGatedMace,
    mixtureProbabilities,
    predictCaseFoldedMissingNameRule,
    predictMaceSof,
    predictMissingNameRule,
    predictRandomForest,
)
from mace.reproductionPipeline import percentageMetrics, preparePaperDataset
from mace.robustness import corruptMetadata


# prepare one real split for model-level checks
@pytest.fixture(scope="module")
def preparedThingSpeak():
    return preparePaperDataset("thingspeak", saveManifest=False)


# check the unmodified name path reproduces the accepted Part 1 checkpoint
@pytest.mark.slow
def test_original_mace_checkpoint(preparedThingSpeak) -> None:
    result = predictMaceSof(preparedThingSpeak, caseFold=False)
    metrics = percentageMetrics(
        preparedThingSpeak["yTest"],
        result["predictions"],
        result["probabilities"],
        preparedThingSpeak["classes"],
    )
    assert np.isclose(metrics["Accuracy"], 88.5, atol=0.2)
    assert np.isclose(metrics["F1"], 79.4, atol=0.2)


# check a soft mixture preserves class order and probability normalisation
def test_mixture_probabilities_sum_to_one() -> None:
    mace = np.array([[0.8, 0.2], [0.1, 0.9]])
    rf = np.array([[0.3, 0.7], [0.6, 0.4]])
    mixed = mixtureProbabilities(np.array([0.25, 0.75]), mace, rf)
    assert mixed.shape == mace.shape
    assert np.allclose(mixed.sum(axis=1), 1.0)


# check identical experts remain unchanged for every possible gate value
def test_identical_experts_are_unchanged() -> None:
    expert = np.array([[0.2, 0.8], [0.7, 0.3]])
    assert np.allclose(mixtureProbabilities(np.array([0.0, 1.0]), expert, expert), expert)


# check the explicit missing-name baseline falls back exactly to RF for every stream
@pytest.mark.slow
def test_missing_name_rule_equals_rf_when_all_names_empty(preparedThingSpeak) -> None:
    empty = np.full(preparedThingSpeak["yTest"].size, "", dtype=str)
    rule = predictMissingNameRule(preparedThingSpeak, empty)
    rf = predictRandomForest(preparedThingSpeak)
    assert np.array_equal(rule["predictions"], rf["predictions"])
    assert np.array_equal(rule["probabilities"], rf["probabilities"])


# check the case-folded missing-name baseline also falls back exactly to RF when every name is empty
@pytest.mark.slow
def test_case_folded_missing_name_rule_equals_rf_when_all_names_empty(preparedThingSpeak) -> None:
    empty = np.full(preparedThingSpeak["yTest"].size, "", dtype=str)
    rule = predictCaseFoldedMissingNameRule(preparedThingSpeak, empty)
    rf = predictRandomForest(preparedThingSpeak)
    assert np.array_equal(rule["predictions"], rf["predictions"])
    assert np.array_equal(rule["probabilities"], rf["probabilities"])


# check every hard-gate prediction is supplied by one of its two experts
@pytest.mark.slow
def test_hard_gate_uses_one_expert_per_stream(preparedThingSpeak) -> None:
    model = ReliabilityGatedMace(hardGate=True, corruptionShare=0.05, seed=100).fit(preparedThingSpeak)
    result = model.predict(preparedThingSpeak)
    mace = result["maceProbabilities"].argmax(axis=1)
    rf = result["rfProbabilities"].argmax(axis=1)
    assert np.all((result["predictions"] == mace) | (result["predictions"] == rf))


# check fixed seeds reproduce the same gate values and final predictions
@pytest.mark.slow
def test_repeated_fits_are_deterministic(preparedThingSpeak) -> None:
    first = ReliabilityGatedMace(corruptionShare=0.05, seed=100).fit(preparedThingSpeak).predict(preparedThingSpeak)
    second = ReliabilityGatedMace(corruptionShare=0.05, seed=100).fit(preparedThingSpeak).predict(preparedThingSpeak)
    assert np.array_equal(first["gate"], second["gate"])
    assert np.array_equal(first["predictions"], second["predictions"])


# check the shared corruption utility never changes its caller-owned metadata
def test_corruption_does_not_modify_input() -> None:
    names = np.array(["alpha", "beta", "gamma"])
    original = names.copy()
    corruptMetadata(names, 1.0, 100, "delete")
    assert np.array_equal(names, original)


# check the notebook and script call the same public experiment runners
def test_notebook_and_script_share_runners() -> None:
    root = Path(__file__).resolve().parents[1]
    scriptTree = ast.parse((root / "scripts" / "ExecuteProposed.py").read_text(encoding="utf-8"))
    scriptNames = {
        node.id
        for node in ast.walk(scriptTree)
        if isinstance(node, ast.Name) and node.id.startswith("runExp")
    }
    notebook = json.loads((root / "notebook" / "4-ProposedSolution.ipynb").read_text(encoding="utf-8"))
    notebookSource = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    notebookTree = ast.parse(notebookSource)
    notebookNames = {
        node.id
        for node in ast.walk(notebookTree)
        if isinstance(node, ast.Name) and node.id.startswith("runExp")
    }
    expected = {
        "runExp22Motivation",
        "runExp23GateDataset",
        "runExp24GateSelection",
        "runExp25CleanComparison",
        "runExp26NaturalFailure",
        "runExp27Robustness",
        "runExp28RepeatedSplits",
        "runExp29Ablation",
        "runExp30EvidenceAndCost",
    }
    assert expected.issubset(scriptNames)
    assert expected.issubset(notebookNames)
