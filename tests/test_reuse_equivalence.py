# confirms reused preparations give identical results and never modify the shared baseline
import numpy as np
import pandas as pd
import pytest

from mace.criticalAnalysis import _sensitivityPrepared
from mace.reproductionPipeline import preparePaperDataset, runPaperDataset

# both tests together take about 35 seconds because each fits the Swissex pool several times
pytestmark = pytest.mark.slow


# snapshot every baseline array a variant could touch so it can be compared afterwards
def baselineSnapshot(prepared: dict) -> dict[str, np.ndarray]:
    arrays = {"series": prepared["bundle"].series, "metadata": prepared["bundle"].metadata}
    for side in ("trainInputs", "testInputs"):
        for name, values in prepared[side].items():
            arrays[f"{side}.{name}"] = values
    for name, values in prepared["testCache"].items():
        arrays[f"testCache.{name}"] = values
    for name, folds in prepared["foldCache"].items():
        for index, (probabilities, labels) in enumerate(folds):
            arrays[f"foldCache.{name}.{index}.probabilities"] = probabilities
            arrays[f"foldCache.{name}.{index}.labels"] = labels
    return {key: np.array(values, copy=True) for key, values in arrays.items()}


# check a supplied preparation gives the same Swissex metrics as a fresh run
def test_prepared_reuse_matches_fresh_run() -> None:
    fresh = runPaperDataset("swissex", includeBruteForce=False, saveManifest=False)
    prepared = preparePaperDataset("swissex", saveManifest=False)
    reused = runPaperDataset("swissex", includeBruteForce=False, prepared=prepared)
    pd.testing.assert_frame_equal(fresh["metrics"], reused["metrics"])


# check the median-moment variant matches a fresh run and leaves every baseline array unchanged
def test_median_variant_matches_fresh_run_and_leaves_baseline_unchanged() -> None:
    options = {"constantMomentPolicy": "medianTraining"}
    baseline = preparePaperDataset("swissex", saveManifest=False)
    before = baselineSnapshot(baseline)
    variant = _sensitivityPrepared(baseline, options)
    after = baselineSnapshot(baseline)
    assert before.keys() == after.keys()
    for key in before:
        assert np.array_equal(before[key], after[key]), key
    fresh = runPaperDataset("swissex", includeBruteForce=False, saveManifest=False, **options)
    reused = runPaperDataset("swissex", includeBruteForce=False, prepared=variant, **options)
    pd.testing.assert_frame_equal(fresh["metrics"], reused["metrics"])
