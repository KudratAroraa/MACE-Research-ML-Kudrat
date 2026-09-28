# validates edit distance and metadata probability outputs
import numpy as np

from mace.ddlNLP import DDLNLPClassifier, damerauLevenshtein


# check swapping two adjacent characters counts as a single edit
def test_damerau_levenshtein_counts_adjacent_transposition() -> None:
    assert damerauLevenshtein("temp", "tme p".replace(" ", "")) == 1


# check DDL-NLP gives one probability column per class and each row sums to one
def test_ddl_probabilities_are_normalised() -> None:
    model = DDLNLPClassifier(distancePower=2)
    model.fit(np.array(["temperature", "humidity", "temperature sensor"]), np.array([0, 1, 0]))
    probabilities = model.predict_proba(np.array(["temp", "humid"]))
    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
