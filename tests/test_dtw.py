# validates the compiled 1NN-DTW distance and its single-pass class scores
import numpy as np

from mace.timeSeries import DTWNearestClassifier, _dtwPair, dynamicTimeWarpingReference


# check the compiled DTW matches the pure Python reference with and without a 10% band
def test_compiled_dtw_matches_reference() -> None:
    rng = np.random.default_rng(100)
    for _ in range(50):
        left = rng.normal(size=int(rng.integers(20, 61)))
        right = rng.normal(size=int(rng.integers(20, 61)))
        length = max(left.size, right.size)
        unconstrained = _dtwPair(left, right, length)
        assert abs(unconstrained - dynamicTimeWarpingReference(left, right, windowFraction=1.0)) <= 1.0e-9
        banded = _dtwPair(left, right, int(np.ceil(length * 0.10)))
        assert abs(banded - dynamicTimeWarpingReference(left, right, windowFraction=0.10)) <= 1.0e-9


# check 1NN-DTW predictions equal the argmax of its class probabilities
def test_dtw_predict_matches_probability_argmax() -> None:
    rng = np.random.default_rng(100)
    X = rng.normal(size=(30, 24))
    y = np.repeat([0, 1, 2], 10)
    X += y[:, None] * 0.5
    queries = rng.normal(size=(12, 24)) + np.repeat([0, 1, 2], 4)[:, None] * 0.5
    model = DTWNearestClassifier().fit(X, y)
    probabilities = model.predict_proba(queries)
    assert np.array_equal(model.predict(queries), model.classes_[np.argmax(probabilities, axis=1)])
