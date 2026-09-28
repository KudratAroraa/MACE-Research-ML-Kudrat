# verifies the released lower-id and current-list tie rules
import numpy as np

from mace.cascadeEngine import cascade, cascadeReference, topKFromFolds


# check equal final scores resolve to the lower class id
def test_equal_final_scores_select_lower_class_id() -> None:
    probabilities = np.array([[0.5, 0.5, 0.0]])
    prediction, _ = cascade([probabilities], 3, "sof", 0.0)
    assert prediction[0] == 0


# check a Top-k tie keeps the previous stage's class order and matches the literal cascade
def test_top_k_stage_preserves_previous_list_order_on_tie() -> None:
    first = np.array([[0.1, 0.8, 0.7]])
    second = np.array([[0.1, 0.5, 0.5]])
    prediction, _ = cascade([first, second], 3, "topK", [2, 1])
    reference, _ = cascadeReference([first, second], 3, "topK", [2, 1])
    assert prediction[0] == 1
    assert np.array_equal(prediction, reference)


# check the top-k accuracy curve ranks tied scores in favour of the lower class id
def test_top_k_curve_sends_score_ties_to_lower_class_id() -> None:
    probabilities = np.array([[0.5, 0.5], [0.2, 0.8]])
    labels = np.array([0, 1])
    curve = topKFromFolds([(probabilities, labels)], 2)
    assert curve[0] == 100.0
