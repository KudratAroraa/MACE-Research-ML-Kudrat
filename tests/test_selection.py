# validates top-k curves and the released change-point selection heuristic
from mace.cascadeEngine import selectFromTopKAccuracy, selectOrder


# check the change-point heuristic ends with the top-one winner and a final Top-k of 1
def test_select_from_top_k_accuracy_returns_ordered_change_points() -> None:
    curves = [[70.0, 90.0, 100.0], [65.0, 95.0, 100.0]]
    selectedIndices, topKValues = selectFromTopKAccuracy(curves, 3)
    order, topK = selectOrder(selectedIndices, ["A", "B"], topKValues)
    assert order[-1] == "A"
    assert topK[-1] == 1
    assert len(selectedIndices) == 3


# check Eq. 6 when the change-point winner goes A then B then A as k decreases
def test_select_order_never_repeats() -> None:
    curves = [[50.0, 60.0, 90.0, 100.0], [40.0, 70.0, 80.0, 100.0]]
    selectedIndices, topKValues = selectFromTopKAccuracy(curves, 4)
    order, topK = selectOrder(selectedIndices, ["A", "B"], topKValues)
    assert len(order) == len(set(order))
    assert len(order) == len(topK)
    assert topK[-1] == 1
    verbatimOrder, verbatimTopK = selectOrder(selectedIndices, ["A", "B"], topKValues, allowRepeats=True)
    assert verbatimOrder == ["A", "B", "A"]
    assert len(verbatimOrder) == len(verbatimTopK)
