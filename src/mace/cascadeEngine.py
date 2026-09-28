# reproduces the authors' cascade and selection logic with verified vectorisation
from __future__ import annotations

from heapq import nlargest
from itertools import combinations, permutations

import numpy as np
from sklearn.metrics import accuracy_score


# run a literal sample-level port of the released cascade
def cascadeReference(
    probabilityList: list[np.ndarray],
    classCount: int,
    criterion: str,
    parameter: float | list[int],
    percentageRule: str = "floor",
    earlyExit: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    sampleCount = probabilityList[0].shape[0]
    predictions = np.empty(sampleCount, dtype=int)
    scores = np.zeros((sampleCount, classCount), dtype=float)
    finalStage = len(probabilityList) - 1
    for sampleIndex in range(sampleCount):
        active = list(range(classCount))
        survivors = active.copy()
        exitedEarly = False
        for stageIndex, probabilities in enumerate(probabilityList):
            sampleScores = probabilities[sampleIndex]
            if stageIndex == finalStage:
                survivors = active.copy()
                active = nlargest(1, active, key=lambda classIndex: sampleScores[classIndex])
            elif criterion == "sof":
                activeScores = [sampleScores[classIndex] for classIndex in active]
                threshold = np.mean(activeScores) + float(parameter) * np.std(activeScores)
                retained = [classIndex for classIndex in active if sampleScores[classIndex] >= threshold]
                active = retained or nlargest(1, active, key=lambda classIndex: sampleScores[classIndex])
            elif criterion == "topK":
                stageK = int(parameter[min(stageIndex, len(parameter) - 1)])
                active = nlargest(stageK, active, key=lambda classIndex: sampleScores[classIndex])
            elif criterion == "pf":
                rawCount = len(active) * float(parameter)
                retainedCount = int(np.ceil(rawCount)) if percentageRule == "ceiling" else int(rawCount)
                retainedCount = max(1, retainedCount)
                active = nlargest(retainedCount, active, key=lambda classIndex: sampleScores[classIndex])
            else:
                raise ValueError("criterion must be 'topK', 'pf' or 'sof'")
            if earlyExit and len(active) == 1 and stageIndex < finalStage:
                survivors = active.copy()
                exitedEarly = True
                break
        predictions[sampleIndex] = active[0]
        finalScores = np.zeros(classCount, dtype=float)
        if exitedEarly:
            finalScores[active[0]] = 1.0
        else:
            finalScores[survivors] = probabilityList[-1][sampleIndex, survivors] + 1.0e-9
        scores[sampleIndex] = finalScores / finalScores.sum()
    return predictions, scores


# rank classes by score while preserving the released stable tie order
def _rank(probabilities: np.ndarray, positions: np.ndarray, active: np.ndarray) -> np.ndarray:
    rankingKey = np.where(active, probabilities, -np.inf)
    orderedPositions = np.argsort(positions, axis=1, kind="stable")
    orderedKeys = np.take_along_axis(rankingKey, orderedPositions, axis=1)
    order = np.take_along_axis(
        orderedPositions,
        np.argsort(-orderedKeys, axis=1, kind="stable"),
        axis=1,
    )
    ranks = np.empty_like(order)
    rankValues = np.arange(probabilities.shape[1])[None, :].repeat(probabilities.shape[0], axis=0)
    np.put_along_axis(ranks, order, rankValues, axis=1)
    return ranks


# run the vectorised cascade with output identical to cascadeReference
def cascade(
    probabilityList: list[np.ndarray],
    classCount: int,
    criterion: str,
    parameter: float | list[int],
    percentageRule: str = "floor",
    earlyExit: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if earlyExit:
        return cascadeReference(
            probabilityList,
            classCount,
            criterion,
            parameter,
            percentageRule,
            earlyExit,
        )
    sampleCount = probabilityList[0].shape[0]
    active = np.ones((sampleCount, classCount), dtype=bool)
    positions = np.tile(np.arange(classCount), (sampleCount, 1))
    finalStage = len(probabilityList) - 1
    rows = np.arange(sampleCount)
    for stageIndex, probabilities in enumerate(probabilityList):
        if stageIndex == finalStage:
            survivors = active.copy()
            predictions = np.argmin(_rank(probabilities, positions, active), axis=1)
            break
        if criterion == "sof":
            activeValues = [probabilities[row, active[row]] for row in rows]
            means = np.array([values.mean() for values in activeValues])
            deviations = np.array([values.std() for values in activeValues])
            retained = active & (probabilities >= (means + float(parameter) * deviations)[:, None])
            emptyRows = ~retained.any(axis=1)
            if emptyRows.any():
                best = np.argmin(_rank(probabilities, positions, active), axis=1)
                retained[emptyRows, best[emptyRows]] = True
            active = retained
        else:
            activeCounts = active.sum(axis=1)
            if criterion == "topK":
                stageK = int(parameter[min(stageIndex, len(parameter) - 1)])
                retainedCounts = np.full(sampleCount, stageK)
            elif criterion == "pf":
                rawCounts = activeCounts * float(parameter)
                retainedCounts = (
                    np.ceil(rawCounts).astype(int)
                    if percentageRule == "ceiling"
                    else np.floor(rawCounts).astype(int)
                )
                retainedCounts = np.maximum(1, retainedCounts)
            else:
                raise ValueError("criterion must be 'topK', 'pf' or 'sof'")
            ranks = _rank(probabilities, positions, active)
            active = active & (ranks < retainedCounts[:, None])
            positions = ranks
    finalScores = np.where(survivors, probabilityList[-1] + 1.0e-9, 0.0)
    return predictions, finalScores / finalScores.sum(axis=1, keepdims=True)


# pool five validation folds into the released top-k accuracy curve
def topKFromFolds(
    foldScores: list[tuple[np.ndarray, np.ndarray]],
    classCount: int,
) -> list[float]:
    hits = {k: [] for k in range(1, classCount + 1)}
    for probabilities, labels in foldScores:
        ranked = np.argsort(-probabilities, axis=1, kind="stable")
        for k in range(1, classCount + 1):
            hits[k].extend((ranked[:, :k] == labels[:, None]).any(axis=1))
    return [float(np.mean(hits[k]) * 100.0) for k in range(1, classCount + 1)]


# reproduce the released change-point selection logic
def selectFromTopKAccuracy(
    accuracyCurves: list[list[float]],
    classCount: int,
) -> tuple[list[int], list[int]]:
    selectedIndices: list[int] = []
    working = [list(curve) for curve in accuracyCurves]
    for position in range(classCount):
        values = [curve[position] for curve in working]
        if position > 0 and max(values) == values[selectedIndices[position - 1]]:
            selectedIndices.append(selectedIndices[position - 1])
            if position < classCount - 1:
                nextValues = [curve[position + 1] for curve in working]
                if max(nextValues) != nextValues[selectedIndices[position]]:
                    working[selectedIndices[position]] = [0.0] * classCount
        else:
            selectedIndices.append(int(np.argmax(values)))
    topKValues = []
    for position in range(classCount - 1, 0, -1):
        if selectedIndices[position] != selectedIndices[position - 1]:
            topKValues.append(position + 1)
    if classCount == 1 or selectedIndices[0] != selectedIndices[min(1, classCount - 1)]:
        topKValues.append(1)
    elif not topKValues or topKValues[-1] != 1:
        topKValues.append(1)
    return selectedIndices, topKValues


# convert change-point winners into a cascade order where each classifier appears at most once (Eq. 6)
# the first stage found by the backward search is kept and the Top-k value of any removed stage is dropped
# allowRepeats reproduces the released select_from_k_accuracy code verbatim
def selectOrder(
    selectedIndices: list[int],
    pool: list[str],
    topKValues: list[int] | None = None,
    allowRepeats: bool = False,
) -> tuple[list[str], list[int] | None]:
    stages = []
    for position in range(len(selectedIndices) - 1, -1, -1):
        if position == len(selectedIndices) - 1 or selectedIndices[position] != selectedIndices[position + 1]:
            stages.append(pool[selectedIndices[position]])
    stageTopK = list(topKValues) if topKValues is not None else [None] * len(stages)
    order, alignedTopK = [], []
    for name, k in zip(stages, stageTopK):
        if allowRepeats or name not in order:
            order.append(name)
            alignedTopK.append(k)
    if topKValues is None:
        return order, None

    # the final stage always keeps one class
    alignedTopK[-1] = 1
    return order, alignedTopK


# tune PF or SoF with the same five training folds and accuracy objective
def tuneParameter(
    order: list[str],
    foldCache: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    classCount: int,
    criterion: str,
    grid: list[float],
    percentageRule: str = "floor",
    earlyExit: bool = False,
) -> tuple[float, list[float]]:
    foldCount = len(foldCache[order[0]])
    foldData = [
        ([foldCache[name][fold][0] for name in order], foldCache[order[0]][fold][1])
        for fold in range(foldCount)
    ]
    bestParameter, bestAccuracy = grid[0], -1.0
    curve = []
    for parameter in grid:
        meanAccuracy = float(
            np.mean(
                [
                    accuracy_score(
                        labels,
                        cascade(
                            probabilities,
                            classCount,
                            criterion,
                            parameter,
                            percentageRule,
                            earlyExit,
                        )[0],
                    )
                    for probabilities, labels in foldData
                ]
            )
        )
        curve.append(meanAccuracy * 100.0)
        if meanAccuracy > bestAccuracy:
            bestParameter, bestAccuracy = parameter, meanAccuracy
    return float(bestParameter), curve


# average aligned class scores across one stated classifier pool
def softVoting(probabilityList: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    average = np.mean(probabilityList, axis=0)
    return average.argmax(axis=1), average


# reproduce the authors' test-optimised brute-force oracle
def bruteForceSoF(
    probabilityCache: dict[str, np.ndarray],
    pool: list[str],
    classCount: int,
    yTestPositions: np.ndarray,
    sigmaValues: list[float],
) -> dict[str, object]:
    best: dict[str, object] = {"accuracy": -1.0}
    searched = 0
    records = []
    for size in range(1, len(pool) + 1):
        for subset in combinations(pool, size):
            for order in permutations(subset):
                for sigma in sigmaValues:
                    predictions, scores = cascade(
                        [probabilityCache[name] for name in order],
                        classCount,
                        "sof",
                        sigma,
                    )
                    accuracy = accuracy_score(yTestPositions, predictions) * 100.0
                    searched += 1
                    records.append(
                        {
                            "Order": " > ".join(order),
                            "Stages": len(order),
                            "Sigma": sigma,
                            "Accuracy": accuracy,
                        }
                    )
                    if accuracy > float(best["accuracy"]):
                        best = {
                            "accuracy": accuracy,
                            "order": list(order),
                            "sigma": sigma,
                            "predictions": predictions,
                            "scores": scores,
                        }
    best["configurationsSearched"] = searched
    best["records"] = records
    best["selectionProtocol"] = "PaperOracle_TestOptimised"
    best["validForRecommendation"] = False
    return best
