# stores the published checkpoints and exact reproduction constants
from __future__ import annotations


PAPER_DATASET_ORDER = ["urban", "swissex", "thingspeak"]

# dataset dimensions and metadata availability stated in Section IV-A of the paper
PAPER_DATASET_FACTS = {
    "urban": {"streams": 1065, "length": 864, "classes": 16, "metadata": False},
    "swissex": {"streams": 346, "length": 445, "classes": 11, "metadata": False},
    "thingspeak": {"streams": 2121, "length": 96, "classes": 21, "metadata": True},
}

PAPER_SHORT_NAMES = {
    "urban": "UrbObs",
    "swissex": "Swissex",
    "thingspeak": "TS",
}

PAPER_POOLS = {
    "urban": ["1NN-ED", "kNN", "SVM", "GradientBoosting", "RandomForest"],
    "swissex": ["1NN-ED", "kNN", "SVM", "GradientBoosting", "RandomForest"],
    "thingspeak": ["1NN-ED", "kNN", "DDL-NLP", "GradientBoosting", "RandomForest"],
}

PAPER_MODEL_NAMES = {
    "Ridge": "Ridge",
    "SVM": "SVM",
    "kNN": "kNN",
    "LogisticRegression": "LR",
    "DecisionTree": "C4.5",
    "GaussianNB": "NB",
    "RandomForest": "RF",
    "GradientBoosting": "GBoost",
    "1NN-ED": "1NN-ED",
    "DDL-NLP": "DDL-NLP",
}

PAPER_FIGURE3B = {
    "Ridge": {"urban": 45.6, "swissex": 36.5, "thingspeak": 31.4},
    "SVM": {"urban": 60.6, "swissex": 43.3, "thingspeak": 62.5},
    "kNN": {"urban": 83.8, "swissex": 71.2, "thingspeak": 71.0},
    "LR": {"urban": 82.2, "swissex": 65.4, "thingspeak": 61.1},
    "C4.5": {"urban": 85.0, "swissex": 66.3, "thingspeak": 72.1},
    "NB": {"urban": 79.1, "swissex": 72.1, "thingspeak": 29.2},
    "RF": {"urban": 88.1, "swissex": 78.8, "thingspeak": 75.7},
    "GBoost": {"urban": 85.6, "swissex": 75.0, "thingspeak": 73.5},
}

PAPER_FIGURE3A = {
    ("1NN-ED", "raw"): {"urban": 76.9, "swissex": 55.8, "thingspeak": 70.8},
    ("1NN-ED", "znorm"): {"urban": 67.2, "swissex": 47.1, "thingspeak": 66.6},
    ("1NN-DTW", "raw"): {"urban": 90.0, "swissex": 74.0, "thingspeak": 73.3},
    ("1NN-DTW", "znorm"): {"urban": 81.6, "swissex": 44.2, "thingspeak": 55.9},
    ("BOPF", "raw"): {"urban": 74.7, "swissex": 37.4, "thingspeak": 41.4},
    ("BOPF", "znorm"): {"urban": 75.3, "swissex": 44.2, "thingspeak": 41.6},
    ("LTS", "raw"): {"urban": 6.9, "swissex": 13.5, "thingspeak": 1.4},
    ("LTS", "znorm"): {"urban": 26.2, "swissex": 20.2, "thingspeak": 18.4},
    ("SDE", "raw"): {"urban": 47.2, "swissex": 31.7, "thingspeak": 29.7},
    ("SDE", "znorm"): {"urban": 36.2, "swissex": 29.8, "thingspeak": 24.0},
}

PAPER_DDL_ACCURACY = 73.2

PAPER_TABLE1 = {
    "Best standalone": {"urban": 88.1, "swissex": 78.8, "thingspeak": 75.7},
    "Soft voting": {"urban": 89.1, "swissex": 83.7, "thingspeak": 79.6},
    "MACE Top-k": {"urban": 90.6, "swissex": 84.6, "thingspeak": 84.3},
    "MACE PF": {"urban": 90.6, "swissex": 83.2, "thingspeak": 83.4},
    "MACE SoF": {"urban": 90.6, "swissex": 82.7, "thingspeak": 88.2},
    "MACE brute-force (SoF)": {"urban": 92.2, "swissex": 86.5, "thingspeak": 88.5},
}

PAPER_TABLE2 = {
    "Best standalone": {"urban": 87.9, "swissex": 83.2, "thingspeak": 71.9},
    "Soft voting": {"urban": 86.4, "swissex": 81.6, "thingspeak": 57.8},
    "MACE Top-k": {"urban": 87.9, "swissex": 83.2, "thingspeak": 67.7},
    "MACE PF": {"urban": 87.9, "swissex": 83.2, "thingspeak": 63.7},
    "MACE SoF": {"urban": 87.9, "swissex": 80.0, "thingspeak": 79.4},
    "MACE brute-force (SoF)": {"urban": 88.7, "swissex": 84.0, "thingspeak": 76.7},
}

PAPER_HEURISTICS = {
    "urban": {"order": ["1NN-ED", "RandomForest"], "topK": "7"},
    "swissex": {"order": ["1NN-ED", "RandomForest"], "topK": "change points at k=10 and k=7"},
    "thingspeak": {"order": ["DDL-NLP", "RandomForest"], "topK": "not stated"},
}

# seed-100 results reproduced by this project (Python 3.11, macOS)
REPRODUCED_TABLE1 = {  # accuracy, UrbObs / Swissex / ThingSpeak
    "Best standalone": {"urban": 91.2, "swissex": 84.6, "thingspeak": 76.0},
    "Soft voting": {"urban": 90.6, "swissex": 81.7, "thingspeak": 80.1},
    "MACE Top-k": {"urban": 91.2, "swissex": 84.6, "thingspeak": 84.3},
    "MACE PF": {"urban": 91.2, "swissex": 84.6, "thingspeak": 83.5},
    "MACE SoF": {"urban": 91.2, "swissex": 82.7, "thingspeak": 88.5},
    "MACE brute-force (SoF)": {"urban": 92.5, "swissex": 86.5, "thingspeak": 89.0},
}

REPRODUCED_TABLE2 = {  # macro F1
    "Best standalone": {"urban": 88.2, "swissex": 83.2, "thingspeak": 71.9},
    "Soft voting": {"urban": 88.5, "swissex": 77.8, "thingspeak": 58.8},
    "MACE Top-k": {"urban": 88.2, "swissex": 83.2, "thingspeak": 67.6},
    "MACE PF": {"urban": 88.2, "swissex": 83.2, "thingspeak": 64.8},
    "MACE SoF": {"urban": 88.2, "swissex": 80.0, "thingspeak": 79.4},
    "MACE brute-force (SoF)": {"urban": 89.8, "swissex": 83.9, "thingspeak": 79.0},
}

REPRODUCED_HEURISTICS = {
    "urban": {"order": ["1NN-ED", "RandomForest"], "topK": [8, 1], "sigma": -0.6, "q": 0.5},
    "swissex": {"order": ["1NN-ED", "RandomForest"], "topK": [8, 1], "sigma": 0.4, "q": 0.8},
    "thingspeak": {"order": ["DDL-NLP", "RandomForest"], "topK": [3, 1], "sigma": 0.2, "q": 0.2},
}
