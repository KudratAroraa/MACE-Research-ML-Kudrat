# exposes the main reusable components of the independent MACE implementation
from .data import DatasetBundle, loadDataset, splitDataset
from .evaluation import classificationMetrics
from .experiments import ModelSpec
from .features import extractBOSFeatures

__all__ = [
    "DatasetBundle",
    "ModelSpec",
    "classificationMetrics",
    "extractBOSFeatures",
    "loadDataset",
    "splitDataset",
]

__version__ = "0.2.0"
