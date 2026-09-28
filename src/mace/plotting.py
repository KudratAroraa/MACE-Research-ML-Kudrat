# applies one visual identity and saves report-ready figures with descriptive names
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns

from .paths import FIGURES_DIR, ensureOutputDirectories


PALETTE = {
    "navy": "#193B5A",
    "teal": "#168C8C",
    "gold": "#E0A526",
    "coral": "#D7645A",
    "purple": "#7356A8",
    "green": "#4D9566",
    "grey": "#7B8794",
    "light": "#EAF1F5",
}

ENTITY_COLORS = {
    "1NN-ED": "#2a78d6",
    "RF": "#eb6834",
    "DDL-NLP": "#1baf7a",
    "GBoost": "#eda100",
    "kNN": "#e87ba4",
    "SVM": "#4a3aa7",
    "Paper": "#8a8984",
    "Mine": "#2a78d6",
}

TOP_K_MARKERS = {
    "1NN-ED": "o",
    "RF": "s",
    "DDL-NLP": "D",
    "GBoost": "^",
    "kNN": "v",
    "SVM": "P",
}


# apply a restrained and consistent plotting style across notebooks
def applyPlotStyle() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (9.2, 5.4),
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": "#e4e3df",
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#e4e3df",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


# save one figure using a descriptive lower camelCase filename
def saveFigure(figure: plt.Figure, name: str, directory: Path | None = None) -> Path:
    if not re.fullmatch(r"[a-z][A-Za-z0-9]*", name):
        raise ValueError("Figure names must use lower camelCase without labels or spaces")
    ensureOutputDirectories()
    path = (directory or FIGURES_DIR) / f"{name}.png"
    figure.savefig(path, dpi=200, facecolor="#fcfcfb")
    return path


# plot class counts while retaining the imbalance visible in the source data
def plotClassDistribution(table: pd.DataFrame, datasetName: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    sns.barplot(data=table, x="Class", y="Count", color=PALETTE["teal"], ax=axis)
    axis.set(title=f"{datasetName} class distribution", xlabel="Class label", ylabel="Observations")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    return figure


# plot controlled stress-test results across levels and optional comparison groups
# xLabel and yLabel default to the column name on x and the metric name on y
def plotRobustness(
    table: pd.DataFrame,
    x: str,
    metric: str,
    group: str | None,
    title: str,
    xLabel: str | None = None,
    yLabel: str | None = None,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9.5, 5.4))
    sns.lineplot(
        data=table,
        x=x,
        y=metric,
        hue=group,
        marker="o",
        color=None if group else PALETTE["coral"],
        ax=axis,
    )
    axis.set(title=title, ylabel=yLabel or metric)
    if xLabel is not None:
        axis.set_xlabel(xLabel)
    figure.tight_layout()
    return figure


# plot one point per seed for every method in each dataset panel with the paper's seed highlighted
def plotSeedSpread(table: pd.DataFrame, metric: str, title: str, highlightSeed: int = 100) -> plt.Figure:
    datasets = list(dict.fromkeys(table["Dataset"]))
    figure, axes = plt.subplots(1, len(datasets), figsize=(13.0, 4.6))
    axes = np.atleast_1d(axes)
    for axis, datasetName in zip(axes, datasets):
        subset = table.loc[table["Dataset"] == datasetName]
        methods = list(dict.fromkeys(subset["Method"]))
        positions = {method: index for index, method in enumerate(methods)}
        otherSeeds = subset.loc[subset["Seed"] != highlightSeed]
        paperSeed = subset.loc[subset["Seed"] == highlightSeed]
        axis.scatter(
            otherSeeds["Method"].map(positions),
            otherSeeds[metric],
            s=34,
            color=PALETTE["grey"],
            alpha=0.75,
            label="Other seeds",
            zorder=3,
        )
        axis.scatter(
            paperSeed["Method"].map(positions),
            paperSeed[metric],
            s=78,
            marker="D",
            color=PALETTE["coral"],
            edgecolor="#fcfcfb",
            linewidth=1.2,
            label=f"Seed {highlightSeed} (paper split)",
            zorder=4,
        )
        axis.set_xticks(range(len(methods)), methods, rotation=35, ha="right")
        axis.set(title=datasetName, ylabel=f"{metric} (%)")
        axis.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    figure.suptitle(title, x=0.01, y=1.04, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0.08, 1, 0.98))
    return figure


# compare published and reproduced values across three datasets
def plotPaperComparison(
    table: pd.DataFrame,
    datasetKeys: list[tuple[str, str]],
    title: str,
    xLabel: str,
) -> plt.Figure:
    figure, axes = plt.subplots(1, len(datasetKeys), figsize=(13.0, 4.1), sharey=True)
    labelColumn = "Method" if "Method" in table.columns else "Classifier"
    for axis, (columnPrefix, displayName) in zip(axes, datasetKeys):
        paperValues = table[f"{columnPrefix} Paper"].to_numpy(float)
        mineValues = table[f"{columnPrefix} Mine"].to_numpy(float)
        positions = np.arange(len(table))[::-1]
        for position, paperValue, mineValue in zip(positions, paperValues, mineValues):
            if np.isfinite(paperValue) and np.isfinite(mineValue):
                axis.plot(
                    [paperValue, mineValue],
                    [position, position],
                    color="#e4e3df",
                    linewidth=3,
                    solid_capstyle="round",
                    zorder=1,
                )
        axis.scatter(
            mineValues,
            positions,
            s=42,
            color=ENTITY_COLORS["Mine"],
            edgecolor="#fcfcfb",
            linewidth=1.2,
            label="Mine (reproduced)",
            zorder=4,
        )
        finitePaper = np.isfinite(paperValues)
        axis.scatter(
            paperValues[finitePaper],
            positions[finitePaper],
            s=110,
            facecolor="none",
            edgecolor=ENTITY_COLORS["Paper"],
            linewidth=1.8,
            label="Paper (reported)",
            zorder=5,
        )

        # zoom each panel onto its own values so small Paper versus Mine differences stay visible
        low = np.nanmin(np.concatenate([paperValues, mineValues]))
        axis.set(title=displayName, xlabel=xLabel, xlim=(max(0, np.floor(low) - 4), 100))
        axis.set_yticks(positions, table[labelColumn])
        axis.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    figure.suptitle(title, x=0.01, y=1.04, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0.08, 1, 0.98))
    return figure


# plot the three Figure 4 panels with fixed classifier styling
def plotTopKPanels(curves: pd.DataFrame, title: str) -> plt.Figure:
    datasets = ["UrbanObservatory", "Swissex", "ThingSpeakEU"]
    lowerLimits = {"UrbanObservatory": 87.5, "Swissex": 72.5, "ThingSpeakEU": 72.5}
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    for axis, datasetName in zip(axes, datasets):
        # drop k columns beyond this dataset's class count before plotting
        subset = curves.loc[curves["Dataset"] == datasetName].dropna(axis=1, how="all")
        kColumns = [column for column in subset.columns if column.startswith("k=")]
        kValues = [int(column.split("=")[1]) for column in kColumns]
        for _, row in subset.iterrows():
            modelName = row["Classifier"]
            axis.plot(
                kValues,
                row[kColumns].to_numpy(float),
                color=ENTITY_COLORS.get(modelName, PALETTE["grey"]),
                marker=TOP_K_MARKERS.get(modelName, "o"),
                markersize=4.5,
                label=modelName,
            )
        axis.set(
            title=datasetName,
            xlabel="k retained classes",
            ylabel="CV top-k accuracy (%)",
            ylim=(lowerLimits[datasetName], 100.6),
        )
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    handles, labels = [], []
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.06))
    figure.suptitle(title, x=0.01, y=1.04, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0.08, 1, 0.98))
    return figure


# plot Figure 5 or 6 distributions across the 26 starting pools
def plotHeuristicDistributions(
    table: pd.DataFrame,
    valueColumn: str,
    title: str,
    yLabel: str,
) -> plt.Figure:
    datasets = ["UrbanObservatory", "Swissex", "ThingSpeakEU"]
    strategyColors = ["#2a78d6", "#eb6834", "#1baf7a"]
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for axis, datasetName in zip(axes, datasets):
        subset = table.loc[table["Dataset"] == datasetName]
        sns.boxplot(
            data=subset,
            x="Heuristic",
            y=valueColumn,
            hue="Heuristic",
            palette=strategyColors,
            legend=False,
            width=0.58,
            showfliers=True,
            ax=axis,
        )
        axis.set(title=datasetName, xlabel="Filtering strategy", ylabel=yLabel)
    figure.suptitle(title, x=0.01, y=1.04, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    return figure
