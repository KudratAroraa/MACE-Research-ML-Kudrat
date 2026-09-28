# creates report-ready Part 2 figures with one consistent method identity
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .plotting import ENTITY_COLORS, PALETTE, saveFigure
from .proposedConfig import PART2_FIGURES_DIR, ensurePart2Directories


METHOD_COLORS = {
    "RF": PALETTE["grey"],
    "Soft voting": PALETTE["purple"],
    "MACE SoF": PALETTE["coral"],
    "MACE SoF + missing-name rule": PALETTE["gold"],
    "Case-folded MACE + missing-name rule": "#8C6D1F",
    "Soft stacking": PALETTE["green"],
    "RG-MACE (hard gate)": PALETTE["navy"],
    "RG-MACE": PALETTE["teal"],
}


# save one Part 2 figure without allowing it to overwrite a Part 1 image
def savePart2Figure(figure: plt.Figure, name: str):
    ensurePart2Directories()
    return saveFigure(figure, name, directory=PART2_FIGURES_DIR)


# compare published reproduced and proposed ThingSpeak values for the two headline metrics
def plotThreeWayComparison(table: pd.DataFrame, title: str) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    styles = {
        "Paper MACE SoF": (ENTITY_COLORS["Paper"], "o", "none", 120),
        "Reproduced MACE SoF": (METHOD_COLORS["MACE SoF"], "o", METHOD_COLORS["MACE SoF"], 52),
        "RG-MACE": (METHOD_COLORS["RG-MACE"], "D", METHOD_COLORS["RG-MACE"], 62),
    }
    for axis, metric in zip(axes, ["Accuracy", "F1"]):
        for position, method in enumerate(styles):
            row = table.loc[table["Method"] == method].iloc[0]
            edge, marker, face, size = styles[method]
            axis.scatter(
                [row[metric]],
                [position],
                marker=marker,
                s=size,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.8,
                label=method,
                zorder=4,
            )
        axis.set_yticks(range(len(styles)), list(styles))
        axis.set(xlabel=f"{metric} (%)", title=metric)
        axis.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    figure.suptitle(title, x=0.01, y=1.03, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0.08, 1, 0.98))
    return figure


# plot the required task metrics for a concise set of comparison methods
def plotMetricBars(
    table: pd.DataFrame,
    methods: list[str],
    metrics: list[str],
    title: str,
) -> plt.Figure:
    subset = table.loc[table["Method"].isin(methods), ["Method", *metrics]].copy()
    long = subset.melt(id_vars="Method", var_name="Metric", value_name="Score")
    figure, axis = plt.subplots(figsize=(11.0, 5.2))
    sns.barplot(
        data=long,
        x="Metric",
        y="Score",
        hue="Method",
        palette={method: METHOD_COLORS.get(method, PALETTE["grey"]) for method in methods},
        ax=axis,
    )
    axis.set(title=title, xlabel="Evaluation metric", ylabel="Score (%)")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3)
    figure.tight_layout()
    return figure


# show robustness separately for each corruption mode and shade the pre-registered primary interval
# zoom gives each panel its own y-limits from its lowest value minus 2 up to 100 so small differences stay visible
def plotRobustnessPanels(table: pd.DataFrame, metric: str, title: str, zoom: bool = False) -> plt.Figure:
    modes = list(dict.fromkeys(table["Mode"]))
    figure, axes = plt.subplots(1, len(modes), figsize=(15.0, 4.4), sharey=not zoom)
    axes = np.atleast_1d(axes)
    for axis, mode in zip(axes, modes):
        subset = table.loc[table["Mode"] == mode]
        for method, group in subset.groupby("Method", sort=False):
            axis.plot(
                group["CorruptionRate"],
                group[metric],
                marker="o",
                color=METHOD_COLORS.get(method, PALETTE["grey"]),
                linestyle="--" if method == "RF" else "-",
                label=method,
            )
        axis.axvspan(0.0, 0.5, color=PALETTE["light"], alpha=0.55, zorder=0)
        axis.set(title=mode.capitalize(), xlabel="Share of test names corrupted")
        if zoom:
            axis.set_ylim(max(0.0, np.floor(subset[metric].min()) - 2.0), 100.0)
    axes[0].set_ylabel(f"{metric} (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    figure.suptitle(title, x=0.01, y=1.04, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0.12, 1, 0.98))
    return figure


# display how often the learned gate routes away from the MACE expert
def plotRoutingShare(table: pd.DataFrame, title: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9.2, 5.0))
    for mode, group in table.groupby("Mode", sort=False):
        axis.plot(group["CorruptionRate"], group["RfRoutingShare"], marker="o", label=mode.capitalize())
    axis.set(
        title=title,
        xlabel="Share of test names corrupted",
        ylabel="Streams routed mainly to RF (%)",
    )
    axis.legend(ncol=2)
    figure.tight_layout()
    return figure


# compare gate features for rows where RF or MACE gives stronger true-class support
def plotGateFeatureDistributions(
    gateDataset: dict,
    features: list[str],
    title: str,
) -> plt.Figure:
    mask = gateDataset["fitMask"]
    frame = gateDataset["features"].loc[mask, features].copy()
    frame["BetterExpert"] = np.where(gateDataset["targets"][mask] == 1, "MACE", "RF")
    long = frame.melt(id_vars="BetterExpert", var_name="Feature", value_name="Value")
    figure, axis = plt.subplots(figsize=(12.0, 5.6))
    sns.boxplot(
        data=long,
        x="Feature",
        y="Value",
        hue="BetterExpert",
        showfliers=False,
        palette={"MACE": METHOD_COLORS["MACE SoF"], "RF": METHOD_COLORS["RF"]},
        ax=axis,
    )
    axis.set(title=title, xlabel="Reliability feature", ylabel="Feature value")
    axis.tick_params(axis="x", rotation=40)
    figure.tight_layout()
    return figure


# show standardised logistic gate coefficients in descending absolute order
def plotGateCoefficients(coefficients: pd.DataFrame, title: str) -> plt.Figure:
    ordered = coefficients.sort_values("Coefficient")
    colors = np.where(ordered["Coefficient"] >= 0.0, METHOD_COLORS["MACE SoF"], METHOD_COLORS["RF"])
    figure, axis = plt.subplots(figsize=(9.5, 6.2))
    axis.barh(ordered["Feature"], ordered["Coefficient"], color=colors)
    axis.axvline(0.0, color=PALETTE["grey"], linewidth=1.0)
    axis.set(title=title, xlabel="Standardised logistic coefficient", ylabel="Reliability feature")
    figure.tight_layout()
    return figure


# visualise the clean-performance constraint and robust gate objective across C
def plotGateSelection(selectionTable: pd.DataFrame, title: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8.8, 5.0))
    for column, color in [
        ("CleanF1", METHOD_COLORS["MACE SoF"]),
        ("CorruptedF1", PALETTE["gold"]),
        ("Objective", METHOD_COLORS["RG-MACE"]),
    ]:
        axis.plot(selectionTable["C"], selectionTable[column], marker="o", label=column, color=color)
    axis.plot(
        selectionTable["C"],
        selectionTable["MinimumAllowedCleanF1"],
        linestyle="--",
        color=PALETTE["grey"],
        label="Minimum allowed clean F1",
    )
    axis.set_xscale("log")
    axis.set(title=title, xlabel="Logistic regularisation C", ylabel="Macro F1 (%)")
    axis.legend()
    figure.tight_layout()
    return figure


# compare real seen-name and unseen-name subsets without mixing their sample counts
def plotSubsetComparison(table: pd.DataFrame, title: str) -> plt.Figure:
    long = table.melt(
        id_vars=["Method", "Subset"],
        value_vars=["Accuracy", "F1"],
        var_name="Metric",
        value_name="Score",
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for axis, subset in zip(axes, ["Seen", "Unseen"]):
        current = long.loc[long["Subset"] == subset]
        sns.barplot(
            data=current,
            x="Metric",
            y="Score",
            hue="Method",
            palette=METHOD_COLORS,
            ax=axis,
        )
        axis.set(title=f"{subset} names", xlabel="Evaluation metric", ylabel="Score (%)")
        if axis.legend_ is not None:
            axis.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    figure.suptitle(title, x=0.01, y=1.03, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0.08, 1, 0.98))
    return figure


# compare proposed-method design choices as an annotated heatmap with colour scaled within each column
# darker cells are higher within their own column so measures on different scales stay comparable
def plotAblation(table: pd.DataFrame, metrics: list[str], title: str) -> plt.Figure:
    values = table.set_index("Variant")[metrics].astype(float)
    spread = (values.max() - values.min()).replace(0.0, 1.0)
    scaled = (values - values.min()) / spread
    figure, axis = plt.subplots(figsize=(max(10.0, 1.3 * len(metrics) + 5.0), 0.55 * len(values) + 2.4))
    sns.heatmap(
        scaled,
        annot=values.to_numpy(),
        fmt=".1f",
        cmap=sns.light_palette(PALETTE["teal"], as_cmap=True),
        cbar=False,
        linewidths=0.8,
        linecolor="#fcfcfb",
        annot_kws={"fontsize": 9},
        ax=axis,
    )
    axis.set(title=title, xlabel="", ylabel="")
    axis.set_xticks(np.arange(len(metrics)) + 0.5, metrics, rotation=30, ha="right")
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    return figure


# place each method by mean log loss and mean accuracy over repeated splits with one standard deviation bars
def plotAccuracyCalibrationTradeoff(table: pd.DataFrame, title: str) -> plt.Figure:
    statistics = table.groupby("Method", sort=False).agg(
        LogLossMean=("LogLoss", "mean"),
        LogLossStd=("LogLoss", "std"),
        AccuracyMean=("Accuracy", "mean"),
        AccuracyStd=("Accuracy", "std"),
    )
    figure, axis = plt.subplots(figsize=(9.4, 5.8))
    for method, row in statistics.iterrows():
        axis.errorbar(
            row["LogLossMean"],
            row["AccuracyMean"],
            xerr=row["LogLossStd"],
            yerr=row["AccuracyStd"],
            fmt="D" if method == "RG-MACE" else "o",
            markersize=8,
            color=METHOD_COLORS.get(method, PALETTE["grey"]),
            ecolor=METHOD_COLORS.get(method, PALETTE["grey"]),
            elinewidth=1.2,
            capsize=4,
            label=method,
        )
    axis.set(title=title, xlabel="Mean log loss (lower is better)", ylabel="Mean accuracy (%)")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=3)
    figure.tight_layout()
    return figure


# plot paired metric differences with their stratified bootstrap confidence intervals
def plotDifferenceForest(table: pd.DataFrame, title: str) -> plt.Figure:
    ordered = table.reset_index(drop=True)
    positions = np.arange(len(ordered))[::-1]
    errors = np.vstack(
        [
            ordered["ObservedDifference"] - ordered["Lower95"],
            ordered["Upper95"] - ordered["ObservedDifference"],
        ]
    )
    figure, axis = plt.subplots(figsize=(9.5, max(4.2, len(ordered) * 0.45)))
    axis.errorbar(
        ordered["ObservedDifference"],
        positions,
        xerr=errors,
        fmt="D",
        color=METHOD_COLORS["RG-MACE"],
        ecolor=PALETTE["grey"],
        capsize=4,
    )
    axis.axvline(0.0, color=PALETTE["coral"], linestyle="--")
    axis.set_yticks(positions, ordered["Comparison"])
    axis.set(title=title, xlabel="RG-MACE difference with 95% confidence interval", ylabel="Comparison")
    figure.tight_layout()
    return figure


# draw top-label calibration curves with equal-width confidence bins
def plotReliabilityDiagram(results: dict[str, dict], title: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    for method, result in results.items():
        probabilities = np.asarray(result["probabilities"])
        predictions = np.asarray(result["predictions"])
        truth = np.asarray(result["yTrue"])
        confidence = probabilities.max(axis=1)
        correct = predictions == truth
        bins = np.linspace(0.0, 1.0, 11)
        centres, observed = [], []
        for lower, upper in zip(bins[:-1], bins[1:]):
            mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
            if np.any(mask):
                centres.append(float(confidence[mask].mean()))
                observed.append(float(correct[mask].mean()))
        axis.plot(centres, observed, marker="o", color=METHOD_COLORS.get(method), label=method)
    axis.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["grey"], label="Perfect calibration")
    axis.set(title=title, xlabel="Mean top-class confidence", ylabel="Observed accuracy")
    axis.legend()
    figure.tight_layout()
    return figure
