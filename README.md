# MACE Reproduction and Reliability-Gated MACE

This repository contains a full reproduction study of the MACE framework (Metadata-Assisted Cascading Ensemble) for automatic annotation of open IoT data streams and a proposed extension called Reliability-Gated MACE (RG-MACE)

- **Part 1** reproduces every experiment in the paper's results section on its three public datasets, compares each result with the published values and critically analyses the differences. It also tests how well the paper's conclusions hold under repeated splits, leakage-aware splits, metadata corruption, label noise and honest model selection.
- **Part 2** addresses the main limitation found in Part 1. MACE trusts the stream name for every stream through a fixed, hard filter, so it fails badly when names are missing or badly formatted and the classes it eliminates can never be recovered. RG-MACE replaces the hard filter with a learned, per-stream soft mixture of the MACE cascade and a Random Forest on statistical features. A logistic-regression gate decides from reliability evidence how far the name can be trusted for each stream and it is trained on leakage-free out-of-fold data that includes controlled examples of missing and corrupted names.

## Repository structure

```text
MACEresearch-Kudrat/
├── data/                         the three source datasets and their SHA-256 checksums
├── notebook/
│   ├── 1-DatasetAudit.ipynb       Exp1 to Exp4: data audit and feature verification
│   ├── 2-MACEReproduction.ipynb   Exp5 to Exp13: reproduction of every paper result
│   ├── 3-CriticalAnalysis.ipynb   Exp14 to Exp21: robustness and critical analysis
│   └── 4-ProposedSolution.ipynb   Exp22 to Exp30: RG-MACE design and evaluation
├── results/
│   ├── tables/                   Part 1 tables (CSV and Markdown)
│   ├── tables/part2/             Part 2 tables
│   ├── figures/                  Part 1 figures
│   ├── figures/part2/            Part 2 figures
│   ├── metrics/                  seed-100 train and test split manifests
│   └── logs/                     run logs with package versions and timings
├── scripts/
│   ├── ExecuteAll.py             command-line runner for Part 1 (Exp1 to Exp21)
│   └── ExecuteProposed.py        command-line runner for Part 2 (Exp22 to Exp30)
├── src/mace/
│   ├── config.py                 every Part 1 protocol setting in one place
│   ├── data.py                   loading, validation, splits and the data audit
│   ├── features.py               the 11 Bag-of-Summaries (BOS) statistical features
│   ├── classifiers.py            the pool classifiers (1NN-ED, kNN, SVM, Gradient Boosting, Random Forest)
│   ├── ddlNLP.py                 DDL-NLP, the Damerau-Levenshtein stream-name classifier
│   ├── timeSeries.py             1NN-DTW (Numba) and the Figure 3(a) comparators
│   ├── cascadeEngine.py          the MACE cascade, Top-k, PF and SoF filters, selection, tuning and brute force
│   ├── experiments.py            model specifications and input representations
│   ├── reproductionPipeline.py   the shared cached reproduction behind Exp5 to Exp13
│   ├── additionalExperiments.py  data audit, tuned baselines, time-series baselines and error analysis
│   ├── criticalAnalysis.py       Exp14 to Exp21
│   ├── robustness.py             metadata corruption, label noise and the Brier score
│   ├── evaluation.py             metrics, probability alignment and per-class tables
│   ├── publishedResults.py       the paper's published numbers and this project's reproduced targets
│   ├── plotting.py               Part 1 figures and the shared visual style
│   ├── paths.py                  output locations
│   ├── proposedConfig.py         every Part 2 setting and the Part 2 output folders
│   ├── reliabilityGate.py        reliability features, out-of-fold gate data, corruption augmentation and gate selection
│   ├── proposedSolution.py       RG-MACE and every Part 2 baseline
│   ├── proposedExperiments.py    Exp22 to Exp30
│   ├── proposedStatistics.py     McNemar, paired bootstrap, Wilcoxon and robustness-curve area
│   └── proposedPlotting.py       Part 2 figures
└── tests/                        unit and regression tests for Part 1 and Part 2
```

## Data

The three datasets are the ones used in the paper and released by its authors. The paper points to https://github.com/stradivarius/TSopendatastreams and the authors' code is at https://github.com/matteodeggi/IoT_Classification. They are included unchanged in `data/`.

| File | Dataset | Streams | Readings per stream | Classes | Stream names used |
|---|---|---:|---:|---:|---|
| `UrbanObservatory.meta.csv` | Urban Observatory (UrbObs) | 1,065 | 864 | 16 | No |
| `Swissex.meta.csv` | Swiss Experiment (Swissex) | 346 | 445 | 11 | No |
| `ThingspeakEU.meta.csv` | ThingSpeak (TS) | 2,121 | 96 | 21 | Yes |

All three files are headerless CSVs with the same column layout:

| Column | Content |
|---|---|
| 0 | unique observation identifier |
| 1 | stream name (the metadata used by DDL-NLP on ThingSpeak) |
| 2 to 7 | channel and location metadata.. column 2 (the channel) is used only for the channel-grouped split |
| 8 | integer class label |
| 9 onward | the fixed-length time series |

The column indices and series lengths are set in `src/mace/config.py`. The loader rejects any file with the wrong number of columns, non-numeric readings or duplicate identifiers.

To check that the data files are the original ones, run this from the `data/` folder:

```bash
shasum -a 256 -c SHA256SUMS     # macOS
sha256sum -c SHA256SUMS         # Linux
```

## Installation

Python 3.11 is required. Install everything into a virtual environment from the project root.

**macOS or Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

When opening the notebooks in VS Code or Jupyter, select the `.venv` kernel.

## Quick check

```bash
python -m pytest -q -m "not slow"     # fast checks, under a minute
python -m pytest -q                   # full suite, including end-to-end regression tests (several minutes)
```

The regression tests check that the reproduction still produces the seed-100 results reported below and that Part 2 has no leakage, is deterministic and gives the expected baseline behaviour.

## Reproducing the results

There are two equivalent routes. Both call the same functions in `src/mace/` and write to the same `results/` folders.

### Option A: notebooks 

Run the notebooks in order. For each one, choose Restart and Run All.

| Notebook | Experiments | Approximate runtime (10-core laptop) |
|---|---|---|
| `1-DatasetAudit.ipynb` | Exp1 to Exp4 | under a minute |
| `2-MACEReproduction.ipynb` | Exp5 to Exp13 | about 3 to 5 minutes |
| `3-CriticalAnalysis.ipynb` | Exp14 to Exp21 | about 20 to 40 minutes |
| `4-ProposedSolution.ipynb` | Exp22 to Exp30 | about 30 to 60 minutes |

### Option B: scripts

```bash
python scripts/ExecuteAll.py                        # Part 1, Exp1 to Exp21
python scripts/ExecuteAll.py --quick                # Part 1, Tables I and II only
python scripts/ExecuteAll.py --only Exp5 Exp11      # selected Part 1 experiments

python scripts/ExecuteProposed.py                   # Part 2, Exp22 to Exp30
python scripts/ExecuteProposed.py --quick           # Part 2, clean comparison (Exp25) only
python scripts/ExecuteProposed.py --only Exp25 Exp27
```

Each script writes a run log with package versions and per-experiment timings to `results/logs/` (`runLog.json` for Part 1 and `proposedRunLog.json` for Part 2).

### Reproducibility

- **Fixed seeds throughout -** The paper split uses seed 100. Repeated-split experiments use seeds 7, 21, 42, 77, 100, 123 and 202. Every model, corruption and resampling step uses a fixed seed and parallel runs are re-sorted into a fixed order. Rerunning gives identical tables apart from timing columns.
- **Operating systems -** Floating-point differences between operating systems can change a single ThingSpeak test prediction (1 of 637 streams, 0.16 points). This was observed for the stand-alone Random Forest between macOS and Linux.
- **Part 2 never touches Part 1 -** It only imports Part 1 functions, writes only to `results/tables/part2/` and `results/figures/part2/` and never overwrites the Part 1 split manifests.

## Experimental protocol

- **Split -** Stratified 70:30 train/test split with seed 100, following the authors' code. This gives 745/320 streams for UrbObs, 242/104 for Swissex and 1,484/637 for ThingSpeak.
- **Model selection -** Every selection and tuning step (classifier order, Top-k values, PF q, SoF sigma, the RG-MACE gate) uses 5-fold stratified cross-validation on the training set only. The test set is used once per configuration.
- **Features -** The 11 Bag-of-Summaries statistics used by the paper, unscaled as in the released code. Undefined skewness and kurtosis of constant series are set to 0, which matches the older SciPy release the authors used.
- **Classifier pools -** UrbObs and Swissex use 1NN-ED, kNN, SVM, Gradient Boosting and Random Forest. ThingSpeak uses 1NN-ED, kNN, DDL-NLP, Gradient Boosting and Random Forest. The settings follow the authors' released code (for example Random Forest with 1,000 trees and seed 100).
- **Metrics -** Accuracy, macro precision, macro recall, macro F1 and macro one-vs-rest AUC for every method. Balanced accuracy, weighted F1, log loss and Brier score are also reported where relevant.
- **Critical analysis protocols -** Seven repeated stratified splits.. a duplicate-aware split.. a channel-grouped ThingSpeak split, where no channel appears in both training and test.. metadata corruption (missing, case-swapped, one-character deletion or replacement) at 0 to 100%.. training-label noise at 0 to 20%.. and an exhaustive configuration search selected on training data only.
- **Part 2 statistics -** Paired stratified bootstrap (2,000 resamples), exact McNemar tests and paired t-intervals and Wilcoxon tests across the seven splits. The hypotheses and success criteria were fixed in notebook 4 before any Part 2 result was produced.

The full list of assumptions, the discrepancy analysis and every finding are in the notebooks and the accompanying report.

## Experiment map

| Label | Notebook | Purpose | Main outputs |
|---|---|---|---|
| Exp1 | 1 | Dataset structure and dimensions against the paper | `datasetPaperCheck`, `datasetStructure`, `splitSummary` |
| Exp2 | 1 | Class distribution and imbalance | `classDistribution`, `classImbalanceSummary`, `*ClassDistribution.png` |
| Exp3 | 1 | Constant series, undefined moments and duplicates | `datasetQualityAudit`, `datasetSummary` |
| Exp4 | 1 | Bag-of-Summaries feature verification | `featureSummary`, `featureOrder` |
| Exp5 | 2 | Standalone BOS classifiers (Figure 3(b)) | `standaloneBOS`, `figureThreeBComparison`, `figure3bPaperVsMine.png` |
| Exp6 | 2 | Training-only hyperparameter optimisation of the BOS classifiers | `tunedBosMetrics` |
| Exp7 | 2 | Time-series methods (Figure 3(a)) | `timeSeriesMetrics`, `figure3aPaperVsMine.png` |
| Exp8 | 2 | DDL-NLP metadata classifier, seen and unseen names | `ddlNLPDiagnostic` |
| Exp9 | 2 | Five-fold top-k accuracy curves (Figure 4) | `topKCurves`, `topKAccuracyCurves.png` |
| Exp10 | 2 | Classifier ordering and cascade parameter tuning | `heuristics`, `tuningCurves` |
| Exp11 | 2 | Main reproduction of Tables I and II | `table1Accuracy`, `table2MacroF1`, `allMetrics`, `maceVsFinal`, `votingSensitivity`, `table1PaperVsMine.png`, `table2PaperVsMine.png` |
| Exp12 | 2 | Heuristic evaluation across 26 classifier pools (Figures 5 and 6) | `heuristicPoolAnalysis`, `heuristicRepeatSensitivity`, `normalisedAccuracyDistributions.png`, `oraclePercentileDistributions.png` |
| Exp13 | 2 | Error analysis, uncertainty and runtime | `perClassMetrics`, `confusionMatrices`, `macroF1BootstrapIntervals`, `runtime` |
| Exp14 | 3 | Repeated-split robustness (7 seeds) | `repeatedSplitResults`, `repeatedSplitSummary`, `repeatedSplitAccuracy.png` |
| Exp15 | 3 | Duplicate-aware and channel-grouped evaluation | `leakageAwareSplitResults` |
| Exp16 | 3 | Class balance and probability quality | `calibrationAndImbalance` |
| Exp17 | 3 | ThingSpeak metadata robustness | `thingspeakMetadataRobustness`, `metadataRobustness.png` |
| Exp18 | 3 | Information-domain ablation | `informationDomainAblation` |
| Exp19 | 3 | Paper-to-code sensitivity analysis | `paperCodeSensitivity` |
| Exp20 | 3 | Training-label noise | `trainingLabelNoise`, `labelNoise.png` |
| Exp21 | 3 | Exhaustive SoF search selected on training data only | `trainingOnlyBruteForce` |
| Exp22 | 4 | Failure case and research motivation | `part2/exp22FailureEvidence`, `part2/part1FailureEvidence.png` |
| Exp23 | 4 | Out-of-fold reliability dataset and leakage checks | `part2/exp23LeakageChecks`, `part2/exp23ExpertSummary`, `part2/exp23CaseFoldCollisionAudit` |
| Exp24 | 4 | Gate training and selection | `part2/exp24GateSelection`, `part2/exp24GateDiagnostics`, `part2/exp24GateCoefficients` |
| Exp25 | 4 | Clean test comparison with the paper and the reproduction | `part2/exp25CleanMetrics`, `part2/exp25PaperComparison`, `part2/exp25Routing` |
| Exp26 | 4 | Seen and unseen names and leakage-aware splits | `part2/exp26NaturalSubsetSummary`, `part2/exp26LeakageAwareSplits` |
| Exp27 | 4 | Metadata robustness (4 corruption modes, 0 to 100%) | `part2/exp27RobustnessAreas`, `part2/metadataRobustnessAccuracy.png` |
| Exp28 | 4 | Repeated-split robustness and paired statistics | `part2/exp28RepeatedSplitSummary`, `part2/exp28RepeatedSplitStatistics`, `part2/exp28GateSelectionBySeed`, `part2/accuracyCalibrationTradeoff.png` |
| Exp29 | 4 | Ablation study and corruption-share sensitivity | `part2/exp29Ablation`, `part2/exp29CorruptionShareSensitivity`, `part2/ablationStudy.png` |
| Exp30 | 4 | Error analysis, statistical evidence and cost | `part2/exp30Mcnemar`, `part2/exp30BootstrapDifferences`, `part2/exp30Cost` |
| Verdict | 4 | Pre-registered success criteria | `part2/part2SuccessCriteria` |

Tables are saved as both `.csv` and `.md` under `results/tables/` (Part 2 under `results/tables/part2/`). Figures are saved as `.png` under `results/figures/` and `results/figures/part2/`.

## Key results

### Part 1: MACE reproduction (seed-100 split)

Accuracy and macro F1 (%), paper vs reproduced (from `results/tables/table1Accuracy.csv` and `results/tables/table2MacroF1.csv`):

| Method | UrbObs accuracy | Swissex accuracy | TS accuracy | UrbObs F1 | Swissex F1 | TS F1 |
|---|---|---|---|---|---|---|
| MACE Top-k | 90.6 / 91.2 | 84.6 / 84.6 | 84.3 / 84.3 | 87.9 / 88.2 | 83.2 / 83.2 | 67.7 / 67.6 |
| MACE PF | 90.6 / 91.2 | 83.2 / 84.6 | 83.4 / 83.5 | 87.9 / 88.2 | 83.2 / 83.2 | 63.7 / 64.8 |
| MACE SoF | 90.6 / 91.2 | 82.7 / 82.7 | 88.2 / 88.5 | 87.9 / 88.2 | 80.0 / 80.0 | 79.4 / 79.4 |

- The MACE results match the paper within 0.6 accuracy points in 8 of 9 cells and within 0.3 macro-F1 points in 8 of 9. The classifier order selected by the paper's heuristic is identical on all three datasets and DDL-NLP reproduces the paper's 73.2% exactly.
- The critical analysis shows that MACE's gain exists only on ThingSpeak, where stream names can be combined with numerical features. The paper's seed-100 split is the most favourable of seven splits for every MACE strategy.

### Part 2: RG-MACE on ThingSpeak

From `results/tables/part2/exp25CleanMetrics.csv` (seed 100) and `results/tables/part2/exp28RepeatedSplitSummary.csv` (mean over seven splits):

| Method | Seed-100 accuracy | Seed-100 macro F1 | Seed-100 log loss | 7-split accuracy | 7-split accuracy, 25% names missing | 7-split log loss |
|---|---|---|---|---|---|---|
| MACE SoF (reproduced) | 88.5 | 79.4 | 2.03 | 86.8 | 74.3 | 2.66 |
| Case-folded MACE + missing-name rule | 89.2 | 79.0 | 1.92 | 87.1 | 84.1 | 2.70 |
| Soft stacking | 87.8 | 74.4 | 0.47 | 87.8 | 84.5 | 0.45 |
| **RG-MACE** | **89.2** | **78.8** | **0.82** | **86.9** | **83.6** | **0.82** |

- RG-MACE keeps MACE's clean accuracy: it is non-inferior across seven splits and 1.0 point above the paper's 88.2% on the paper split.
- It removes MACE's collapse when names are missing: 75.5% with every name missing, against 12.6% for MACE.
- Its log loss is much lower than MACE's, because the soft mixture never assigns a class probability zero.
- The gains hold on the channel-grouped split, where no test channel appears in training.
- The ablation shows the accuracy gains come mainly from case-insensitive name matching and falling back to the numerical expert. The learned gate provides both without a hand-written rule and adds better-calibrated probabilities.
- The hypothesis that names unseen in training are a failure case was not supported and generic soft stacking proved a strong alternative across splits. Both are discussed in notebook 4.

## Environment

The project was tested on macOS with Python 3.11 and the versions pinned in `requirements.txt`: numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.9, seaborn 0.13.2, pyxDamerauLevenshtein 1.10.0, numba 0.67.0, joblib 1.6.0, tabulate 0.10.0, jupyterlab 4.5.6, ipykernel 7.3.0, nbconvert 7.17.1, nbformat 5.11.1 and pytest 9.0.2.

## Reference

[1] F. Montori, K. Liao, M. De Giosa, P. P. Jayaraman, L. Bononi, T. Sellis, and D. Georgakopoulos, "A metadata-assisted cascading ensemble classification framework for automatic annotation of open IoT data," *IEEE Internet of Things Journal*, vol. 10, no. 15, pp. 13401–13413, Aug. 2023, doi: 10.1109/JIOT.2023.3263213.
