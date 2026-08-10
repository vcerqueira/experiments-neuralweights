# Forecasting Forecasting Accuracy of Neural Networks using Their Weights

Code accompanying the paper **Forecasting Forecasting Accuracy of Neural Networks using Their Weights**.

This repository studies whether spectral weight statistics (via [WeightWatcher](https://github.com/CalculatedContent/WeightWatcher)) collected during training of neural forecasters can predict final forecast accuracy—and whether that signal can early-stop unpromising hyperparameter trials during HPO.

We call the approach **Weightcast**: a meta-model (classifier or conformal regressor) maps WeightWatcher features + hyperparameters + training step to the probability that a run will underperform a seasonal-naive baseline, and, during HPO, stops training when that probability exceeds a threshold.

## Overview

The experimental pipeline has three stages:

1. **Metadata collection** — Train neural forecasters under many hyperparameter configs, snapshot WeightWatcher features during training, and record final MASE (and baseline MASE).
2. **Meta-learning (LOO-CV)** — Train leave-one-dataset-out meta-models that predict whether (or how much) a config will exceed the baseline.
3. **HPO with Weightcast** — Use the meta-model as a PyTorch Lightning callback during Optuna-based AutoModel search, and compare against no early stopping.

Supports all architectures from NeuralForecast, with experiments being conducted with: **MLP**, **NHITS**, **PatchTST**.

## Repository structure

```
experiments-neuralweights/
├── src/
│   ├── weightcast/          # Core method: callbacks, meta-learners, AutoModel wrapper
│   ├── neural/              # Model configs, search spaces, NeuralForecast helpers
│   ├── workflows/           # HPO utilities, metadata I/O, analysis helpers
│   ├── loaders/             # Dataset loaders
│   └── config.py            # Global experiment settings
├── scripts/
│   ├── pre_metadata_collection/   # Offline WeightWatcher metadata pipeline
│   ├── metal/                     # LOO-CV for classifier / regressor meta-models
│   ├── metal_stepwise/            # Step-wise meta-evaluation & plots
│   └── hpo_search/                # Controlled & open HPO experiments + analysis
├── assets/                  # Metadata CSVs, results, plots (not all committed)
└── requirements.txt
```

### `src/weightcast/`

| Module | Role |
|--------|------|
| `callbacks.py` | `WeightcastClassifier` / `WeightcastRegressor` Lightning callbacks |
| `learner_classifier.py` | CatBoost binary classifier (optional Platt/isotonic calibration) |
| `learner_regressor.py` | CatBoost regressor + conformal predictive distributions |
| `auto.py` | `WeightcastAutoConfig` — inject Weightcast into NeuralForecast AutoModels |
| `watcher_callback.py` | Collect WeightWatcher snapshots during training |
| `watcher_summarizer.py` | Feature engineering over WeightWatcher layer details |

### `scripts/`

| Directory | Role |
|-----------|------|
| `pre_metadata_collection/` | Train configs, extract weight features, EDA, feature importance |
| `metal/` | LOO-CV evaluation; Optuna tuning scripts that cache best CatBoost params |
| `metal_stepwise/` | Metrics vs training step analysis |
| `hpo_search/` | Controlled (fixed config pool) and open (AutoModel + Optuna) HPO |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set paths and hardware defaults in `src/config.py` (e.g. `ENGINE`, `SEED`, `N_SAMPLES`).

## Reproducing experiments

Run from the repository root so that `./assets/...` paths resolve.

### 1. Metadata collection

```bash
python scripts/pre_metadata_collection/1_experiment_collection.py
python scripts/pre_metadata_collection/2_weight_feature_engineering.py
# optional: 3_eda.py, 4_feature_importance.py
```

Produces per-model metadata under `assets/` (e.g. `metadata_MLP.csv`).

### 2. Meta-model LOO-CV

Optionally tune CatBoost hyperparameters once (saved under `assets/results_cv/`):

```bash
python scripts/metal/loo_optimize_classifier.py
python scripts/metal/loo_optimize_regressor.py
```

Evaluate leave-one-dataset-out performance:

```bash
python scripts/metal/loo_cv_classifier.py
python scripts/metal/loo_cv_regressor.py
python scripts/metal/analysis.py
```

### 3. Controlled HPO (fixed config pool)

Compares Weightcast vs training without the callback on the same sampled configs. Supports **in-domain** vs **transfer** configs and **classifier** vs **regressor** meta-models:

```bash
# Edit DO_TRANSFER / USE_REGRESSOR / MODEL_NAME in the script, then:
python scripts/hpo_search/controlled_search.py
python scripts/hpo_search/controlled_analysis.py
```

Results: `assets/results_search/controlled_{search,test}_{MODEL}_{ind|transfer}_{clf|reg}.csv`


## Method sketch

Overall training of the meta-model:

![Meta-model training overview](assets/sketch.png)

During trial training, every `CB_N_STEPS` steps (after a minimum burn-in):

1. Run WeightWatcher on the current network and summarize layer statistics.
2. Concatenate config hyperparameters and the current step.
3. Predict \(P(\text{MASE} > \text{baseline})\) (classifier) or a conformal exceedance probability (regressor).
4. If the probability exceeds `STOPPING_THRESHOLD`, set `trainer.should_stop = True`.

Unlike Optuna pruners (which need within-study trial history), Weightcast uses **cross-dataset meta-knowledge** and can prune from the first trial.

## Citation (TBD)

```bibtex
@article{weightcast,
  title={Forecasting Forecasting Accuracy of Neural Networks using Their Weights},
  author={},
  year={}
}
```

