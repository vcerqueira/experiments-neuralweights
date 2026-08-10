"""HPO workflows, metadata utilities, and result analysis helpers.

Modules:
    search_utils: Train meta-models; run controlled HPO and test evaluation.
    extra_callbacks: Optuna pruning + step counting for AutoModel baselines.
    metadata_utils: Load datasets/metadata; build meta-learning (X, y) pairs.
    analysis_utils: Tables and metrics for controlled-search analysis.
    cb_config: Cached LOO-tuned CatBoost hyperparameters per model/dataset.
"""
