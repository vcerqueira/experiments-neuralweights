from pathlib import Path

import pandas as pd

from src.config import DATASET_MAPPING

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

META_TYPE = 'clf'  # 'clf'/'reg'
MODEL_NAME = 'MLP'
N_TRIALS = 30

RESULTS_DIR = Path(f'./assets/results_search_open_{META_TYPE}')


def load_partial_results(results_dir: Path, model: str, n_trials: int) -> pd.DataFrame:
    pattern = f"open_test_{model}_{n_trials}_*.csv"
    files = sorted(results_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No results found matching {results_dir / pattern}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")

    if not dfs:
        raise ValueError("No valid result files found")

    return pd.concat(dfs, ignore_index=True)


def analyze_results(df: pd.DataFrame) -> None:
    """Analyze and print summary statistics."""
    df = df.set_index('dataset')
    df = df.rename(index=DATASET_MAPPING)

    perf_cols = [c for c in df.columns if not c.endswith('_steps')]
    step_cols = [c for c in df.columns if c.endswith('_steps')]

    df_perf = df[perf_cols]
    df_steps = df[step_cols]

    print(df_perf.round(3).to_string())

    print("\nMean MASE across datasets:")
    print(df_perf.mean().round(4).to_string())

    print("\nMean rank (lower is better):")
    ranks = df_perf.rank(axis=1).mean().sort_values()
    print(ranks.round(2).to_string())

    # Step efficiency
    print("\n--- Training Steps Efficiency ---")
    total_steps = df_steps.sum()
    step_pct = (total_steps / total_steps.sum() * 100).round(1)

    step_summary = pd.DataFrame({
        'Total Steps': total_steps.astype(int),
        '% of Total': step_pct,
    })
    step_summary.index = step_summary.index.str.replace('_steps', '')
    print(step_summary.to_string())

    # Relative efficiency vs baseline (RS)
    if 'RS_steps' in df_steps.columns:
        baseline_steps = df_steps['RS_steps'].sum()
        print("\nRelative steps vs RS baseline:")
        for col in df_steps.columns:
            if col != 'RS_steps':
                method = col.replace('_steps', '')
                rel = df_steps[col].sum() / baseline_steps * 100
                print(f"  {method}: {rel:.1f}%")


df = load_partial_results(RESULTS_DIR, MODEL_NAME, N_TRIALS)
analyze_results(df)
