"""EDA density plots for top features grouped by class (better vs worse than SeasonalNaive)."""
from pathlib import Path

import pandas as pd
from plotnine import (
    ggplot, aes, geom_density,
    facet_wrap, labs, theme_minimal, theme,
    element_text, scale_fill_brewer, scale_color_brewer,
    save_as_pdf_pages,
)

from src.utils import read_all_metadata

# =============================================================================
# Configuration
# =============================================================================
MODEL_NAME = 'MLP'
OUTPUT_DIR = Path('./assets/outputs/eda')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_FEATURES = ['entropy', 'sv_min', 'num_layers', 'stable_rank']

# =============================================================================
# Load data
# =============================================================================
print("Loading metadata...")
metadata, _ = read_all_metadata(
    './assets',
    MODEL_NAME,
    processed_file=f'./assets/metadata_{MODEL_NAME}.csv',
    sample_n=100000,
)

# Create class variable: better than SeasonalNaive
metadata['class'] = (metadata['mase'] < metadata['mase_sn']).map({True: 'Better', False: 'Worse'})
metadata['class'] = pd.Categorical(metadata['class'], categories=['Better', 'Worse'])

print(f"Loaded {len(metadata)} samples")
print(f"Class distribution:\n{metadata['class'].value_counts()}")

for feature in TOP_FEATURES:
    if feature not in metadata.columns:
        print(f"  Skipping {feature} (not in columns)")
        continue

    df_plot = metadata[[feature, 'class']].dropna()

    p = (
            ggplot(df_plot, aes(x=feature, fill='class', color='class'))
            + geom_density(alpha=0.4)
            + scale_fill_brewer(type='qual', palette='Set1')
            + scale_color_brewer(type='qual', palette='Set1')
            + labs(
        title=f'Distribution of {feature} by Class',
        x=feature,
        y='Density',
        fill='vs SN',
        color='vs SN',
    )
            + theme_minimal()
            + theme(figure_size=(8, 5))
    )

    save_path = OUTPUT_DIR / f'density_{feature}.pdf'
    p.save(save_path, width=7, height=7)
    print(f"  Saved: {save_path}")
