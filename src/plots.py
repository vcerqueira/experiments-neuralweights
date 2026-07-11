from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import plotnine as p9
from sklearn.calibration import calibration_curve

ArrayLike = Union[np.ndarray, list[float]]

_CALIB_COLORS = {
    "Perfectly calibrated": "#94a3b8",
    "Raw": "#64748b",
    "Isotonic": "#2563eb",
    "Platt": "#dc2626",
}


def _calibration_points(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        *,
        method: str,
        n_bins: int,
        strategy: str,
) -> pd.DataFrame:
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )
    return pd.DataFrame({
        "method": method,
        "mean_predicted": prob_pred,
        "fraction_positives": prob_true,
    })


def plot_calibration_curve(
        y_true: ArrayLike,
        y_prob: ArrayLike,
        *,
        y_prob_calibrated: Optional[dict[str, ArrayLike]] = None,
        n_bins: int = 10,
        strategy: str = "uniform",
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
        width: float = 7.0,
        height: float = 7.0,
        raw_label: str = "Raw",
) -> p9.ggplot:
    """Plot calibration curve (reliability diagram) for probability predictions.

    Args:
        y_true: Binary ground truth labels.
        y_prob: Raw (uncalibrated) predicted probabilities.
        y_prob_calibrated: Dict mapping method names to calibrated probabilities.
        n_bins: Number of bins for calibration curve.
        strategy: Binning strategy ("uniform" or "quantile").
        title: Plot title.
        save_path: Path to save the plot.
        show: Whether to display the plot.
        width: Saved figure width in inches.
        height: Saved figure height in inches.
        raw_label: Legend label for the raw probability curve.

    Returns:
        plotnine ggplot object.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_prob_arr = np.asarray(y_prob).ravel()

    curve_parts = [
        _calibration_points(
            y_true_arr,
            y_prob_arr,
            method=raw_label,
            n_bins=n_bins,
            strategy=strategy,
        ),
    ]

    if y_prob_calibrated:
        for method_name, probs in y_prob_calibrated.items():
            probs_arr = np.asarray(probs).ravel()
            curve_parts.append(
                _calibration_points(
                    y_true_arr,
                    probs_arr,
                    method=method_name.capitalize(),
                    n_bins=n_bins,
                    strategy=strategy,
                )
            )

    curve_df = pd.concat(curve_parts, ignore_index=True)
    perfect_df = pd.DataFrame({
        "mean_predicted": [0.0, 1.0],
        "fraction_positives": [0.0, 1.0],
    })

    color_values = [
        _CALIB_COLORS.get(method, "#64748b")
        for method in curve_df["method"].unique()
    ]

    p = (
            p9.ggplot()
            + p9.geom_line(
        perfect_df,
        p9.aes(x="mean_predicted", y="fraction_positives"),
        linetype="dashed",
        color="#94a3b8",
        size=1.0,
    )
            + p9.geom_line(
        curve_df,
        p9.aes(
            x="mean_predicted",
            y="fraction_positives",
            color="method",
            group="method",
        ),
        size=1.2,
    )
            + p9.geom_point(
        curve_df,
        p9.aes(
            x="mean_predicted",
            y="fraction_positives",
            color="method",
        ),
        size=2.5,
    )
            + p9.scale_color_manual(values=dict(zip(
        curve_df["method"].unique(),
        color_values,
    )))
            + p9.labs(
        x="Mean predicted probability",
        y="Fraction of positives",
        color=None,
        title=title,
    )
            + p9.scale_x_continuous(limits=(0, 1), breaks=np.arange(0, 1.1, 0.2))
            + p9.scale_y_continuous(limits=(0, 1), breaks=np.arange(0, 1.1, 0.2))
            + p9.theme_538(base_family="Palatino", base_size=14)
            + p9.theme(
        plot_margin=0.025,
        panel_background=p9.element_rect(fill="white"),
        plot_background=p9.element_rect(fill="white"),
        legend_box_background=p9.element_rect(fill="white"),
        strip_background=p9.element_rect(fill="white"),
        legend_background=p9.element_rect(fill="white"),
        legend_position='top',
        axis_text_y=p9.element_text(size=9),
        legend_title=p9.element_blank(),
        aspect_ratio=1,
    )
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        p.save(save_path, width=width, height=height, verbose=False)

    if show:
        print(p)

    return p
