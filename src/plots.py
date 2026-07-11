from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve

ArrayLike = Union[np.ndarray, list[float]]


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
        figsize: tuple[float, float] = (7.0, 6.0),
        dpi: int = 150,
) -> tuple[Figure, Axes]:
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
        figsize: Figure size.
        dpi: Figure resolution.

    Returns:
        Figure and Axes objects.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_prob_arr = np.asarray(y_prob).ravel()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")

    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1.5, label="Perfectly calibrated")

    prob_true_raw, prob_pred_raw = calibration_curve(
        y_true_arr, y_prob_arr, n_bins=n_bins, strategy=strategy
    )
    ax.plot(
        prob_pred_raw,
        prob_true_raw,
        marker="o",
        markersize=6,
        linewidth=2,
        color="#64748b",
        label="Raw (conformal)",
    )

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    if y_prob_calibrated:
        for i, (method_name, probs) in enumerate(y_prob_calibrated.items()):
            probs_arr = np.asarray(probs).ravel()
            prob_true, prob_pred = calibration_curve(
                y_true_arr, probs_arr, n_bins=n_bins, strategy=strategy
            )
            color = colors[i % len(colors)]
            ax.plot(
                prob_pred,
                prob_true,
                marker="s",
                markersize=5,
                linewidth=2,
                color=color,
                label=method_name.capitalize(),
            )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted probability", fontsize=12, labelpad=10)
    ax.set_ylabel("Fraction of positives", fontsize=12, labelpad=10)

    if title:
        ax.set_title(title, fontsize=14, fontweight="semibold", pad=14)

    ax.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.95, fontsize=10)
    ax.grid(True, linestyle="-", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    elif save_path is not None:
        plt.close(fig)

    return fig, ax
