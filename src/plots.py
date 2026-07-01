from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve

ArrayLike = Union[np.ndarray, list[float]]


def plot_roc_curve(
    y_true: ArrayLike,
    y_score: ArrayLike,
    auc: float,
    *,
    title: Optional[str] = None,
    label: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: tuple[float, float] = (6.5, 6.5),
    dpi: int = 150,
) -> tuple[Figure, Axes]:
    """Plot a ROC curve and optionally save it as PDF."""
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)

    curve_label = label or f"Model (AUC = {auc:.3f})"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")

    ax.plot(fpr, tpr, color="#2563eb", linewidth=2.8, label=curve_label)
    ax.fill_between(fpr, tpr, color="#2563eb", alpha=0.08)
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#94a3b8",
        linewidth=1.5,
        label="Random",
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
    ax.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)

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


def plot_feature_importance(
    importances: pd.Series,
    *,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    top_n: Optional[int] = 20,
    figsize: tuple[float, float] = (8.0, 6.0),
    dpi: int = 150,
) -> tuple[Figure, Axes]:
    """Plot feature importances as a horizontal bar chart."""
    scores = importances.sort_values(ascending=True)
    if top_n is not None:
        scores = scores.tail(top_n)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")

    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(scores)))
    ax.barh(scores.index, scores.values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Importance", fontsize=12, labelpad=10)
    ax.set_ylabel("Feature", fontsize=12, labelpad=10)

    if title:
        ax.set_title(title, fontsize=14, fontweight="semibold", pad=14)

    ax.grid(True, axis="x", linestyle="-", alpha=0.25, linewidth=0.8)
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
