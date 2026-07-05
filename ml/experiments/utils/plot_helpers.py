"""Shared plotting helpers for transfer learning experiment notebooks."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy


def plot_transfer_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str = "",
    ax: plt.Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float = 0.0,
    fmt: str = ".3f",
) -> plt.Axes:
    """Annotated heatmap with a diverging (RdYlGn) colormap.

    Args:
        matrix: NxN float array.
        labels: Row/column tick labels (organism names).
        title: Axes title.
        ax: Existing Axes to draw on; creates a new figure if None.
        vmin, vmax: Color scale bounds (auto if None).
        center: Value mapped to the neutral colour (default 0).
        fmt: Annotation format string.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))

    mask = np.isnan(matrix)
    sns.heatmap(
        matrix,
        ax=ax,
        mask=mask,
        annot=True, fmt=fmt,
        cmap="RdYlGn",
        center=center,
        vmin=vmin, vmax=vmax,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Eval organism ->")
    ax.set_ylabel("Model organism ↓")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    return ax


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels: list[str],
    ax: plt.Axes | None = None,
    cut_height: float | None = None,
    title: str = "Hierarchical clustering of organisms",
) -> plt.Axes:
    """Draw a scipy dendrogram with an optional cluster-cut line.

    Args:
        linkage_matrix: Output of scipy.cluster.hierarchy.linkage().
        labels: Leaf labels (organism names).
        ax: Existing Axes; creates one if None.
        cut_height: If given, draws a horizontal dashed line at this height.
        title: Axes title.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4))

    hierarchy.dendrogram(
        linkage_matrix,
        labels=labels,
        ax=ax,
        leaf_rotation=30,
        color_threshold=cut_height if cut_height is not None else 0,
    )
    if cut_height is not None:
        ax.axhline(cut_height, color="crimson", linestyle="--", linewidth=1.5,
                   label=f"cut @ {cut_height:.3f}")
        ax.legend(fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Distance")
    return ax


def plot_training_curves(
    history_df,
    ax: plt.Axes | None = None,
    label_prefix: str = "",
    metric: str = "junc_acc",
) -> plt.Axes:
    """Plot train/val metric vs epoch from a training_history DataFrame.

    Args:
        history_df: DataFrame with columns epoch, train_<metric>, val_<metric>.
        ax: Existing Axes; creates one if None.
        label_prefix: Prepended to legend labels (e.g. "ecoli -> cerevisiae: ").
        metric: Metric suffix to plot (default: 'junc_acc').
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if history_df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return ax

    epochs = history_df["epoch"]
    train_col = f"train_{metric}"
    val_col   = f"val_{metric}"

    if train_col in history_df.columns:
        ax.plot(epochs, history_df[train_col],
                label=f"{label_prefix}train", linestyle="--", alpha=0.7)
    if val_col in history_df.columns:
        ax.plot(epochs, history_df[val_col],
                label=f"{label_prefix}val", linewidth=2)

    ax.set_xlabel("Epoch")
    metric_label = metric.replace("_", " ").title()
    ax.set_ylabel(metric_label)
    ax.set_title(f"{label_prefix}{metric_label} vs epoch" if label_prefix else f"{metric_label} vs epoch")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def plot_model_comparison_bars(
    comparison_df,
    organism_col: str = "organism",
    model_cols: list[str] | None = None,
    metric: str = "junction_accuracy",
    ax: plt.Axes | None = None,
    title: str = "Model comparison per organism",
) -> plt.Axes:
    """Grouped bar chart comparing multiple model types per organism.

    Args:
        comparison_df: DataFrame with organism column and one column per model type.
        organism_col: Column name holding organism labels.
        model_cols: Columns to plot as groups (auto-detected if None).
        metric: Used as y-axis label.
        ax: Existing Axes; creates one if None.
        title: Axes title.
    """
    if model_cols is None:
        model_cols = [c for c in comparison_df.columns if c != organism_col]

    if ax is None:
        w = max(6, len(comparison_df) * len(model_cols) * 0.5)
        _, ax = plt.subplots(figsize=(w, 4))

    x = np.arange(len(comparison_df))
    width = 0.8 / len(model_cols)
    offsets = np.linspace(-(0.8 - width) / 2, (0.8 - width) / 2, len(model_cols))

    for offset, col in zip(offsets, model_cols):
        ax.bar(x + offset, comparison_df[col], width, label=col)

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df[organism_col], rotation=20, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, label="random baseline")
    ax.legend(fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    return ax
