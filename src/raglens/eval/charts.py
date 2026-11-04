"""Matplotlib charts for the evaluation report (PNG, headless)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

# Reference categorical palette (slots 1-5, fixed order) and chart tokens.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def _style_axes(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bar_chart(
    metrics: list[str],
    series: dict[str, list[float]],
    title: str,
    out_path: Path,
    ylabel: str = "score",
    ylim: tuple[float, float] | None = (0.0, 1.05),
) -> Path:
    """One group per metric, one bar per series (series colors in fixed order)."""
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    n = len(series)
    width = 0.8 / max(n, 1)
    x = np.arange(len(metrics))
    for i, (name, values) in enumerate(series.items()):
        offsets = x - 0.4 + width * (i + 0.5)
        bars = ax.bar(
            offsets,
            values,
            width=width - 0.02,
            color=SERIES_COLORS[i % len(SERIES_COLORS)],
            label=name,
            edgecolor=SURFACE,
            linewidth=1.0,
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=7, color=TEXT_SECONDARY, padding=2)

    ax.set_xticks(x, metrics)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, loc="left", pad=12)
    legend = ax.legend(
        frameon=False, fontsize=9, ncols=min(n, 5), loc="upper left", bbox_to_anchor=(0, -0.08)
    )
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def heatmap(
    rows: list[str],
    cols: list[str],
    values: list[list[float]],
    title: str,
    out_path: Path,
    col_label: str = "",
) -> Path:
    """Single-hue sequential heatmap with the value printed in each cell."""
    data = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    # Scale to the data so small differences between configs stay visible.
    vmin = max(0.0, float(data.min()) - 0.02)
    vmax = min(1.0, float(data.max()) + 0.005)
    image = ax.imshow(data, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (r, c), val in np.ndenumerate(data):
        ax.text(
            c,
            r,
            f"{val:.3f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if val > (vmin + vmax) / 2 else TEXT_PRIMARY,
        )
    if col_label:
        ax.set_xlabel(col_label, color=TEXT_SECONDARY)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, loc="left", pad=10)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path
