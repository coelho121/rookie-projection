"""Shared chart style for the notebooks.

Light surface, recessive grid and axes, thin marks. Categorical colours are
used in this fixed order (never cycled); past three series, fold into
"other" or split the chart.

Usage:
    from plot_style import apply_style, BLUE, ORANGE, AQUA, NEUTRAL
    apply_style()
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
NEUTRAL = "#8a8984"  # reference lines, "other", baselines

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "red_gray_blue", ["#e34948", "#f0efec", "#2a78d6"])


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "font.size": 10, "figure.dpi": 110, "lines.linewidth": 2,
        "legend.frameon": False,
    })
