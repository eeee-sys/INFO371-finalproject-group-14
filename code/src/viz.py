"""Matplotlib styling shared by all figures.

Figures are drawn at the size they are printed in the report (FULL = the A4 text width),
so that text appears at the same size in every figure. One categorical palette in fixed
order, one sequential blue ramp, gray for context and colour for the point of the figure,
hairline grid, the report's sans-serif font.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

from common import FIGURES, ROOT

# Categorical slots, always used in this order (blue, orange, aqua, yellow, magenta, green, violet, red)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Sequential blue ramp, light -> dark
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#ffffff"
CONTEXT = "#c4c3bd"         # marks that give context (baselines, other methods)
CONTEXT_DARK = "#8f8e88"
HIGHLIGHT = CAT[0]          # the mark the figure is about

FULL = 6.6                  # inches: the text width of the A4 report
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd"] + SEQ)

# Optional font (TeX Gyre Heros from a TeX Live installation in the home folder). When it is not
# installed, matplotlib falls back to DejaVu Sans and the figures only differ in typeface.
_FONT_DIRS = [Path.home() / "texlive/main/texmf-dist/fonts/opentype/public/tex-gyre"]
for folder in _FONT_DIRS:
    for name in ("texgyreheros-regular.otf", "texgyreheros-bold.otf"):
        if (folder / name).exists():
            font_manager.fontManager.addfont(str(folder / name))


def style() -> None:
    plt.rcParams.update({
        "font.family": ["TeX Gyre Heros", "DejaVu Sans"],
        "font.size": 8,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.6,
        "axes.labelcolor": TEXT2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 8.8,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 6,
        "axes.labelsize": 7.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.color": TEXT2,
        "ytick.color": TEXT2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.4,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 3,
        "ytick.major.pad": 3,
        "legend.fontsize": 7.2,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handleheight": 0.8,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
    })


def bar_labels(ax, ys, values, texts, pad: float = 0.008, size: float = 6.9, color: str = TEXT2) -> None:
    """Value labels just past the end of horizontal bars (pad is a share of the x range)."""
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    for y, v, t in zip(ys, values, texts):
        ax.text(v + pad * span, y, t, va="center", ha="left", fontsize=size, color=color)


def titled(ax, title: str, sub: str | None = None) -> None:
    """Bold left-aligned title, with an optional gray line between the title and the plot."""
    ax.set_title(title, pad=15 if sub else 6)
    if sub:
        ax.annotate(sub, (0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                    ha="left", va="bottom", fontsize=7.2, color=TEXT2)


def save(fig, name: str) -> None:
    path = FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04, facecolor=SURFACE)
    plt.close(fig)
    print(f"figure -> {path.relative_to(ROOT)}")
