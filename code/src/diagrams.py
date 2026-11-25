"""Schematic figures for the report (no data needed). All three are drawn at the printed size,
with one data unit = one inch, so that text appears at the same size as in the charts.

    workflow_rq.png     how the four research questions feed each other
    time_split.png      the evaluation protocol of RQ4: weekly cutoffs, validation week, test week
    rq_requirements.png which required kind of analysis each research question covers, and with what

Usage:
    python src/diagrams.py
"""
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.patches import Rectangle

import viz
from rq2_categories import HOLDOUT_CUTOFF, LABEL_DAYS, PROBA_FROM, TRAIN_CUTOFFS, VALIDATION_CUTOFF

RQ_COLOUR = {"RQ1": viz.CAT[2], "RQ2": viz.CAT[3], "RQ3": viz.CAT[1], "RQ4": viz.CAT[0], "data": viz.TEXT2}
LINE = 0.135          # inches between lines of box text


def canvas(width: float, height: float):
    fig, ax = viz.plt.subplots(figsize=(width, height))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, width); ax.set_ylim(0, height); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, lines=(), colour=viz.TEXT2, bar=0.2, title_size=6.9):
    """Rectangle with a coloured title bar and a few lines of text (all sizes in inches)."""
    ax.add_patch(Rectangle((x, y), w, h, linewidth=0.8, edgecolor=colour, facecolor=viz.SURFACE, zorder=2))
    ax.add_patch(Rectangle((x, y + h - bar), w, bar, linewidth=0, facecolor=colour, zorder=3))
    ax.text(x + 0.07, y + h - bar / 2, title, ha="left", va="center", fontsize=title_size, color="white", weight="bold", zorder=4)
    for i, line in enumerate(lines):
        ax.text(x + 0.07, y + h - bar - 0.1 - LINE * i, line, ha="left", va="center", fontsize=6.2, color=viz.TEXT, zorder=4)


def path_arrow(ax, points, colour, label=None, label_at=None):
    """Polyline whose last segment ends in an arrow head; optional label on a white patch."""
    xs, ys = zip(*points)
    if len(points) > 2:
        ax.plot(xs[:-1], ys[:-1], color=colour, linewidth=1.0, solid_joinstyle="round", zorder=1)
    ax.annotate("", points[-1], points[-2], arrowprops=dict(arrowstyle="-|>", mutation_scale=7, linewidth=1.0,
                                                             color=colour, shrinkA=0, shrinkB=0), zorder=1)
    if label:
        ax.text(*label_at, label, ha="center", va="center", fontsize=5.9, color=viz.TEXT2, zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", facecolor=viz.SURFACE, edgecolor="none"))


def workflow() -> None:
    fig, ax = canvas(viz.FULL, 3.55)
    top, low, h = 2.0, 0.25, 1.2
    box(ax, 0.0, top, 1.35, h, "H&M data", ["transactions  31.8 M rows", "articles  105,542", "customers  1.37 M",
                                            "Parquet copies", "1-in-20 customer sample"], RQ_COLOUR["data"])
    box(ax, 0.0, low, 1.35, h, "Exploration", ["monthly purchases", "garment-group shares", "customer activity",
                                               "basket sizes"], RQ_COLOUR["data"])
    box(ax, 1.6, top, 1.55, h, "RQ1  Similar alternatives", ["TF-IDF of product text", "cosine similarity",
                                                              "no colour variants,", "same department,", "attribute agreement"], RQ_COLOUR["RQ1"])
    box(ax, 1.6, low, 1.55, h, "RQ3  Purchase patterns", ["baskets = customer x day", "FP-growth rules, lift, phi",
                                                           "category network", "next-visit transitions"], RQ_COLOUR["RQ3"])
    box(ax, 3.4, top, 1.4, h, "RQ2  Customer groups", ["activity tiers", "taste vectors built", "from RQ1 TF-IDF",
                                                        "SVD + K-means, K = 6", "named by their words"], RQ_COLOUR["RQ2"])
    box(ax, 5.05, low, 1.55, top + h - low, "RQ4  Next-week suggestions", [], RQ_COLOUR["RQ4"])
    box(ax, 5.13, 2.12, 1.39, 0.83, "Category layer", ["21 next-week targets", "gradient boosting", "uses RQ3 rules"],
        viz.SEQ[6], bar=0.17, title_size=6.5)
    box(ax, 5.13, 1.1, 1.39, 0.83, "Product layer", ["7 candidate sources", "ranker, 33 features", "from RQ1 to RQ3"],
        viz.SEQ[6], bar=0.17, title_size=6.5)
    box(ax, 5.13, 0.33, 1.39, 0.6, "Top 10 with reasons", ["\"you bought this\",", "\"bought together ...\""],
        viz.SEQ[6], bar=0.17, title_size=6.5)

    path_arrow(ax, [(1.35, 2.75), (1.6, 2.75)], RQ_COLOUR["data"])
    path_arrow(ax, [(1.35, 2.2), (1.47, 2.2), (1.47, 0.85), (1.6, 0.85)], RQ_COLOUR["data"])
    path_arrow(ax, [(0.675, top), (0.675, low + h)], RQ_COLOUR["data"])
    path_arrow(ax, [(3.15, 2.75), (3.4, 2.75)], RQ_COLOUR["RQ1"], "taste", (3.275, 2.86))
    path_arrow(ax, [(3.15, 2.12), (3.27, 2.12), (3.27, 1.58), (5.13, 1.58)], RQ_COLOUR["RQ1"], "text similarity", (3.85, 1.58))
    path_arrow(ax, [(4.4, top), (4.4, 1.77), (5.13, 1.77)], RQ_COLOUR["RQ2"], "style groups", (4.77, 1.77))
    path_arrow(ax, [(3.15, 1.3), (5.13, 1.3)], RQ_COLOUR["RQ3"], "rules, transitions, co-purchase", (4.1, 1.3))
    path_arrow(ax, [(5.825, 2.12), (5.825, 1.93)], viz.SEQ[6])
    path_arrow(ax, [(5.825, 1.1), (5.825, 0.93)], viz.SEQ[6])
    ax.text(0, 3.47, "Four research questions, one pipeline: the first three feed the fourth", ha="left", va="center",
            fontsize=8.8, weight="bold", color=viz.TEXT)
    viz.save(fig, "workflow_rq")


def time_split() -> None:
    train = pd.to_datetime(TRAIN_CUTOFFS)
    val, hold = pd.Timestamp(VALIDATION_CUTOFF), pd.Timestamp(HOLDOUT_CUTOFF)
    proba_from = pd.Timestamp(PROBA_FROM)
    week = pd.Timedelta(days=LABEL_DAYS)
    fig, axes = viz.plt.subplots(2, 1, figsize=(viz.FULL, 3.3), gridspec_kw=dict(height_ratios=[1, 1.15], hspace=0.75))

    # top: one cutoff, zoomed
    ax = axes[0]
    c = pd.Timestamp("2020-07-15")
    ax.axvspan(c - pd.Timedelta(days=365), c, color=viz.SEQ[0], zorder=1, linewidth=0)
    ax.axvspan(c - pd.Timedelta(days=90), c, color=viz.SEQ[2], zorder=2, linewidth=0)
    ax.axvspan(c, c + week, color=viz.CAT[1], zorder=3, linewidth=0)
    ax.axvspan(c - pd.Timedelta(days=365), c - pd.Timedelta(days=337), ymin=0, ymax=0.22, color=viz.CAT[2], zorder=3, linewidth=0)
    ax.axvline(c, color=viz.TEXT, linewidth=0.9, zorder=4)
    ax.text(c - pd.Timedelta(days=240), 0.62, "features: purchases in the 365 days before the cutoff", ha="center", fontsize=6.6, color=viz.TEXT)
    ax.text(c - pd.Timedelta(days=45), 0.62, "last 90 days", ha="center", fontsize=6.6, color=viz.TEXT)
    ax.text(c - pd.Timedelta(days=331), 0.11, "same 4 weeks one year earlier: seasonal signal", ha="left", va="center", fontsize=6.3, color=viz.TEXT)
    ax.annotate("label: bought in the\n7 days after the cutoff", (c + week, 0.5), xytext=(10, 0), textcoords="offset points",
                ha="left", va="center", fontsize=6.6, color=viz.TEXT)
    ax.set_xlim(c - pd.Timedelta(days=380), c + pd.Timedelta(days=95)); ax.set_ylim(0, 1)
    ax.set_yticks([]); ax.grid(False); ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10))); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    viz.titled(ax, f"(a) Feature and label windows at one cutoff date ({c:%d %B %Y})")

    # bottom: all cutoffs
    ax = axes[1]
    ax.axvspan(train[0], train[-1] + week, ymin=0.35, ymax=0.75, color=viz.SEQ[0], zorder=1, linewidth=0)
    for t in train:
        ax.plot([t, t], [0.4, 0.7], color=viz.HIGHLIGHT, linewidth=0.9, zorder=3)
    ax.axvspan(val, val + week, ymin=0.35, ymax=0.75, color=viz.CAT[3], zorder=2, linewidth=0)
    ax.axvspan(hold, hold + week, ymin=0.35, ymax=0.75, color=viz.CAT[1], zorder=2, linewidth=0)
    ax.text(train[0], 0.86, f"{len(train)} weekly training cutoffs, {train[0]:%d %b %Y} to {train[-1]:%d %b %Y}; label weeks do not overlap",
            ha="left", fontsize=6.6, color=viz.TEXT)
    ax.annotate(f"validation week from {val:%d %b}", (val + week / 2, 0.75), xytext=(hold + pd.Timedelta(days=30), 0.86),
                ha="left", va="center", fontsize=6.4, color=viz.TEXT, arrowprops=dict(arrowstyle="-", color=viz.TEXT2, linewidth=0.6))
    ax.annotate(f"test week {hold:%d}-{(hold + week - pd.Timedelta(days=1)):%d %b}:\nthe last full week, used once", (hold + week, 0.55),
                xytext=(hold + pd.Timedelta(days=30), 0.45), ha="left", va="center", fontsize=6.4, color=viz.TEXT,
                arrowprops=dict(arrowstyle="-", color=viz.TEXT2, linewidth=0.6))
    ax.annotate("", (proba_from, 0.2), (train[-1] + week, 0.2), arrowprops=dict(arrowstyle="<->", color=viz.TEXT2, linewidth=0.7))
    ax.text(proba_from + (train[-1] + week - proba_from) / 2, 0.06, "product layer: 12 training weeks", ha="center", fontsize=6.3, color=viz.TEXT2)
    ax.set_xlim(pd.Timestamp("2019-12-10"), pd.Timestamp("2021-01-20")); ax.set_ylim(0, 1)
    ax.set_yticks([]); ax.grid(False); ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11))); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    viz.titled(ax, "(b) Time-based split into training, validation and test weeks")
    viz.save(fig, "time_split")


REQUIREMENTS = ["Exploratory\nanalysis", "Statistical\nanalysis", "Unsupervised\nmodelling", "Predictive\nmodelling", "Visualisation"]
RQ_ROWS = [
    ("RQ1", "Similar\nalternatives", ["TF-IDF of product\ntext, cosine\nsimilarity", "", "", "", "alternatives with\nshared terms"]),
    ("RQ2", "Customer\ngroups", ["activity tiers", "silhouette by K,\nARI across seeds", "taste vectors,\nSVD, K-means", "", "group words,\nheatmap, map"]),
    ("RQ3", "Purchase\npatterns", ["basket sizes,\nvisit gaps", "support, confidence,\nlift, phi, stability", "category network,\ncommunities", "", "rules, network,\ntransitions"]),
    ("RQ4", "Next-week\nsuggestions", ["candidate\ncoverage", "validation vs test,\nablation, importance", "", "four classifiers,\nboosting ranker", "method comparison,\nablation"]),
]


def rq_requirements() -> None:
    """Matrix: which required kind of analysis each research question covers, and with what."""
    fig, ax = canvas(viz.FULL, 2.78)
    left, head, cw, rh, gap = 1.2, 2.3, 1.08, 0.54, 0.04
    for j, name in enumerate(REQUIREMENTS):
        ax.text(left + j * cw + cw / 2, head + 0.14, name, ha="center", va="center", fontsize=6.6, weight="bold", color=viz.TEXT)
    for i, (rq, name, cells) in enumerate(RQ_ROWS):
        y = head - (i + 1) * rh
        colour = RQ_COLOUR[rq]
        ax.add_patch(Rectangle((0, y + gap / 2), left - gap, rh - gap, facecolor=colour, linewidth=0))
        ax.text(0.08, y + rh / 2, rq, ha="left", va="center", fontsize=7.4, weight="bold", color="white")
        ax.text(0.42, y + rh / 2, name, ha="left", va="center", fontsize=6.4, color="white", linespacing=1.0)
        for j, text in enumerate(cells):
            x = left + j * cw
            ax.add_patch(Rectangle((x + gap / 2, y + gap / 2), cw - gap, rh - gap, linewidth=0,
                                   facecolor=colour if text else "#f4f4f2", alpha=0.16 if text else 1))
            if text:
                ax.text(x + cw / 2, y + rh / 2, text, ha="center", va="center", fontsize=5.8, color=viz.TEXT, linespacing=1.05)
    ax.text(0, 2.71, "Together the four questions cover every kind of analysis the project requires",
            ha="left", va="center", fontsize=8.8, weight="bold", color=viz.TEXT)
    viz.save(fig, "rq_requirements")


if __name__ == "__main__":
    viz.style()
    workflow()
    time_split()
    rq_requirements()
