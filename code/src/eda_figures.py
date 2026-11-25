"""Exploratory figures for the report, drawn from the full-data overview tables
written by prepare.py (no raw data needed).

    overview_monthly.csv         -> eda_monthly_purchases.png
    overview_garment_groups.csv  -> eda_garment_groups.png

Usage:
    python src/eda_figures.py
"""
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from common import TABLES
import viz


def monthly_purchases() -> None:
    m = pd.read_csv(TABLES / "overview_monthly.csv", parse_dates=["purchase_month"])
    m["millions"] = m.purchases / 1e6
    full = m.iloc[1:-1]                                   # first and last months are partial
    fig, ax = viz.plt.subplots(figsize=(viz.FULL, 2.35))
    ax.fill_between(full.purchase_month, full.millions, color=viz.HIGHLIGHT, alpha=0.08, linewidth=0)
    ax.plot(full.purchase_month, full.millions, color=viz.HIGHLIGHT, linewidth=1.8, zorder=3)
    for part in (m.iloc[:2], m.iloc[-2:]):                 # partial months: thin dotted continuation
        ax.plot(part.purchase_month, part.millions, color=viz.HIGHLIGHT, linewidth=1.0, linestyle=(0, (1, 2)), zorder=2)
    peaks = [full.millions[:12].idxmax(), full.millions[12:].idxmax()]
    low = full.millions.idxmin()
    for i, dy in [(peaks[0], 7), (peaks[1], 7), (low, -12)]:
        x, y = m.purchase_month[i], m.millions[i]
        ax.plot(x, y, "o", markersize=4.5, color=viz.HIGHLIGHT, markeredgecolor=viz.SURFACE, markeredgewidth=1.2, zorder=4)
        ax.annotate(f"{x:%b %Y}  {y:.2f} M", (x, y), xytext=(0, dy), textcoords="offset points",
                    ha="center", fontsize=7, color=viz.TEXT2)
    ax.set_ylim(0, 2.2); ax.set_ylabel("purchases (millions)")
    ax.grid(axis="x", visible=False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlim(m.purchase_month.iloc[0] - pd.Timedelta(days=10), m.purchase_month.iloc[-1] + pd.Timedelta(days=10))
    viz.titled(ax, "Purchases peak in June in both years",
               "Purchases per month, all 31.8 million transactions; dotted: first and last months are partial")
    viz.save(fig, "eda_monthly_purchases")


def garment_groups() -> None:
    g = pd.read_csv(TABLES / "overview_garment_groups.csv").sort_values("purchases", ascending=False).reset_index(drop=True)
    g["share"] = g.purchases / g.purchases.sum()
    g["per_article"] = g.purchases / g.articles
    top = 7
    y = np.arange(len(g))
    fig, axes = viz.plt.subplots(1, 2, figsize=(viz.FULL, 3.9), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.08), sharey=True)
    ax = axes[0]
    colours = [viz.HIGHLIGHT if i < top else viz.SEQ[1] for i in range(len(g))]
    ax.barh(y, g.purchases / 1e6, color=colours, height=0.66)
    ax.set_xlim(0, g.purchases.max() / 1e6 * 1.32)
    viz.bar_labels(ax, y, g.purchases / 1e6, [f"{p:.2f} M  {s:.0%}" for p, s in zip(g.purchases / 1e6, g.share)])
    ax.set_yticks(y); ax.set_yticklabels(g.garment_group_name); ax.invert_yaxis()
    ax.grid(axis="y", visible=False); ax.set_xlabel("purchases (millions)")
    ax.set_title(f"The {top} largest groups make {g.share[:top].sum():.0%} of purchases")
    ax = axes[1]
    best = g.per_article.idxmax()
    ax.barh(y, g.per_article, color=[viz.HIGHLIGHT if i == best else viz.CONTEXT for i in range(len(g))], height=0.66)
    ax.set_xlim(0, g.per_article.max() * 1.22)
    viz.bar_labels(ax, y, g.per_article, [f"{v:,.0f}" for v in g.per_article])
    ax.grid(axis="y", visible=False); ax.set_xlabel("purchases per article")
    ax.set_title(f"{g.garment_group_name[best]} sells most per article")
    viz.save(fig, "eda_garment_groups")


if __name__ == "__main__":
    viz.style()
    monthly_purchases()
    garment_groups()
