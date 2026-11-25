"""Figures of the three nuggets, drawn like the maps of RQ1 to RQ3: every mark is one object of the data.

  nugget 1   one dot = one product bought in the validation or test week; coloured if the initial model found it
             (the position inside a circle has no meaning)
  nugget 2   one dot = one product type; the share of its purchases that are duplicates, online and in stores
  nugget 3   left:  first-time purchases found by a best-seller list of a given age (line: ten weeks together, band: lowest
                    and highest week)
             right: one dot = one article in a list of a week; how its daily sales changed, last 7 days against the 23 days before

Inputs: tables rq2_model_steps, insight_repeat_types, insight_repeat_families, and the customer sample.
Outputs: tables nugget3_weeks, nugget3_articles; figures nugget1_map, nugget2_map, nugget3_lines (slide), nugget3 (report).

Usage:
    python src/nugget_figures.py --mod 20        (after insight_checks.py)
    python src/nugget_figures.py --figure        (redraw only, from the saved tables)
"""
import argparse

import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.lines import Line2D

from common import FIGURES, ROOT, SEED, TABLES, base_views, connect, sample_events, save_table
from insight_checks import FAMILIES, MEASURES
from insight_first_time import BAND_SQL, CUTOFFS, MAIN_SIZE, one_week, weekly_sales
import viz

plt = viz.plt
INK, GREEN, CORAL, GOLD, BLUE, GREY = "#27383D", "#44786C", "#E76F57", "#C9962F", "#4D7EA8", "#CFCAC0"
FAMILY_COLOUR = {"Clothes and shoes": CORAL, "Underwear, socks, tights": GOLD, "Accessories": BLUE}
HALO = [patheffects.withStroke(linewidth=9, foreground="white")]
AGES = {0: "last 7 days", 1: "7-day list, 1 week old", 2: "7-day list, 2 weeks old", 4: "7-day list, 4 weeks old", 8: "7-day list, 8 weeks old"}


def style() -> None:
    """Large type: the figures are shown on slides and, at about half the size, in the report."""
    viz.style()
    plt.rcParams.update({"font.size": 20, "axes.labelsize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 1.0, "grid.linewidth": 0.8})


def save(fig, name: str, tight: bool = True) -> None:
    """White background with a small margin, used as it is on the slides and in the report."""
    path = FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight" if tight else None, pad_inches=0.3, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"figure -> {path.relative_to(ROOT)}")


# ------------------------------------------------------------------------------------------------ nugget 1
def disc(n: int) -> np.ndarray:
    """n points filling a circle evenly (sunflower spiral); the area is proportional to n."""
    k = np.arange(n) + 0.5
    r, a = np.sqrt(k), k * np.pi * (3 - np.sqrt(5))
    return np.c_[r * np.cos(a), r * np.sin(a)]


def nugget1() -> None:
    t = pd.read_csv(TABLES / "rq2_model_steps.csv")
    t = t[t.step == t.step.iloc[0]].sum(numeric_only=True)             # the initial model, validation + test week
    bought = {"Re-buys": t.bought_same_article + t.bought_other_variant, "First-time purchases": t.bought_new_style}
    hits = {"Re-buys": t.hits_same_article + t.hits_other_variant, "First-time purchases": t.hits_new_style}
    total, found = sum(bought.values()), sum(hits.values())
    print({k: (int(bought[k]), int(hits[k]), round(bought[k] / total, 4), round(hits[k] / found, 4)) for k in bought}, int(total), int(found))

    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(14, 5.6))
    centre = {"Re-buys": (0, 0), "First-time purchases": (265, 0)}
    for name, (cx, cy) in centre.items():
        n, h = int(bought[name]), int(hits[name])
        p = disc(n) + (cx, cy)
        hit = np.zeros(n, bool); hit[rng.choice(n, h, replace=False)] = True
        ax.scatter(*p[~hit].T, s=1.5, color=GREY, linewidths=0)
        ax.scatter(*p[hit].T, s=6, color=CORAL, linewidths=0, zorder=3)
        r = np.sqrt(n)
        tx = cx - r - 14 if name == "Re-buys" else cx + r + 14
        side = "right" if name == "Re-buys" else "left"
        ax.text(tx, 52, name, ha=side, va="center", fontsize=27, weight="bold", color=INK)
        ax.text(tx, 12, f"{n / total:.0%} of purchases", ha=side, va="center", fontsize=25, color=INK)
        ax.text(tx, -34, f"{h / found:.0%} of hits", ha=side, va="center", fontsize=36, weight="bold", color=CORAL)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=9, color=CORAL, label="hit (one point = one purchase)"), Line2D([], [], marker="o", ls="", ms=9, color=GREY, label="miss")],
              loc="lower left", bbox_to_anchor=(0.0, -0.06), fontsize=21, handletextpad=0.2, labelspacing=0.3, borderpad=0)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-262, 640); ax.set_ylim(-136, 136)
    ax.set_title("Hits and misses by purchase type (n = 19,310 purchases)", fontsize=24, color="black", loc="left", pad=16)
    save(fig, "nugget1_map")


# ------------------------------------------------------------------------------------------------ nugget 2
def swarm(x: np.ndarray, gap: float, step: float) -> np.ndarray:
    """Vertical offsets so that dots closer than `gap` do not cover each other (0, +step, -step, +2 step, ...)."""
    y, placed = np.zeros(len(x)), []
    for i in np.argsort(x):
        k = 0
        while any(abs(x[i] - a) < gap and abs(k - b) < 1 for a, b in placed):
            k = -k if k > 0 else 1 - k
        placed.append((x[i], k)); y[i] = k * step
    return y


TITLES = {"two_units": "(a) Same-day purchase of two\nidentical products", "again_7d": "(b) Repurchase of the same\nproduct within 7 days"}


def nugget2() -> None:
    types, fam = pd.read_csv(TABLES / "insight_repeat_types.csv"), pd.read_csv(TABLES / "insight_repeat_families.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.4), gridspec_kw=dict(wspace=0.46))
    rows = {"Clothes and shoes": 2.45, "Accessories": 0.0}                 # y of the store row; the online row is 0.85 above
    for ax, (m, label) in zip(axes, MEASURES.items()):
        print(label, "more online:", int((types[f"{m}_online"] > types[f"{m}_store"]).sum()), "of", len(types))
        shown = types[types.group.isin(sum((FAMILIES[f] for f in rows), []))]
        top = shown[f"{m}_online"].max() * 100
        ticks, names, colours = [], [], []
        for family, y0 in rows.items():
            part, colour = types[types.group.isin(FAMILIES[family])], FAMILY_COLOUR[family]
            avg = fam[(fam.measure == label) & (fam.family == family)].iloc[0]
            for where, y, c in (("online", y0 + 0.85, colour), ("store", y0, "#8F8A80")):
                x = part[f"{m}_{where}"].to_numpy() * 100
                ax.scatter(x, y + swarm(x, top * 0.02, 0.1), s=80, color=c, alpha=0.4, linewidths=0, zorder=3)
                mean = avg[where] * 100
                ax.plot([mean, mean], [y - 0.36, y + 0.36], color=c, lw=6, solid_capstyle="round", zorder=4, path_effects=HALO)
                ticks.append(y); names.append(f"{where}  {mean:.1f}%"); colours.append(c)
            ax.text(0, y0 + 1.5, family, fontsize=23, weight="bold", color=colour, va="center")
        ax.set_xlim(0, top * 1.04); ax.set_ylim(-0.5, 4.3)
        ax.xaxis.set_major_locator(plt.FixedLocator(np.arange(0, top, 5 if top > 8 else 1)))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))
        ax.set_yticks(ticks, names, fontsize=22, weight="bold")
        for t, c in zip(ax.get_yticklabels(), colours):
            t.set_color(c)
        ax.set_xlabel("share of purchases (each point = one product type)", loc="left"); ax.grid(axis="y", visible=False)
        ax.set_title(TITLES[m], fontsize=25, color="black", pad=12, loc="left", linespacing=1.1)
    save(fig, "nugget2_map")


# ------------------------------------------------------------------------------------------------ nugget 3
def nugget3_data(mod: int) -> None:
    con = connect()
    base_views(con)
    sample_events(con, mod)
    con.execute(f"""CREATE TEMP TABLE daily AS
        SELECT t.article_id, t.t_dat AS d, {BAND_SQL} AS band, count(*) AS n
        FROM transactions_train t JOIN customers c USING (customer_id)
        WHERE t.t_dat >= DATE '{CUTOFFS[0]}' - 63 GROUP BY ALL""")
    ev = con.sql(f"""SELECT e.customer_id, e.article_id, e.t_dat AS d, e.garment_group_name AS g, {BAND_SQL} AS band
                     FROM events e JOIN customers c USING (customer_id) ORDER BY 1, 3, 2""").df()
    ev["d"] = pd.to_datetime(ev.d)
    group_of = con.sql("SELECT article_id, garment_group_name FROM articles").df().set_index("article_id").garment_group_name
    weeks, articles = [], []
    for cutoff in CUTOFFS:
        info, covered, _ = one_week(con, ev, group_of, cutoff)
        for strategy in [*AGES.values(), "last 30 days"]:
            weeks.append(dict(cutoff=cutoff, strategy=strategy, first_time=info["first_time_pairs"], covered=covered[(strategy, MAIN_SIZE)]))
        w = weekly_sales(con, cutoff).sort_index()
        rank = lambda col: w[col].rank(method="first", ascending=False).where(w[col] > 0, np.inf)
        in7, in30 = rank("s7") <= MAIN_SIZE, rank("s30") <= MAIN_SIZE
        part = np.select([in7 & in30, in7, in30], ["in both lists", "only in the 7-day list", "only in the 30-day list"], "")
        a = w[in7 | in30].reset_index()[["article_id", "s7", "s30"]].assign(cutoff=cutoff, part=part[in7 | in30])
        articles.append(a)
    save_table(pd.DataFrame(weeks), "nugget3_weeks")
    save_table(pd.concat(articles), "nugget3_articles")


def lines(ax, title: str) -> None:
    w = pd.read_csv(TABLES / "nugget3_weeks.csv")
    w["share"] = w.covered / w.first_time * 100
    c = pd.read_csv(TABLES / "insight_first_time_coverage.csv")        # the ten weeks together, as in the report
    mean = c[c.list_size == MAIN_SIZE].set_index("strategy").coverage * 100
    by_age = w.groupby("strategy").share.agg(["min", "max"]).loc[list(AGES.values())]
    xs, ys = np.array(list(AGES)), mean[list(AGES.values())].to_numpy()
    ax.fill_between(xs, by_age["min"], by_age["max"], color=CORAL, alpha=0.13, linewidth=0)
    ax.axhline(mean["last 30 days"], color=GREEN, lw=2.6, ls=(0, (4, 3)), zorder=2)
    ax.text(8.1, mean["last 30 days"] + 0.2, f"30-day list: {mean['last 30 days']:.1f}%", ha="right", va="bottom", color=GREEN, fontsize=23, weight="bold")
    ax.plot(xs, ys, color=CORAL, lw=4.5, marker="o", ms=13, zorder=4)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}%", (x, y), xytext=(-18, -36) if x == 1 else (12, 8), textcoords="offset points", color="#C9482F", fontsize=28 if x == 0 else 22, weight="bold", zorder=5)
    ax.text(8.4, 0.9, "band: lowest and highest of ten weeks", color="#B0776A", fontsize=18, ha="right", va="center")
    ax.set_xticks(xs, ["fresh", "1", "2", "4", "8"]); ax.set_xlabel("age of the 7-day best-seller list, in weeks")
    ax.set_xlim(-0.45, 8.7); ax.set_ylim(0, w.share.max() * 1.05); ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_ylabel("first-time purchases covered")
    ax.set_title(title, fontsize=24, color="black", pad=14, loc="left")


def articles(ax) -> None:
    a = pd.read_csv(TABLES / "nugget3_articles.csv")
    a["change"] = (a.s7 / 7) / ((a.s30 - a.s7) / 23).replace(0, np.nan)
    top = 16
    rng = np.random.default_rng(SEED)
    for y0, (part, label, colour) in enumerate((("only in the 30-day list", "only in\n30-day list", GREEN), ("in both lists", "in both", "#8F8A80"), ("only in the 7-day list", "only in\n7-day list", CORAL))):
        p = a[a.part == part]
        ratio = (p.s7.sum() / 7) / ((p.s30 - p.s7).sum() / 23)
        print(part, len(p), "daily sales ratio:", round(ratio, 2), "| rising:", round((p.change.fillna(np.inf) > 1).mean(), 3))
        ax.scatter(p.change.fillna(top).clip(upper=top), y0 + rng.uniform(-0.3, 0.3, len(p)), s=22, color=colour, alpha=0.5, linewidths=0, zorder=2)
        ax.plot([ratio, ratio], [y0 - 0.36, y0 + 0.36], color=colour, lw=5, solid_capstyle="round", zorder=4, path_effects=HALO)
    ax.axvline(1, color=INK, lw=2)
    ax.set_xscale("log", base=2); ax.set_xlim(1 / 15, top * 1.25); ax.set_ylim(-0.5, 2.85)
    ax.set_xticks([1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16], ["\u00f78", "\u00f74", "\u00f72", "same", "\u00d72", "\u00d74", "\u00d78", "\u226516"])
    ax.set_yticks([0, 1, 2], ["only in\n30-day list", "in both", "only in\n7-day list"], weight="bold")
    for t, colour in zip(ax.get_yticklabels(), (GREEN, "#8F8A80", CORAL)):
        t.set_color(colour)
    ax.text(0.93, 2.72, "\u2190 sales falling", ha="right", va="center", fontsize=19); ax.text(1.08, 2.72, "sales rising \u2192", ha="left", va="center", fontsize=19)
    ax.set_xlabel("sales per day: last 7 days against before"); ax.grid(axis="y", visible=False); ax.spines["left"].set_visible(False)
    ax.set_title("(b) Sales trend of listed products", fontsize=24, color="black", pad=14, loc="left")


def nugget3() -> None:
    fig, ax = plt.subplots(figsize=(14.8, 4.7)); lines(ax, "Coverage of first-time purchases by age of the best-seller list (100 products)"); save(fig, "nugget3_lines")                       # slide
    fig, (left, right) = plt.subplots(1, 2, figsize=(15.4, 5.9), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.36))      # report
    lines(left, "(a) Coverage by list age"); articles(right); save(fig, "nugget3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod", type=int, default=20)
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args()
    if not args.figure:
        nugget3_data(args.mod)
    style()
    nugget1()
    nugget2()
    nugget3()
