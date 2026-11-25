"""Unique insights: the detailed numbers behind nugget 2 and nugget 3.

Nugget 2 (the same product bought twice). The product-type groups of insight_repeat.py are joined into three
families: clothes and shoes (groups 1 and 2), underwear, socks and tights (groups 4 and 5), and accessories
(group 3). Shares are weighted by purchase lines; the difference between online and store is in percentage points.

Nugget 3 (first-time purchases), on the ten test weeks of insight_first_time.py:
  windows   why the 7-day list covers more than the 30-day list although both hold 100 articles: the articles in
            both lists, only in one of them, what they cover, and how their daily sales last week compare with the
            23 days before
  groups    why a list restricted to the customer's garment groups covers less: coverage for first-time purchases
            inside and outside the garment groups the customer bought in the past year

Outputs: tables insight_repeat_families, insight_first_time_windows, insight_first_time_groups; figure insight_repeat_families.

Usage:
    python src/insight_checks.py --mod 20        (after insight_repeat.py and insight_first_time.py)
"""
import argparse

import numpy as np
import pandas as pd

from common import TABLES, base_views, connect, sample_events, save_table
from insight_first_time import BAND_SQL, CUTOFFS, MAIN_SIZE, slots_by_history, weekly_sales
import viz

FAMILIES = {"Clothes and shoes": [1, 2], "Underwear, socks, tights": [4, 5], "Accessories": [3]}     # groups of insight_repeat_clusters
MEASURES = {"two_units": "Two of the same on one day", "again_7d": "The same again within a week"}


def repeat_families() -> pd.DataFrame:
    t = pd.read_csv(TABLES / "insight_repeat_types.csv")
    rows = []
    for family, members in FAMILIES.items():
        part = t[t.group.isin(members)]
        for m, label in MEASURES.items():
            online = np.average(part[f"{m}_online"], weights=part.lines_online)
            store = np.average(part[f"{m}_store"], weights=part.lines_store)
            rows.append(dict(family=family, product_types=len(part), measure=label, online=online, store=store, points_higher_online=(online - store) * 100))
    return pd.DataFrame(rows)


def families_figure(f: pd.DataFrame) -> None:
    viz.style()
    fig, axes = viz.plt.subplots(1, 2, figsize=(viz.FULL, 1.75), sharey=True, gridspec_kw=dict(wspace=0.12))
    y, h = np.arange(len(FAMILIES)), 0.36
    for ax, label in zip(axes, MEASURES.values()):
        part = f[f.measure == label].set_index("family").loc[list(FAMILIES)]
        ax.barh(y - h / 2, part.online, height=h, color=viz.HIGHLIGHT, label="online")
        ax.barh(y + h / 2, part.store, height=h, color=viz.CONTEXT_DARK, label="store")
        ax.set_xlim(0, part.online.max() * 1.5)
        digits = 1 if part.online.max() > 0.1 else 2
        viz.bar_labels(ax, y - h / 2, part.online, [f"{v:.{digits}%}" for v in part.online])
        viz.bar_labels(ax, y + h / 2, part.store, [f"{v:.{digits}%}" for v in part.store])
        ax.set_title(label, fontsize=7.6); ax.grid(axis="y", visible=False); ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
    axes[0].set_yticks(y); axes[0].set_yticklabels(list(FAMILIES)); axes[0].invert_yaxis()
    axes[1].legend(loc="lower right")
    viz.save(fig, "insight_repeat_families")


def first_time_checks(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    con.execute(f"""CREATE TEMP TABLE daily AS
        SELECT t.article_id, t.t_dat AS d, {BAND_SQL} AS band, count(*) AS n
        FROM transactions_train t JOIN customers c USING (customer_id)
        WHERE t.t_dat >= DATE '{CUTOFFS[0]}' - 63 GROUP BY ALL""")
    ev = con.sql("SELECT customer_id, article_id, t_dat AS d, garment_group_name AS g FROM events ORDER BY 1, 3, 2").df()
    ev["d"] = pd.to_datetime(ev.d)
    group_of = con.sql("SELECT article_id, garment_group_name FROM articles").df().set_index("article_id").garment_group_name
    weeks, pairs = [], []
    for cutoff in CUTOFFS:
        day = pd.Timestamp(cutoff)
        w = weekly_sales(con, cutoff).sort_index()          # fixed order, so ties are broken the same way every run
        w["g"] = group_of.reindex(w.index)
        rank = lambda col: w[col].rank(method="first", ascending=False).where(w[col] > 0, np.inf)
        r7, r30 = rank("s7"), rank("s30")
        r7_group = w.groupby("g").s7.rank(method="first", ascending=False).where(w.s7 > 0, np.inf)

        # first-time purchases of the week, defined as in insight_first_time
        past = ev[ev.d < day]
        hist = past[past.d >= day - pd.Timedelta(days=365)]
        week = ev[(ev.d >= day) & (ev.d < day + pd.Timedelta(days=7))].drop_duplicates(["customer_id", "article_id"])
        week = week[week.customer_id.isin(hist.customer_id.unique())]
        seen = past[["customer_id", "article_id"]].drop_duplicates().assign(seen=True)
        week = week.merge(seen, how="left", on=["customer_id", "article_id"])
        truth = week[week.seen.isna()].copy()

        top7, top30 = set(r7[r7 <= MAIN_SIZE].index), set(r30[r30 <= MAIN_SIZE].index)
        parts = {"in both lists": top7 & top30, "only in the 7-day list": top7 - top30, "only in the 30-day list": top30 - top7}
        for name, articles in parts.items():
            s = w.loc[list(articles)]
            weeks.append(dict(cutoff=cutoff, part=name, articles=len(articles), covered=int(truth.article_id.isin(articles).sum()),
                              first_time=len(truth), sold_last_7=s.s7.sum(), sold_23_before=s.s30.sum() - s.s7.sum(),
                              articles_sold_7=int((w.s7 > 0).sum()), articles_sold_30=int((w.s30 > 0).sum())))

        known = hist[["customer_id", "g"]].drop_duplicates().assign(known=True)
        truth = truth.merge(known, how="left", on=["customer_id", "g"]).merge(slots_by_history(hist, MAIN_SIZE), how="left", on=["customer_id", "g"])
        truth["known"] = truth.known.notna()
        truth["overall"] = (truth.article_id.map(r7).fillna(np.inf) <= MAIN_SIZE)
        truth["own_groups"] = (truth.article_id.map(r7_group).fillna(np.inf) <= truth.q.fillna(0))
        pairs.append(truth[["known", "overall", "own_groups"]])

    t = pd.DataFrame(weeks)
    total = t[t.part == "in both lists"].first_time.sum()
    windows = t.groupby("part", sort=False).agg(articles=("articles", "mean"), covered=("covered", "sum"), sold_last_7=("sold_last_7", "sum"),
                                                sold_23_before=("sold_23_before", "sum")).reset_index()
    windows["coverage"] = windows.covered / total
    windows["daily_sales_ratio"] = (windows.sold_last_7 / 7) / (windows.sold_23_before / 23)
    windows["articles_sold_7_days"], windows["articles_sold_30_days"] = t.articles_sold_7.mean(), t.articles_sold_30.mean()
    p = pd.concat(pairs)
    rows = [dict(first_time_purchases="in garment groups bought in the past year" if k else "in other garment groups", pairs=len(g), share=len(g) / len(p),
                 overall_best_sellers=g.overall.mean(), only_own_groups=g.own_groups.mean()) for k, g in p.groupby("known", sort=False)]
    rows = sorted(rows, key=lambda r: -r["share"])
    rows.append(dict(first_time_purchases="all", pairs=len(p), share=1.0, overall_best_sellers=p.overall.mean(), only_own_groups=p.own_groups.mean()))
    return windows, pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod", type=int, default=20)
    args = parser.parse_args()
    pd.set_option("display.width", 200, "display.max_columns", 20)
    con = connect()
    base_views(con)
    sample_events(con, args.mod)

    families = repeat_families()
    save_table(families, "insight_repeat_families")
    print(families.round(4).to_string(index=False))
    families_figure(families)
    windows, groups = first_time_checks(con)
    save_table(windows, "insight_first_time_windows")
    save_table(groups, "insight_first_time_groups")
    print(windows.round(4).to_string(index=False))
    print(groups.round(4).to_string(index=False))
