"""Unique insight, part 2: which articles do customers buy for the first time?

Most of what customers buy in a week are articles they have not bought before, and the progress-report
recommender found very few of them. This script measures what predicts them. For ten test weeks between
October 2019 and September 2020 (the weeks of the store closure are left out), a strategy gives every
customer N articles, chosen only with data from before the week. Coverage is the share of the first-time
purchases of that week that are among the N articles.

  window     the best sellers of the last 1, 3, 7, 14 or 30 days
  list age   the 7-day best-seller list as it was 1, 2, 4 or 8 weeks earlier
  history    last week's best sellers, but only from the garment groups the customer bought before
             (slots in proportion to past purchases); last week's sales weighted by the customer's share
             of the garment group; last week's best sellers in the customer's age band

Sales are counted over all customers; the first-time purchases come from the customer sample.

Outputs: tables insight_first_time_weeks, insight_first_time_coverage, insight_first_time_rank;
figure insight_first_time_coverage.

Usage:
    python src/insight_first_time.py --mod 20
"""
import argparse

import numpy as np
import pandas as pd

from common import base_views, connect, sample_events, save_table
import viz

CUTOFFS = ["2019-10-16", "2019-11-13", "2019-12-11", "2020-01-15", "2020-02-12", "2020-03-04",
           "2020-06-10", "2020-07-08", "2020-08-12", "2020-09-16"]          # each test week starts on one of these days
LIST_SIZES = [50, 100, 200]                    # articles given to each customer
MAIN_SIZE = 100                                # list size shown in the figure
TOP_K = [10, 50, 100, 200, 500, 1000, 2000, 5000]
SOFT_WEIGHT = 0.3                              # score = last week's sales x (SOFT_WEIGHT + customer's share of the garment group)
BANDS = ["a16-24", "b25-34", "c35-44", "d45-54", "e55+"]
BAND_SQL = """CASE WHEN try_cast(c.age AS INT) IS NULL THEN 'unknown' WHEN try_cast(c.age AS INT) < 25 THEN 'a16-24'
                   WHEN try_cast(c.age AS INT) < 35 THEN 'b25-34' WHEN try_cast(c.age AS INT) < 45 THEN 'c35-44'
                   WHEN try_cast(c.age AS INT) < 55 THEN 'd45-54' ELSE 'e55+' END"""

# strategy -> (family, label in the figure); the order is the order of the bars
STRATEGIES = {
    "last 1 day": ("window", "Best sellers of the last day"),
    "last 3 days": ("window", "Best sellers of the last 3 days"),
    "last 7 days": ("window", "Best sellers of the last 7 days"),
    "last 14 days": ("window", "Best sellers of the last 14 days"),
    "last 30 days": ("window", "Best sellers of the last 30 days"),
    "7-day list, 1 week old": ("list age", "7-day list, one week old"),
    "7-day list, 2 weeks old": ("list age", "7-day list, two weeks old"),
    "7-day list, 4 weeks old": ("list age", "7-day list, four weeks old"),
    "7-day list, 8 weeks old": ("list age", "7-day list, eight weeks old"),
    "only the customer's garment groups": ("history", "Only garment groups the customer bought"),
    "sales weighted by the customer's garment groups": ("history", "Sales weighted by the customer's garment groups"),
    "customer's age band": ("history", "Best sellers in the customer's age band"),
}
REFERENCE = "last 7 days"


def weekly_sales(con, cutoff: str) -> pd.DataFrame:
    """Per article: units sold by all customers in several windows before the cutoff, and by age band in the last 7 days."""
    bands = ", ".join(f"sum(n) FILTER (d >= DATE '{cutoff}' - 7 AND band = '{b}') AS band_{i}" for i, b in enumerate(BANDS))
    return con.sql(f"""
        SELECT article_id,
               sum(n) FILTER (d >= DATE '{cutoff}' - 1) s1, sum(n) FILTER (d >= DATE '{cutoff}' - 3) s3,
               sum(n) FILTER (d >= DATE '{cutoff}' - 7) s7, sum(n) FILTER (d >= DATE '{cutoff}' - 14) s14,
               sum(n) FILTER (d >= DATE '{cutoff}' - 30) s30,
               sum(n) FILTER (d >= DATE '{cutoff}' - 14 AND d < DATE '{cutoff}' - 7) old1,
               sum(n) FILTER (d >= DATE '{cutoff}' - 21 AND d < DATE '{cutoff}' - 14) old2,
               sum(n) FILTER (d >= DATE '{cutoff}' - 35 AND d < DATE '{cutoff}' - 28) old4,
               sum(n) FILTER (d >= DATE '{cutoff}' - 63 AND d < DATE '{cutoff}' - 56) old8,
               {bands}
        FROM daily WHERE d < DATE '{cutoff}' AND d >= DATE '{cutoff}' - 63 GROUP BY 1""").df().fillna(0).set_index("article_id")


def slots_by_history(hist: pd.DataFrame, n: int) -> pd.DataFrame:
    """Split n slots over the garment groups a customer bought, in proportion to past purchases (largest remainders first)."""
    s = hist.groupby(["customer_id", "g"]).size().rename("cnt").reset_index()
    s["w"] = s.cnt / s.groupby("customer_id").cnt.transform("sum") * n
    s["q"] = np.floor(s.w)
    left = n - s.groupby("customer_id").q.transform("sum")
    s["order"] = (s.w - s.q).groupby(s.customer_id).rank(method="first", ascending=False)
    s["q"] = s.q + (s.order <= left)
    return s[["customer_id", "g", "q"]]


def one_week(con, ev: pd.DataFrame, group_of: pd.Series, cutoff: str) -> tuple[dict, dict, pd.Series]:
    """First-time purchases of the week from `cutoff`, and for every strategy and list size how many of them are covered."""
    day = pd.Timestamp(cutoff)
    w = weekly_sales(con, cutoff)
    w["g"] = group_of.reindex(w.index)

    def rank(col: str, by: str | None = None) -> pd.Series:      # 1 = best seller; articles without sales are never listed
        r = w[col].rank(method="first", ascending=False) if by is None else w.groupby(by)[col].rank(method="first", ascending=False)
        return r.where(w[col] > 0, np.inf)

    ranks = pd.DataFrame({c: rank(c) for c in ["s1", "s3", "s7", "s14", "s30", "old1", "old2", "old4", "old8"] + [f"band_{i}" for i in range(len(BANDS))]})
    ranks["s7_group"] = rank("s7", "g")

    # first-time purchases: (customer, article) pairs of the week that the customer never bought before;
    # only customers with a purchase in the 365 days before the week, as in the recommender
    past = ev[ev.d < day]
    hist = past[past.d >= day - pd.Timedelta(days=365)]
    week = ev[(ev.d >= day) & (ev.d < day + pd.Timedelta(days=7))].drop_duplicates(["customer_id", "article_id"])
    week = week[week.customer_id.isin(hist.customer_id.unique())]
    seen = past[["customer_id", "article_id"]].drop_duplicates().assign(seen=True)
    week = week.merge(seen, how="left", on=["customer_id", "article_id"])
    truth = week[week.seen.isna()].join(ranks, on="article_id")
    truth[ranks.columns] = truth[ranks.columns].fillna(np.inf)      # no sale in the last 63 days

    # rank of the bought article in the list of the customer's own age band (unknown age: the overall list)
    band_rank = np.select([truth.band == b for b in BANDS], [truth[f"band_{i}"] for i in range(len(BANDS))], default=truth.s7)

    # rank under the weighted score: last week's sales x (SOFT_WEIGHT + customer's share of the article's garment group)
    groups = sorted(w.g.dropna().unique())
    sold = [np.sort(w.s7[(w.g == g) & (w.s7 > 0)].to_numpy()) for g in groups]
    share = hist.groupby(["customer_id", "g"]).size().unstack(fill_value=0).reindex(columns=groups, fill_value=0)
    share = share.div(share.sum(axis=1), axis=0).reindex(truth.customer_id).to_numpy()
    weight = SOFT_WEIGHT + share                                      # pairs x garment groups
    own = np.array([groups.index(g) if g in groups else 0 for g in truth.g])
    s7 = w.s7.reindex(truth.article_id).fillna(0).to_numpy()
    score = s7 * weight[np.arange(len(truth)), own]
    soft_rank = np.ones(len(truth))
    for h, arr in enumerate(sold):                                    # articles of group h that score higher for this customer
        soft_rank += len(arr) - np.searchsorted(arr, score / weight[:, h], side="right")
    soft_rank = np.where(s7 > 0, soft_rank, np.inf)

    covered = {}
    for n in LIST_SIZES:
        q = truth.merge(slots_by_history(hist, n), how="left", on=["customer_id", "g"]).q.fillna(0).to_numpy()
        by_rank = {"last 1 day": truth.s1, "last 3 days": truth.s3, "last 7 days": truth.s7, "last 14 days": truth.s14,
                   "last 30 days": truth.s30, "7-day list, 1 week old": truth.old1, "7-day list, 2 weeks old": truth.old2,
                   "7-day list, 4 weeks old": truth.old4, "7-day list, 8 weeks old": truth.old8,
                   "sales weighted by the customer's garment groups": soft_rank, "customer's age band": band_rank}
        for name, r in by_rank.items():
            covered[(name, n)] = int((np.asarray(r) <= n).sum())
        covered[("only the customer's garment groups", n)] = int((truth.s7_group.to_numpy() <= q).sum())
    info = dict(cutoff=cutoff, customers=truth.customer_id.nunique(), bought_pairs=len(week), first_time_pairs=len(truth),
                first_time_share=len(truth) / len(week), articles_sold_last_week=int((w.s7 > 0).sum()))
    return info, covered, truth.s7


def figure(coverage: pd.DataFrame, n_pairs: int) -> None:
    """Coverage of first-time purchases by strategy for the main list size; the 7-day best-seller list is highlighted."""
    viz.style()
    c = coverage[coverage.list_size == MAIN_SIZE].set_index("strategy").loc[list(STRATEGIES)]
    families = [f for f, _ in STRATEGIES.values()]
    y, pos = [], 0.0
    for i, fam in enumerate(families):                               # a gap between the three families
        pos += 1.0 if i == 0 or fam == families[i - 1] else 1.9
        y.append(pos)
    y = np.array(y)
    ref = c.coverage[REFERENCE]
    fig, ax = viz.plt.subplots(figsize=(viz.FULL, 3.7))
    ax.barh(y, c.coverage, color=[viz.HIGHLIGHT if s == REFERENCE else viz.SEQ[2] for s in c.index], height=0.66)
    ax.axvline(ref, color=viz.CONTEXT, linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
    ax.set_xlim(0, c.coverage.max() * 1.22)
    viz.bar_labels(ax, y, c.coverage, [f"{v:.1%}" + ("" if s == REFERENCE else f"   {int(b)} of {len(CUTOFFS)} weeks")
                                       for s, v, b in zip(c.index, c.coverage, c.weeks_better_than_reference)])
    ax.set_yticks(y); ax.set_yticklabels([STRATEGIES[s][1] for s in c.index]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    for fam, text in (("window", "How many days of sales the list counts"), ("list age", "How old the list is"),
                      ("history", "Using the customer's own history")):
        first = y[families.index(fam)]
        ax.text(0.0, first - 0.72, text, fontsize=6.8, color=viz.TEXT2, style="italic", va="center")
    ax.set_ylim(y.max() + 0.7, y.min() - 1.3)
    ax.xaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel(f"share of first-time purchases found among {MAIN_SIZE} articles per customer\n"
                  f"(n of {len(CUTOFFS)} weeks: weeks in which the list covers more than the 7-day list)")
    viz.titled(ax, "Coverage of first-time purchases by recommendation strategy",
               f"{n_pairs:,} first-time purchases in {len(CUTOFFS)} test weeks, October 2019 to September 2020")
    viz.save(fig, "insight_first_time_coverage")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod", type=int, default=20)
    args = parser.parse_args()
    con = connect()
    base_views(con)
    sample_events(con, args.mod)

    # daily sales of every article by age band (all customers), and the purchases of the sampled customers
    con.execute(f"""CREATE TEMP TABLE daily AS
        SELECT t.article_id, t.t_dat AS d, {BAND_SQL} AS band, count(*) AS n
        FROM transactions_train t JOIN customers c USING (customer_id)
        WHERE t.t_dat >= DATE '{CUTOFFS[0]}' - 63 GROUP BY ALL""")
    ev = con.sql(f"""SELECT e.customer_id, e.article_id, e.t_dat AS d, e.garment_group_name AS g, {BAND_SQL} AS band
                     FROM events e JOIN customers c USING (customer_id) ORDER BY 1, 3, 2""").df()
    ev["d"] = pd.to_datetime(ev.d)
    group_of = con.sql("SELECT article_id, garment_group_name FROM articles").df().set_index("article_id").garment_group_name

    weeks, counts, ranks = [], [], []
    for cutoff in CUTOFFS:
        info, covered, s7_rank = one_week(con, ev, group_of, cutoff)
        weeks.append(info); counts.append(covered); ranks.append(s7_rank)
        print(f"{cutoff}: {info['first_time_pairs']:,} first-time purchases ({info['first_time_share']:.1%} of the pairs bought); "
              f"7-day list covers {covered[(REFERENCE, MAIN_SIZE)] / info['first_time_pairs']:.1%}")
    weeks = pd.DataFrame(weeks)
    save_table(weeks, "insight_first_time_weeks")
    n_pairs = int(weeks.first_time_pairs.sum())
    print(f"first-time purchases: {n_pairs:,} = {n_pairs / weeks.bought_pairs.sum():.1%} of the (customer, article) pairs bought")

    # coverage pooled over the weeks, the weakest and the best week, and in how many weeks a strategy beats the 7-day list
    per_week = pd.DataFrame(counts).div(weeks.first_time_pairs.to_numpy(), axis=0)
    pooled = pd.DataFrame(counts).sum() / n_pairs
    rows = []
    for (name, n), value in pooled.items():
        rows.append(dict(family=STRATEGIES[name][0], strategy=name, list_size=n, coverage=value,
                         weakest_week=per_week[(name, n)].min(), best_week=per_week[(name, n)].max(),
                         weeks_better_than_reference=int((per_week[(name, n)] > per_week[(REFERENCE, n)]).sum())))
    coverage = pd.DataFrame(rows).sort_values(["list_size", "family"], kind="stable").reset_index(drop=True)
    save_table(coverage, "insight_first_time_coverage")
    print(coverage[coverage.list_size == MAIN_SIZE].round(4).to_string(index=False))

    # where the bought article stood among last week's best sellers
    all_ranks = pd.concat(ranks)
    rank_table = pd.DataFrame(dict(top_k=TOP_K, share_of_first_time_purchases=[(all_ranks <= k).mean() for k in TOP_K],
                                   share_of_articles_sold_last_week=[k / weeks.articles_sold_last_week.mean() for k in TOP_K]))
    save_table(rank_table, "insight_first_time_rank")
    print(rank_table.round(4).to_string(index=False))
    print(f"not sold at all in the week before: {np.isinf(all_ranks).mean():.1%}")

    figure(coverage, n_pairs)
