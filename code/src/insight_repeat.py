"""Unique insight: what are repeat purchases?

Repeat purchases of the same article are rare, yet the recommender earns most of its hits on them.
This script looks at when, where and for which products the same article is bought again.

Unit of analysis: a purchase line = one customer, one article (style + colour), one day.
The data set has no size field and no returns, so everything is counted from five fields only:
customer, article, date, sales channel and the number of identical rows. All customers are used.

  two units        the line has 2+ identical rows (the same article bought twice on one day)
  two variants     the customer bought 2+ different articles of the same style (product code) that day
  again in 7 days  the customer bought the same article again 1-7 days later
  other variant    the customer bought a different article of the same style 1-7 days later
  again later      the customer bought the same article again 31-365 days later

Product types are then grouped with k-means on how they are bought (no hand-made list of
"sized" products is used).

Tables (results/tables):
  insight_repeat_channel      the measures per sales channel
  insight_repeat_weeks        chance that the same article is bought again in week k after a purchase
  insight_repeat_types        the measures per product type and channel, with the k-means group
  insight_repeat_k_selection  silhouette for 2 to 8 groups
  insight_repeat_clusters     profile and members of each group
  insight_repeat_customers    online customers by how often they buy two of the same article
Figures (results/figures): insight_repeat_groups, insight_repeat_weeks

Usage:
    python src/insight_repeat.py
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import DATA_END, SEED, base_views, connect, save_table
import viz

K_TYPES = 5                 # product-type groups shown in the report (silhouette for 2..8 is saved as well)
MIN_ONLINE, MIN_STORE = 8000, 5000      # purchase lines a product type needs in each channel to be clustered
WEEKS = 12
FEATURES = ["two_units_online", "two_units_store", "two_variants_online", "two_variants_store",
            "again_7d_online", "again_7d_store", "again_later_online", "again_later_store", "three_units_online"]


def build_lines(con) -> None:
    """Purchase lines with what the customer did next for the same article and the same style."""
    con.execute("""CREATE OR REPLACE TEMP TABLE lines AS
        SELECT t.customer_id, t.article_id, t.t_dat d, count(*) units, max((t.sales_channel_id = 2)::INT) online,
               any_value(a.product_code) style, any_value(a.product_type_name) ptype
        FROM transactions_train t JOIN articles a USING (article_id) GROUP BY 1, 2, 3""")
    # next day on which the customer bought the same style, and which articles of the style were bought then
    con.execute("""CREATE OR REPLACE TEMP TABLE style_days AS
        SELECT customer_id, style, d, count(*) variants,
               lead(d) OVER w next_d, lead(list(article_id)) OVER w next_articles, lead(count(*)) OVER w next_variants
        FROM lines GROUP BY 1, 2, 3 WINDOW w AS (PARTITION BY customer_id, style ORDER BY d)""")
    con.execute("""CREATE OR REPLACE TEMP TABLE x AS
        SELECT l.*, s.variants,
               date_diff('day', l.d, lead(l.d) OVER (PARTITION BY l.customer_id, l.article_id ORDER BY l.d)) gap_same,
               date_diff('day', l.d, s.next_d) gap_style,
               coalesce(s.next_variants - list_contains(s.next_articles, l.article_id)::INT > 0, FALSE) next_has_other
        FROM lines l JOIN style_days s USING (customer_id, style, d)""")


MEASURES = f"""
    count(*) lines,
    avg((units >= 2)::INT) two_units, avg((units >= 3)::INT) three_units, avg((variants >= 2)::INT) two_variants,
    avg(CASE WHEN d <= DATE '{DATA_END}' - 8 THEN coalesce(gap_same <= 7, FALSE)::INT END) again_7d,
    avg(CASE WHEN d <= DATE '{DATA_END}' - 8 THEN coalesce(gap_style <= 7 AND next_has_other, FALSE)::INT END) other_variant_7d,
    avg(CASE WHEN d <= DATE '{DATA_END}' - 366 THEN coalesce(gap_same BETWEEN 31 AND 365, FALSE)::INT END) again_later"""


def by_channel(con) -> pd.DataFrame:
    return con.sql(f"SELECT CASE WHEN online = 1 THEN 'online' ELSE 'store' END channel, {MEASURES} FROM x GROUP BY 1 ORDER BY 1").df()


def weekly_chance(con) -> pd.DataFrame:
    """Chance that the same article is bought again in week k, among lines not bought again before week k."""
    w = con.sql(f"""
        SELECT CASE WHEN x.online = 1 THEN 'online' ELSE 'store' END channel, k.wk, count(*) at_risk,
               sum((x.gap_same BETWEEN 7 * (k.wk - 1) + 1 AND 7 * k.wk)::INT) bought_again
        FROM x, (SELECT unnest(range(1, {WEEKS + 1})) wk) k
        WHERE x.d + CAST(7 * k.wk AS INTEGER) < DATE '{DATA_END}' AND (x.gap_same IS NULL OR x.gap_same > 7 * (k.wk - 1))
        GROUP BY 1, 2 ORDER BY 1, 2""").df().rename(columns={"wk": "week"})
    w["chance"] = w.bought_again / w.at_risk
    return w


def by_type(con) -> pd.DataFrame:
    t = con.sql(f"SELECT ptype, CASE WHEN online = 1 THEN 'online' ELSE 'store' END channel, {MEASURES} FROM x GROUP BY 1, 2").df()
    wide = t.pivot(index="ptype", columns="channel")
    wide.columns = [f"{m}_{c}" for m, c in wide.columns]
    return wide.dropna(subset=["lines_online", "lines_store"])


def cluster_types(types: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """k-means on standardised buying measures; groups are numbered by their online / store ratio of two-unit lines."""
    big = types[(types.lines_online >= MIN_ONLINE) & (types.lines_store >= MIN_STORE)].copy()
    X = StandardScaler().fit_transform(big[FEATURES])
    rows, models = [], {}
    for k in range(2, 9):
        models[k] = KMeans(k, n_init=20, random_state=SEED).fit(X)
        rows.append(dict(k=k, silhouette=silhouette_score(X, models[k].labels_), smallest_group=int(np.bincount(models[k].labels_).min())))
    big["label"] = models[K_TYPES].labels_

    def profile(g: pd.DataFrame) -> pd.Series:
        out = {c: np.average(g[c], weights=g.lines_online if c.endswith("_online") else g.lines_store) for c in FEATURES}
        out.update(product_types=len(g), lines=g.lines_online.sum() + g.lines_store.sum(),
                   members=", ".join(g.sort_values("lines_online", ascending=False).index))
        return pd.Series(out)

    prof = big.groupby("label").apply(profile, include_groups=False)
    prof["online_to_store"] = prof.two_units_online / prof.two_units_store
    order = prof.sort_values("online_to_store", ascending=False).index
    prof = prof.loc[order].reset_index(drop=True)
    prof.insert(0, "group", np.arange(1, len(prof) + 1))
    prof["share_of_lines"] = prof.lines / prof.lines.sum()
    big["group"] = big.label.map({lab: i + 1 for i, lab in enumerate(order)})
    return big.drop(columns="label"), prof, pd.DataFrame(rows)


def customers(con) -> pd.DataFrame:
    """Online customers with 10+ online purchase lines, by the share of their lines that hold two or more units."""
    c = con.sql("""
        WITH c AS (SELECT customer_id, count(*) lines, sum((units >= 2)::INT) doubles FROM lines WHERE online = 1 GROUP BY 1)
        SELECT CASE WHEN lines < 10 THEN 'fewer than 10 lines' WHEN doubles = 0 THEN 'never' WHEN doubles * 1.0 / lines < 0.1 THEN 'under 10% of lines'
                    WHEN doubles * 1.0 / lines < 0.25 THEN '10-25% of lines' ELSE '25%+ of lines' END buys_two_of_the_same,
               count(*) customers, avg(lines) lines_per_customer, sum(lines) lines, sum(doubles) double_lines
        FROM c GROUP BY 1""").df()
    order = ["fewer than 10 lines", "never", "under 10% of lines", "10-25% of lines", "25%+ of lines"]
    c = c.set_index("buys_two_of_the_same").loc[order].reset_index()
    c["share_of_customers"] = c.customers / c.customers.sum()
    c["share_of_double_lines"] = c.double_lines / c.double_lines.sum()
    c["share_of_lines"] = c.lines / c.lines.sum()
    return c


def figures(prof: pd.DataFrame, weeks: pd.DataFrame) -> None:
    viz.style()
    # groups of product types: two units of the same article, online against store
    y = np.arange(len(prof)); h = 0.36
    fig, ax = viz.plt.subplots(figsize=(viz.FULL, 0.52 * len(prof) + 0.9))
    ax.barh(y - h / 2, prof.two_units_online, height=h, color=viz.HIGHLIGHT, label="online")
    ax.barh(y + h / 2, prof.two_units_store, height=h, color=viz.CONTEXT_DARK, label="store")
    ax.set_xlim(0, max(prof.two_units_online.max(), prof.two_units_store.max()) * 1.18)
    viz.bar_labels(ax, y - h / 2, prof.two_units_online, [f"{v:.1%}" for v in prof.two_units_online])
    viz.bar_labels(ax, y + h / 2, prof.two_units_store, [f"{v:.1%}" for v in prof.two_units_store])
    labels = [", ".join(m.split(", ")[:4]) + (" ..." if m.count(",") >= 4 else "") for m in prof.members]
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xticks(np.arange(0, ax.get_xlim()[1], 0.05))
    ax.xaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("share of purchase lines with two or more units of the same article")
    ax.legend(loc="upper right")
    viz.titled(ax, "Buying two of the same article is an online habit, and it depends on the product",
               f"Product types grouped by k-means on how they are bought; all customers, {int(prof.lines.sum()):,} purchase lines")
    viz.save(fig, "insight_repeat_groups")

    # weeks after a purchase
    fig, ax = viz.plt.subplots(figsize=(viz.FULL, 2.4))
    for channel, colour in (("online", viz.HIGHLIGHT), ("store", viz.CONTEXT_DARK)):
        w = weeks[weeks.channel == channel]
        ax.plot(w.week, w.chance, color=colour, marker="o", markersize=3.5, label=channel)
        for k in (1, 2):
            v = w.chance[w.week == k].iat[0]
            ax.annotate(f"{v:.2%}", (k, v), xytext=(6, 2), textcoords="offset points", fontsize=6.9, color=viz.TEXT2)
    ax.set_xticks(range(1, WEEKS + 1)); ax.set_ylim(0, None); ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.set_xlabel("weeks after the purchase"); ax.set_ylabel("bought again in that week")
    ax.legend(loc="upper right")
    viz.titled(ax, "Repurchase probability of the same product by week after purchase",
               "All purchase lines, by sales channel")
    viz.save(fig, "insight_repeat_weeks")


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 30)
    con = connect()
    base_views(con)
    build_lines(con)

    channel = by_channel(con)
    save_table(channel, "insight_repeat_channel")
    print(channel.round(4).to_string(index=False))

    weeks = weekly_chance(con)
    save_table(weeks, "insight_repeat_weeks")
    print(weeks.pivot(index="week", columns="channel", values="chance").round(5).to_string())

    types, prof, k_table = cluster_types(by_type(con))
    save_table(types.reset_index(), "insight_repeat_types")
    save_table(k_table, "insight_repeat_k_selection")
    save_table(prof, "insight_repeat_clusters")
    print(k_table.round(3).to_string(index=False))
    print(prof[["group", "product_types", "share_of_lines", "two_units_online", "two_units_store", "online_to_store",
                "two_variants_online", "two_variants_store", "again_7d_online", "again_7d_store", "again_later_online"]].round(4).to_string(index=False))
    for row in prof.itertuples():
        print(f"  group {row.group}: {row.members}")

    cust = customers(con)
    save_table(cust, "insight_repeat_customers")
    print(cust.round(3).to_string(index=False))

    figures(prof, weeks)
