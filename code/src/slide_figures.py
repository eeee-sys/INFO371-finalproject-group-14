"""Figures shown on the result slides, in large type and with the title inside the figure.

  slide_rq1_map       UMAP map of the product text vectors (RQ1), same 9,000 products as rq3_product_map
  slide_rq2_map       UMAP map of 8,000 regular customers, coloured by taste group (RQ2)
  slide_rq3_network   garment groups bought together (RQ3), same links as rq4_category_network
  slide_baselines     MAP@12 of the baselines and of the mid-term (initial) model in the test week (RQ4)
The figures of the three nuggets are drawn by nugget_figures.py.

Outputs: tables map_products, map_customers (map coordinates); the four figures above.

Usage:
    python src/slide_figures.py --mod 20        (after the RQ scripts)
    python src/slide_figures.py --figure        (redraw only, from the saved tables)
"""
import argparse
import math

import numpy as np
import pandas as pd
from matplotlib import patheffects as pe

from common import CACHE, DATA_END, SEED, TABLES, base_views, connect, sample_events, save_table
from nugget_figures import CORAL, GOLD, GREEN, INK, BLUE, save, style
import viz

plt = viz.plt
MAP_SIZE = (6.4, 6.8)
QUERIES = ["Knit dress", "Kevin softshell jacket (1)", "Fix it tote bag"]
PRODUCT_LABELS = {"Garment Upper body": "Upper body", "Garment Lower body": "Lower body", "Garment Full body": "Full body", "Shoes": "Shoes",
                  "Accessories": "Accessories", "Underwear": "Underwear", "Swimwear": "Swimwear"}
GROUP_LABELS = {0: "Trousers", 1: "Lingerie", 2: "Swimwear", 3: "Knitwear", 4: "Dresses", 5: "Jersey basics"}      # leading garment group of C0 to C5
HALO = [pe.withStroke(linewidth=4, foreground="white")]


def map_products(con) -> None:
    """Map coordinates of the product sample; the same steps as advanced_figures.product_map."""
    import umap
    from scipy.sparse import load_npz
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    art = con.sql("SELECT article_id, prod_name, product_group_name FROM articles ORDER BY article_id").df()
    Z = normalize(TruncatedSVD(50, random_state=SEED).fit_transform(load_npz(CACHE / "rq3_tfidf_all_fields.npz").tocsr()))
    alt = pd.read_csv(TABLES / "rq3_alternatives_final.csv", dtype={"query_id": str, "article_id": str})
    row_of = pd.Index(art.article_id)
    role = {}
    for q in QUERIES:
        part = alt[alt.query_name == q]
        role.update({row_of.get_loc(a.zfill(10)): ("alternative", q) for a in part.article_id})
        role[row_of.get_loc(part.query_id.iloc[0].zfill(10))] = ("query", q)
    keep = set(np.random.default_rng(SEED).choice(len(art), 9000, replace=False).tolist()) | set(role)
    idx = np.array(sorted(keep))
    xy = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=SEED).fit_transform(Z[idx])
    save_table(pd.DataFrame(dict(article_id=art.article_id.iloc[idx].to_numpy(), group=art.product_group_name.iloc[idx].to_numpy(), x=xy[:, 0], y=xy[:, 1],
                                 role=[role.get(i, ("", ""))[0] for i in idx], query=[role.get(i, ("", ""))[1] for i in idx])), "map_products")


def map_customers(con) -> None:
    """Map coordinates of 8,000 regular customers; the same steps as rq1_clustering."""
    import umap
    from rq1_clustering import customer_features, style_vectors
    activity, _ = customer_features(con, 365, DATA_END)
    regulars = activity[activity.n_purchases >= 5]
    X, _, _ = style_vectors(con, regulars.index)
    cluster = pd.read_csv(TABLES / "rq1_assignments.csv").set_index("customer_id").cluster.reindex(regulars.index).to_numpy()
    idx = np.random.default_rng(SEED).choice(len(X), min(8000, len(X)), replace=False)
    xy = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED).fit_transform(X[idx])
    save_table(pd.DataFrame(dict(x=xy[:, 0], y=xy[:, 1], cluster=cluster[idx])), "map_customers")


def empty_axes(title: str, sub: str):
    fig = plt.figure(figsize=MAP_SIZE)
    ax = fig.add_axes([0.04, 0.03, 0.92, 0.75])
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.05, 0.965, title, fontsize=24, weight="bold", color="black", va="top", linespacing=1.1)
    fig.text(0.05, 0.835, sub, fontsize=18, color=INK, va="top")
    return fig, ax


def rq1() -> None:
    m = pd.read_csv(TABLES / "map_products.csv", keep_default_na=False)
    lo, hi = m[["x", "y"]].quantile(0.004) - 0.6, m[["x", "y"]].quantile(0.996) + 0.6
    fig, ax = empty_axes("RQ1: UMAP projection of\nproduct text vectors", "TF-IDF; each point is one product")
    names = list(PRODUCT_LABELS)
    for g in [*names, "Other"]:
        p = m[m.group == g] if g != "Other" else m[~m.group.isin(names)]
        colour = viz.CONTEXT if g == "Other" else viz.CAT[names.index(g)]
        ax.scatter(p.x, p.y, s=7, color=colour, alpha=0.45, linewidths=0)
        if g != "Other":
            # label at the densest spot of the group
            hist, ex, ey = np.histogram2d(p.x, p.y, bins=18, range=[[lo.x, hi.x], [lo.y, hi.y]])
            i, j = np.unravel_index(hist.argmax(), hist.shape)
            ax.text((ex[i] + ex[i + 1]) / 2, ey[j + 1] + 0.02 * (hi.y - lo.y), PRODUCT_LABELS[g], fontsize=18, weight="bold", color=colour, ha="center", va="bottom",
                    linespacing=0.95, path_effects=HALO, zorder=6)
    ax.set_xlim(lo.x, hi.x + 0.07 * (hi.x - lo.x)); ax.set_ylim(lo.y, hi.y)
    save(fig, "slide_rq1_map", tight=False)


def rq2() -> None:
    m = pd.read_csv(TABLES / "map_customers.csv")
    fig, ax = empty_axes("RQ2: UMAP projection of\ncustomer taste clusters (K = 6)", "K-means; each point is one customer")
    for c, label in GROUP_LABELS.items():
        p = m[m.cluster == c]
        ax.scatter(p.x, p.y, s=8, color=viz.CAT[c], alpha=0.5, linewidths=0)
        ax.text(p.x.median(), p.y.median(), label, fontsize=21, weight="bold", color=viz.CAT[c], ha="center", va="center", path_effects=HALO, zorder=5)
    save(fig, "slide_rq2_map", tight=False)


def rq3() -> None:
    from rq4_association import EDGE_COUNT, EDGE_LIFT, EDGE_SUPPORT, community_layout
    e = pd.read_csv(TABLES / "rq4_pair_edges.csv")
    e = e[(e.lift >= EDGE_LIFT) & (e.support >= EDGE_SUPPORT) & (e.co_count >= EDGE_COUNT)]
    info = pd.read_csv(TABLES / "rq4_network_nodes.csv").set_index("category")
    info = info[info.degree > 0]
    comm, size = info.community, info.purchases
    members = {c: sorted(comm.index[comm == c], key=lambda n: -size[n]) for c in sorted(comm.unique()) if c >= 0}
    pos, out = community_layout(members)
    colour = dict(zip(members, [GREEN, CORAL, GOLD, BLUE]))
    print("links:", len(e), "communities:", len(members))
    fig, ax = empty_axes("RQ3: Co-purchase network of\ngarment groups", "edge width = phi coefficient")
    for same in (False, True):
        for r in e.itertuples():
            a, b = r.category_a, r.category_b
            if (comm[a] == comm[b]) == same:
                ax.plot(*zip(pos[a], pos[b]), solid_capstyle="round", linewidth=0.8 + 26 * r.phi if same else 0.6, color=colour[comm[a]] if same else "#CFCAC0", alpha=0.75 if same else 0.5, zorder=1 + same)
    for c, names in members.items():
        for n in names:
            s = 60 + 620 * size[n] / size.max()
            ax.scatter(*pos[n], s=s, color=colour[c], edgecolor="white", linewidth=1.5, zorder=3)
            if n in ("Unknown", "Special Offers") or "/" in n:
                continue
            dx, dy = out[n]
            gap = math.sqrt(s) / 2 + 3
            text = {"Dresses Ladies": "Dresses", "Under-, Nightwear": "Underwear", "Trousers Denim": "Denim", "Socks and Tights": "Socks", "Jersey Basic": "Jersey\nBasic"}.get(n, n)
            ax.annotate(text, pos[n], xytext=(dx * gap, dy * gap), textcoords="offset points", fontsize=17, color=INK, zorder=4, path_effects=HALO,
                        ha="left" if dx > 0.35 else ("right" if dx < -0.35 else "center"), va="bottom" if dy > 0.35 else ("top" if dy < -0.35 else "center"))
    ax.margins(x=0.36, y=0.13)
    save(fig, "slide_rq3_network", tight=False)


def baselines() -> None:
    t = pd.read_csv(TABLES / "rq2_baselines.csv").query("split == 'holdout'")
    first = pd.read_csv(TABLES / "rq2_model_steps.csv").query("split == 'holdout'").iloc[0]
    rows = [("Trending (last 7 days)", "Trending", "#CFCAC0"), ("Re-buy", "Re-buy", "#CFCAC0"), ("Naive Bayes", "Naive Bayes", "#9FB4AF"),
            ("Logistic regression", "Logistic regression", "#9FB4AF"), ("Decision tree", "Decision tree", "#9FB4AF")]
    base = t[t.family.isin(["simple rule", "classifier, base features"])].set_index("method").map12
    values = [base[m] for m, _, _ in rows] + [first.map12]
    labels, colours = [l for _, l, _ in rows] + ["Our mid-term model"], [c for _, _, c in rows] + [CORAL]
    fig, ax = plt.subplots(figsize=(14.8, 5.4))
    y = np.arange(len(values))
    ax.barh(y, values, color=colours, height=0.66)
    for k, v in enumerate(values):
        last = k == len(values) - 1
        ax.text(v + 0.0005, k, f"{v:.4f}", va="center", fontsize=30 if last else 24, weight="bold" if last else "normal", color=CORAL if last else INK)
    ax.set_yticks(y, labels, fontsize=24); ax.get_yticklabels()[-1].set(weight="bold", color=CORAL)
    ax.set_xlim(0, max(values) * 1.16); ax.set_xticks([]); ax.grid(False); ax.spines["bottom"].set_visible(False)
    ax.set_title("MAP@12 by method in the test week (n = 3,031 customers)", fontsize=25, color="black", pad=14, loc="left")
    print(dict(zip(labels, np.round(values, 4))), int(first.customers))
    save(fig, "slide_baselines")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod", type=int, default=20)
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args()
    if not args.figure:
        con = connect()
        base_views(con)
        sample_events(con, args.mod)
        map_products(con)
        map_customers(con)
    style()
    rq1()
    rq2()
    rq3()
    baselines()
