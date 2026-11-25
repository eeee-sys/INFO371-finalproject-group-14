"""RQ1: Who are our customer groups?

Two-stage segmentation over the last 365 days of the sample:

1. Activity tiers (descriptive): how many purchases a customer made in the year.
2. Preference clusters (K-means) among regular customers (>= --min-purchases in the year).
   Two feature spaces are available:
     style       (default) each customer is the TF-IDF "taste vector" of the products they
                 bought (from RQ3), reduced to 50 dimensions with SVD. Clusters are named by
                 the words that distinguish them, e.g. "floral, chiffon, midi".
     categories  the share of each garment group and department in the customer's purchases
                 (square-rooted); clusters are named by over-represented groups.

Usage:
    python src/rq1_clustering.py --mod 20 --k-final 6            # report run: style clusters, K fixed at 6
    python src/rq1_clustering.py --mod 20                        # automatic K (highest silhouette)
    python src/rq1_clustering.py --mod 20 --features categories  # category-share clusters for comparison
"""
import argparse
import textwrap

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

from common import CACHE, DATA_END, DEPARTMENT_SQL, DEPARTMENTS, SEED, base_views, connect, garment_groups, sample_events, save_table
import viz

TIERS = [(1, 1, "1 purchase"), (2, 4, "2-4 purchases"), (5, 19, "5-19 purchases"), (20, 10**9, "20+ purchases")]
SVD_DIMS = 50
TOP_TERMS = 8


# ----------------------------------------------------------------------------- features
def customer_features(con, window_days: int, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-customer activity numbers, and garment-group + department shares, for [end - window, end)."""
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW w AS
        SELECT e.*, {DEPARTMENT_SQL} AS department
        FROM events e JOIN articles a USING (article_id)
        WHERE e.t_dat >= DATE '{end}' - INTERVAL '{window_days} days' AND e.t_dat < DATE '{end}'""")
    activity = con.sql(f"""
        SELECT customer_id,
               count(*)                                   AS n_purchases,
               count(DISTINCT t_dat)                      AS n_days,
               count(DISTINCT garment_group_name)         AS n_categories,
               sum(price)                                 AS spend,
               avg(price)                                 AS mean_price,
               date_diff('day', max(t_dat), DATE '{end}') AS recency_days,
               avg(CASE WHEN sales_channel_id = 2 THEN 1.0 ELSE 0.0 END) AS online_share
        FROM w GROUP BY 1 ORDER BY 1""").df().set_index("customer_id")   # fixed row order for K-means
    age = con.sql("SELECT customer_id, TRY_CAST(age AS DOUBLE) AS age FROM customers").df().set_index("customer_id")
    activity = activity.join(age, how="left")

    counts = con.sql("SELECT customer_id, garment_group_name, count(*) n FROM w GROUP BY 1, 2").df()
    shares = counts.pivot(index="customer_id", columns="garment_group_name", values="n")
    shares = shares.reindex(index=activity.index, columns=garment_groups(con)).fillna(0.0)
    shares = shares.div(shares.sum(axis=1), axis=0)
    dept = con.sql("SELECT customer_id, department, count(*) n FROM w GROUP BY 1, 2").df()
    dept = dept.pivot(index="customer_id", columns="department", values="n")
    dept = dept.reindex(index=activity.index, columns=DEPARTMENTS).fillna(0.0)
    dept = dept.div(dept.sum(axis=1), axis=0)
    return activity, pd.concat([shares, dept], axis=1)


def style_vectors(con, customers: pd.Index) -> tuple[np.ndarray, csr_matrix, np.ndarray]:
    """Taste vector per customer: the TF-IDF rows of the products bought (RQ3), weighted by
    log purchase count, averaged and L2-normalised; then reduced with SVD."""
    tfidf = load_npz(CACHE / "rq3_tfidf_all_fields.npz").tocsr()
    terms = np.array((CACHE / "rq3_tfidf_terms.txt").read_text(encoding="utf8").split("\n"))
    catalog = con.sql("SELECT article_id FROM articles ORDER BY article_id").df().article_id
    assert tfidf.shape[0] == len(catalog), "rerun src/rq3_text_mining.py: TF-IDF matrix does not match the article table"
    article_pos = pd.Index(catalog)
    hist = con.sql("SELECT customer_id, article_id, count(*) n FROM w GROUP BY 1, 2").df()
    hist = hist[hist.customer_id.isin(customers)]
    rows = customers.get_indexer(hist.customer_id)
    cols = article_pos.get_indexer(hist.article_id)
    H = csr_matrix((np.log1p(hist.n.to_numpy(dtype=np.float32)), (rows, cols)), shape=(len(customers), len(catalog)))
    profiles = normalize(H @ tfidf)                                   # customers x terms, sparse
    svd = TruncatedSVD(SVD_DIMS, random_state=SEED)
    Z = normalize(svd.fit_transform(profiles)).astype(np.float32)   # customers x 50, unit length
    print(f"style vectors: {profiles.shape[0]:,} customers x {profiles.shape[1]:,} terms -> {SVD_DIMS} dims "
          f"(explained variance {svd.explained_variance_ratio_.sum():.0%})")
    return Z, profiles, terms


# ----------------------------------------------------------------------------- clustering
def activity_tiers(activity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for low, high, label in TIERS:
        part = activity[(activity.n_purchases >= low) & (activity.n_purchases <= high)]
        rows.append(dict(
            tier=label, customers=len(part), share=len(part) / len(activity),
            mean_purchases=part.n_purchases.mean(), mean_spend=part.spend.mean(), mean_price=part.mean_price.mean(),
            online_share=part.online_share.mean(), mean_categories=part.n_categories.mean(), median_age=part.age.median(),
            purchase_share=part.n_purchases.sum() / activity.n_purchases.sum()))
    return pd.DataFrame(rows)


def choose_k(X: np.ndarray, ks: list[int], sample_n: int) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), min(sample_n, len(X)), replace=False)
    rows, labels = [], {}
    for k in ks:
        model = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit(X)
        lab = model.labels_
        sizes = np.bincount(lab, minlength=k) / len(lab)
        rows.append(dict(k=k, silhouette=silhouette_score(X[idx], lab[idx]), inertia=model.inertia_, smallest_cluster_share=sizes.min()))
        labels[k] = lab
        print(f"k={k}: silhouette={rows[-1]['silhouette']:.3f} smallest={sizes.min():.3f}")
    return pd.DataFrame(rows), labels


def distinctive_terms(profiles: csr_matrix, terms: np.ndarray, labels: np.ndarray) -> dict[int, list[tuple[str, float]]]:
    """Per cluster: words whose average TF-IDF weight is highest relative to all regular customers."""
    overall = np.asarray(profiles.mean(axis=0)).ravel()
    out = {}
    for c in np.unique(labels):
        mean_c = np.asarray(profiles[labels == c].mean(axis=0)).ravel()
        score = mean_c - overall
        top = np.argsort(-score)[:TOP_TERMS]
        out[int(c)] = [(terms[t], float(mean_c[t] / max(overall[t], 1e-9))) for t in top]   # (term, times the average)
    return out


def describe_clusters(regulars: pd.DataFrame, shares: pd.DataFrame, labels: np.ndarray,
                      names: dict[int, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = shares.mean()
    by_cluster = shares.groupby(labels).mean()
    lift = by_cluster.div(overall, axis=1)
    rows = []
    for c in by_cluster.index:
        s, l = by_cluster.loc[c], lift.loc[c]
        part = regulars[labels == c]
        focus = l[(s >= 0.05) & (l >= 1.25)].sort_values(ascending=False).head(3)
        rows.append(dict(
            cluster=int(c), name=names[int(c)], customers=len(part), share=len(part) / len(regulars),
            over_represented_groups="; ".join(f"{k} x{v:.1f}" for k, v in focus.items()),
            top_groups="; ".join(f"{k} {v:.0%}" for k, v in s.drop(DEPARTMENTS).sort_values(ascending=False).head(3).items()),
            womenswear=s["Womenswear dept."], menswear=s["Menswear dept."], kids=s["Kids dept."],
            mean_purchases=part.n_purchases.mean(), mean_spend=part.spend.mean(), mean_price=part.mean_price.mean(),
            online_share=part.online_share.mean(), mean_categories=part.n_categories.mean(), median_age=part.age.median()))
    profiles = pd.DataFrame(rows)
    matrix = by_cluster.T.copy()
    matrix.columns = [f"cluster_{c}" for c in matrix.columns]
    matrix.insert(0, "overall_share", overall)
    matrix.index.name = "garment_group"
    return profiles, matrix.reset_index()


def category_names(shares: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    overall = shares.mean()
    names = {}
    for c in np.unique(labels):
        s = shares[labels == c].mean()
        focus = (s / overall)[(s >= 0.05) & (s / overall >= 1.25)].sort_values(ascending=False).head(2)
        names[int(c)] = " + ".join(focus.index) if len(focus) else f"{s.idxmax()} (average mix)"
    return names


# ----------------------------------------------------------------------------- figures
def figures(k_table, profiles, matrix, X, labels, tiers, term_table) -> None:
    viz.style()
    k_final = len(profiles)
    names = profiles.set_index("cluster")["name"]
    colour = {c: viz.CAT[c % len(viz.CAT)] for c in names.index}

    # activity tiers: customers and their share of purchases
    fig, ax = viz.plt.subplots(figsize=(viz.FULL * 0.8, 1.9))
    y = np.arange(len(tiers))
    ax.barh(y, tiers.customers, color=viz.SEQ[3], height=0.6)
    ax.set_xlim(0, tiers.customers.max() * 1.55)
    viz.bar_labels(ax, y, tiers.customers, [f"{n:,} customers, {s:.0%} of purchases" for n, s in zip(tiers.customers, tiers.purchase_share)])
    ax.set_yticks(y); ax.set_yticklabels(tiers.tier); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    viz.titled(ax, "A quarter of the customers make two thirds of the purchases", "Customers active in the last 365 days, by number of purchases")
    viz.save(fig, "rq1_activity_tiers")

    # choice of K: silhouette, with the chosen K marked
    fig, ax = viz.plt.subplots(figsize=(viz.FULL * 0.62, 2.1))
    ax.plot(k_table.k, k_table.silhouette, color=viz.CONTEXT_DARK, linewidth=1.4, zorder=2)
    ax.scatter(k_table.k, k_table.silhouette, s=22, color=viz.CONTEXT_DARK, edgecolor=viz.SURFACE, linewidth=1.2, zorder=3)
    chosen = k_table[k_table.k == k_final].iloc[0]
    ax.scatter([k_final], [chosen.silhouette], s=46, color=viz.HIGHLIGHT, edgecolor=viz.SURFACE, linewidth=1.4, zorder=4)
    ax.annotate(f"chosen K = {k_final}", (k_final, chosen.silhouette), xytext=(0, -14), textcoords="offset points",
                ha="center", fontsize=7, color=viz.TEXT)
    ax.set_xticks(k_table.k); ax.set_xlabel("number of groups K"); ax.set_ylabel("silhouette")
    ax.set_ylim(k_table.silhouette.min() - 0.012, k_table.silhouette.max() + 0.006); ax.grid(axis="x", visible=False)
    viz.titled(ax, "Silhouette is low for every K", "5,000-customer sample of regular customers")
    viz.save(fig, "rq1_k_selection")

    # distinctive words per cluster (style features only)
    if term_table is not None:
        clusters = sorted(term_table.cluster.unique())
        ncol = min(3, len(clusters)); nrow = int(np.ceil(len(clusters) / ncol))
        fig, axes = viz.plt.subplots(nrow, ncol, figsize=(viz.FULL, 1.75 * nrow + 0.2), squeeze=False,
                                     gridspec_kw=dict(wspace=0.95, hspace=0.75))
        xmax = term_table.times_average.max() * 1.05
        for ax, c in zip(axes.ravel(), clusters):
            part = term_table[term_table.cluster == c].sort_values("times_average", ascending=False).head(6)
            yy = np.arange(len(part))
            ax.barh(yy, part.times_average, color=colour[c], height=0.62)
            ax.axvline(1, color=viz.CONTEXT_DARK, linewidth=0.7)
            ax.set_yticks(yy); ax.set_yticklabels(part.term, fontsize=7); ax.invert_yaxis()
            ax.set_xlim(0, xmax); ax.grid(axis="y", visible=False); ax.tick_params(axis="x", labelsize=6.6)
            p = profiles.set_index("cluster").loc[c]
            ax.set_title(f"C{c}  {p.share:.0%} of regulars", fontsize=7.8)
        for ax in axes.ravel()[len(clusters):]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("times the average weight", fontsize=7)
        viz.save(fig, "rq1_style_terms")

    # garment-group share by cluster
    garments = matrix[~matrix.garment_group.str.endswith("dept.")]
    order = garments.sort_values("overall_share", ascending=False).garment_group
    cols = [c for c in matrix.columns if c.startswith("cluster_")]
    heat = garments.set_index("garment_group").loc[order, cols]
    fig, ax = viz.plt.subplots(figsize=(viz.FULL * 0.8, 4.9))
    im = ax.imshow(heat.values, cmap=viz.SEQ_CMAP, aspect="auto", vmin=0, vmax=0.45)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"C{int(c.split('_')[1])}\n" + "\n".join(textwrap.wrap(names.loc[int(c.split('_')[1])], 12)) for c in cols], fontsize=6.6)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6.8); ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6.2, color="white" if v > 0.24 else viz.TEXT2)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.4)
    cb.ax.yaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    viz.save(fig, "rq1_cluster_heatmap")

    # 2-D map of a customer sample
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), min(8000, len(X)), replace=False)
    try:
        import umap  # optional: nicer separation than PCA
        xy = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED).fit_transform(X[idx]); method = "UMAP"
    except Exception:
        xy = PCA(2, random_state=SEED).fit_transform(X[idx]); method = "PCA"
    fig, ax = viz.plt.subplots(figsize=(viz.FULL * 0.72, 3.5))
    for c in sorted(np.unique(labels)):
        m = labels[idx] == c
        ax.scatter(xy[m, 0], xy[m, 1], s=2.5, alpha=0.55, color=colour[c], edgecolors="none", rasterized=True)
        ax.scatter([], [], s=18, color=colour[c], label=f"C{c}  {names.loc[c]}")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handletextpad=0.3)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    viz.titled(ax, f"{method} projection of customer taste clusters (K-means, K = {len(np.unique(labels))})", "8,000 regular customers; each dot is one customer")
    viz.save(fig, "rq1_map")


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod", type=int, default=5)
    parser.add_argument("--features", choices=["style", "categories"], default="style")
    parser.add_argument("--window-days", type=int, default=365)
    parser.add_argument("--min-purchases", type=int, default=5)
    parser.add_argument("--ks", default="2,3,4,5,6,7,8")
    parser.add_argument("--k-final", type=int, default=None, help="override the automatic K choice")
    args = parser.parse_args()

    con = connect()
    base_views(con)
    sample_events(con, args.mod)

    activity, shares = customer_features(con, args.window_days, DATA_END)
    print(f"{len(activity):,} customers bought something in the last {args.window_days} days")
    tiers = activity_tiers(activity)
    save_table(tiers, "rq1_activity_tiers")
    print(tiers.round(3).to_string(index=False))

    regular = activity.n_purchases >= args.min_purchases
    regulars, reg_shares = activity[regular], shares[regular]
    print(f"{len(regulars):,} regular customers (>= {args.min_purchases} purchases) go into K-means, features = {args.features}")
    if args.features == "style":
        X, profiles_sparse, terms = style_vectors(con, regulars.index)
    else:
        X, profiles_sparse, terms = np.sqrt(reg_shares.to_numpy(dtype=np.float32)), None, None

    ks = [int(k) for k in args.ks.split(",")]
    k_table, labels_by_k = choose_k(X, ks, sample_n=5000)
    save_table(k_table, "rq1_k_selection")
    eligible = k_table[k_table.smallest_cluster_share >= 0.03]
    k = args.k_final or int(eligible.sort_values("silhouette", ascending=False).iloc[0].k)
    labels = labels_by_k[k]
    print(f"selected K = {k}")

    stability = []
    for seed in (SEED + 1, SEED + 2):
        other = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(X)
        stability.append(dict(k=k, seed=seed, adjusted_rand_vs_primary=adjusted_rand_score(labels, other)))
    save_table(pd.DataFrame(stability), "rq1_seed_stability")
    print(pd.DataFrame(stability).round(3).to_string(index=False))

    term_table = None
    if args.features == "style":
        distinct = distinctive_terms(profiles_sparse, terms, labels)
        names = {c: ", ".join(t for t, _ in v[:3]) for c, v in distinct.items()}
        term_table = pd.DataFrame([dict(cluster=c, rank=i + 1, term=t, times_average=x) for c, v in distinct.items() for i, (t, x) in enumerate(v)])
        save_table(term_table, "rq1_style_terms")
    else:
        names = category_names(reg_shares, labels)

    profiles, matrix = describe_clusters(regulars, reg_shares, labels, names)
    save_table(profiles, "rq1_cluster_profiles")
    save_table(matrix, "rq1_cluster_category_shares")
    save_table(pd.DataFrame(dict(customer_id=regulars.index, cluster=labels)), "rq1_assignments")
    print(profiles[["cluster", "name", "customers", "share", "over_represented_groups", "mean_purchases", "mean_price", "online_share", "median_age"]]
          .round(3).to_string(index=False))
    figures(k_table, profiles, matrix, X, labels, tiers, term_table)
