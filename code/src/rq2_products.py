"""RQ2, product layer: which products should we show each customer?

Two stages, as in production recommenders.

1. Candidate generation. Seven sources propose products for a customer:
     repurchase    products bought before, weighted by count and recency
     colour        other colours of products bought before
     co-purchase   products that share baskets with the customer's products
                   (item co-occurrence from same-customer, same-day baskets; RQ4 at product level)
     similar       products whose text is close to the customer's taste vector (RQ3 TF-IDF)
     cluster       best sellers among customers of the same style cluster (RQ1)
     age           best sellers of the last 7 days in the customer's age group
     popular       best sellers of the last 7 days overall
2. Ranking. A gradient-boosting classifier scores every (customer, candidate) pair. Its
   features come from the other research questions: RQ3 text similarity to the customer's
   taste and to the last purchase, the RQ1 style cluster and its purchase rate of the item,
   RQ4 rule confidence, next-visit transition probability and co-purchase strength, and
   the category layer's probability that the customer buys the candidate's garment group.
   Label: bought in the 7 days after the cutoff. Trained on twelve weekly cutoffs from
   2020-06-17 to 2020-09-02 with negative sampling, tuned on the week of 2020-09-09, tested
   once on the last full week of data (2020-09-16 .. 09-22). Removing one feature group at a
   time measures what each research question contributes.
   The "repeat timing" group comes from the repeat-purchase finding (insight_repeat.py): the same
   article is mostly bought again within a few days, so the ranker gets the number of days since
   the customer bought this article, how often, and the days since any variant of its style.
   The best-seller sources and the "recent sales" group come from the first-time-purchase finding
   (insight_first_time.py): articles a customer has not bought before come mostly from last week's
   best sellers, and a 7-day list covers more of them than a 30-day list. The two best-seller
   sources therefore use the last 7 days and 100 products (before: 30 days, 50 products), and the
   ranker gets last week's sales, yesterday's sales and the rank within the garment group.
3. Steps. The script also retrains the progress-report model (30-day lists, neither of the two new
   feature groups) and the model with repeat timing only, on the same split, and counts the top-10
   hits separately for articles bought before, other variants of a style bought before, and styles
   not bought before (table rq2_model_steps).

Usage:
    python src/rq2_products.py --mod 20      (after rq3_text_mining.py and rq2_categories.py)
"""
import argparse
import textwrap
import time

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MultiLabelBinarizer, normalize

from common import CACHE, SEED, base_views, connect, garment_groups, sample_events, save_table
from rq2_categories import (DEPARTMENTS, HOLDOUT_CUTOFF, LABEL_DAYS, MIN_REGULAR, PROBA_FROM, VALIDATION_CUTOFF,
                            TRAIN_CUTOFFS as CATEGORY_CUTOFFS, build_frame, mine_rules, rule_features, season_features)
import viz

TRAIN_CUTOFFS = [c for c in CATEGORY_CUTOFFS if c >= PROBA_FROM]   # weekly, the category layer saves probabilities for these
TOP_N = 10                  # products shown
LIST_LEN = 12               # list length scored by MAP@12
MAX_PER_PRODUCT = 2         # colours of one product allowed in the list
NEGATIVES = 20              # sampled negatives per customer and training cutoff
NEIGHBOURS = 30             # co-purchase neighbours kept per product
PER_SOURCE = dict(repurchase=30, colour=30, copurchase=30, similar=30, cluster=50, age=100, popular=100)
SOURCES = list(PER_SOURCE)
BEST_SELLER_DAYS = 7        # the overall and age-group best-seller lists count the sales of the last 7 days
EARLIER_LISTS = dict(days=30, n=50)   # best-seller lists of the progress-report model, kept to measure the change
SVD_DIMS = 50               # taste-vector dimensions, as in rq1_clustering
AGE_EDGES = [25, 35, 45, 55]        # age groups: <25, 25-34, 35-44, 45-54, 55+
MAX_GAP_DAYS = 90           # a "next visit" must follow within this many days, as in rq4_association

# feature groups: "base" is always in; the other four are removed one at a time in the ablation.
# style clusters come from rq1_clustering, text similarity from rq3_text_mining, purchase patterns from rq4_association.
FEATURE_GROUPS = {
    "base": ["s_repurchase", "s_colour", "s_age", "s_popular", "n_sources",
             "item_group", "item_index", "item_price", "item_age_days", "item_sold30_log", "item_trend",
             "u_n365_log", "u_n90_log", "u_recency", "u_online", "u_age", "u_novelty",
             "u_dept_women", "u_dept_men", "u_dept_kids", "season_lift"],
    "style clusters": ["cluster", "s_cluster"],
    "text similarity": ["cos_taste", "cos_last", "s_similar"],
    "purchase patterns": ["rule_conf", "trans_prob", "trans_lift", "s_copurchase"],
    "category layer": ["cat_proba", "u_p365_group", "u_p90_group"],
    # from the repeat-purchase finding (insight_repeat.py): the same article is mostly bought again within days,
    # so the ranker is told when and how often the customer bought this article or another variant of its style
    "repeat timing": ["d_item", "n_item", "d_code"],
    # from the first-time-purchase finding (insight_first_time.py): articles a customer has not bought before
    # come mostly from last week's best sellers, so the ranker sees how the article sold in the last 7 days and yesterday
    "recent sales": ["item_sold7_log", "item_sold1_log", "item_rank7_group"],
}
# the two changes that followed the findings, added one at a time: (name, best-seller lists, feature groups left out)
MODEL_STEPS = [("progress-report model", "earlier", ["repeat timing", "recent sales"]),
               ("+ repeat timing", "earlier", ["recent sales"]),
               ("+ last week's best sellers", "current", [])]
HISTORY_CLASSES = {"same_article": "Same article bought before", "other_variant": "Other variant of a style bought before",
                   "new_style": "Style not bought before"}
CATEGORICAL = ["item_group", "item_index", "cluster"]
WEIGHTS_PREVIOUS = dict(repurchase=1.0, copurchase=0.3, similar=0.3, cluster=0.1, popular=0.1)   # the weighted-sum version


# ----------------------------------------------------------------------------- catalogue and per-cutoff context
def load_catalog(con, groups: list[str]) -> pd.DataFrame:
    cat = con.sql("""SELECT article_id, product_code, prod_name, product_type_name, garment_group_name, index_group_name,
                            colour_group_name FROM articles ORDER BY article_id""").df()
    cat["group_idx"] = cat.garment_group_name.map({g: i for i, g in enumerate(groups)}).astype(np.int16)
    cat["index_idx"] = cat.index_group_name.astype("category").cat.codes.astype(np.int16)
    cat["code_idx"] = cat.product_code.astype("category").cat.codes.astype(np.int32)
    return cat


def item_stats(con, cutoff: str, row_of: pd.Index) -> pd.DataFrame:
    """Products sold in the 90 days before the cutoff (all transactions): sales, price, age."""
    df = con.sql(f"""
        SELECT article_id,
               count(*) FILTER (WHERE t_dat >= DATE '{cutoff}' - INTERVAL '90 days') sold90,
               count(*) FILTER (WHERE t_dat >= DATE '{cutoff}' - INTERVAL '30 days') sold30,
               count(*) FILTER (WHERE t_dat >= DATE '{cutoff}' - INTERVAL '7 days')  sold7,
               count(*) FILTER (WHERE t_dat >= DATE '{cutoff}' - INTERVAL '1 day')   sold1,
               avg(price) FILTER (WHERE t_dat >= DATE '{cutoff}' - INTERVAL '90 days') price90,
               date_diff('day', min(t_dat), DATE '{cutoff}') age_days
        FROM transactions_train WHERE t_dat < DATE '{cutoff}' GROUP BY 1 HAVING sold90 > 0""").df()
    df["row"] = row_of.get_indexer(df.article_id)
    return df[df.row >= 0].set_index("row").sort_index()


def top_normalised(items: np.ndarray, values: np.ndarray, n: int) -> dict[int, float]:
    if len(items) == 0:
        return {}
    if len(values) > n:
        keep = np.argpartition(-values, n - 1)[:n]
        items, values = items[keep], values[keep]
    m = values.max()
    return dict(zip(items.tolist(), (values / m).tolist())) if m > 0 else {}


def age_popularity(con, cutoff: str, row_of: pd.Index, days: int, n: int) -> dict[int, dict]:
    """The `n` best sellers of the last `days` days per age group (all transactions)."""
    df = con.sql(f"""
        SELECT TRY_CAST(c.age AS INTEGER) age, t.article_id, count(*) n
        FROM transactions_train t JOIN customers c USING (customer_id)
        WHERE t.t_dat >= DATE '{cutoff}' - INTERVAL '{days} days' AND t.t_dat < DATE '{cutoff}' AND c.age IS NOT NULL AND c.age <> ''
        GROUP BY 1, 2""").df()
    df["bucket"] = np.digitize(df.age.to_numpy(dtype=float), AGE_EDGES)
    df["row"] = row_of.get_indexer(df.article_id)
    agg = df[df.row >= 0].groupby(["bucket", "row"]).n.sum().reset_index()
    return {int(b): top_normalised(g.row.to_numpy(), g.n.to_numpy(dtype=np.float32), n) for b, g in agg.groupby("bucket")}


def copurchase_neighbours(con, cutoff: str, active_rows: np.ndarray, row_of: pd.Index) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Top co-purchase neighbours per product from same-customer, same-day baskets (all transactions, 90 days)."""
    baskets = con.sql(f"""
        SELECT list(DISTINCT article_id) basket_items FROM transactions_train
        WHERE t_dat >= DATE '{cutoff}' - INTERVAL '90 days' AND t_dat < DATE '{cutoff}'
        GROUP BY customer_id, t_dat HAVING count(DISTINCT article_id) BETWEEN 2 AND 30
        ORDER BY customer_id, t_dat""").df().basket_items
    pos = pd.Series(np.arange(len(active_rows)), index=active_rows)
    b_idx, i_idx = [], []
    for b, items in enumerate(baskets):
        rows = pos.reindex(row_of.get_indexer(items)).dropna()
        b_idx.extend([b] * len(rows)); i_idx.extend(rows.astype(int).tolist())
    B = sparse.csr_matrix((np.ones(len(b_idx), dtype=np.float32), (b_idx, i_idx)), shape=(len(baskets), len(active_rows)))
    C = (B.T @ B).tocsr()
    C.setdiag(0); C.eliminate_zeros()
    deg = np.sqrt(np.asarray(B.sum(axis=0)).ravel()) + 1e-6
    C = (sparse.diags(1 / deg) @ C @ sparse.diags(1 / deg)).tocsr()      # cosine-normalised co-occurrence
    out = {}
    for i in range(C.shape[0]):
        s, e = C.indptr[i], C.indptr[i + 1]
        if e == s:
            continue
        idx, val = C.indices[s:e], C.data[s:e]
        if len(val) > NEIGHBOURS:
            keep = np.argpartition(-val, NEIGHBOURS - 1)[:NEIGHBOURS]
            idx, val = idx[keep], val[keep]
        out[int(active_rows[i])] = (active_rows[idx], (val / val.max()).astype(np.float32))
    print(f"    co-purchase: {len(baskets):,} baskets, {len(out):,} products with neighbours")
    return out


def transition_matrix(con, cutoff: str, groups: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """P(group B on the next visit | group A on this visit), from visits before the cutoff (RQ4)."""
    visits = con.sql(f"""
        WITH v AS (SELECT customer_id, t_dat, list(DISTINCT garment_group_name) cats FROM events
                   WHERE t_dat < DATE '{cutoff}' GROUP BY 1, 2)
        SELECT cats, lead(cats) OVER w AS next_cats, date_diff('day', t_dat, lead(t_dat) OVER w) AS gap_days
        FROM v WINDOW w AS (PARTITION BY customer_id ORDER BY t_dat) QUALIFY next_cats IS NOT NULL""").df()
    pairs = visits[visits.gap_days <= MAX_GAP_DAYS]
    mlb = MultiLabelBinarizer(classes=groups)
    V1 = mlb.fit_transform(pairs.cats).astype(np.float32)
    V2 = mlb.transform(pairs.next_cats).astype(np.float32)
    P = (V1.T @ V2) / np.maximum(V1.sum(axis=0)[:, None], 1)
    return P.astype(np.float32), V2.mean(axis=0).astype(np.float32)


class StyleModel:
    """RQ1 style clusters: SVD + K-means on taste vectors, fitted once and applied at every cutoff."""

    def __init__(self, k: int):
        self.k, self.svd, self.kmeans = k, None, None

    def fit(self, profiles, regular: np.ndarray) -> "StyleModel":
        self.svd = TruncatedSVD(SVD_DIMS, random_state=SEED).fit(profiles[regular])
        self.kmeans = KMeans(n_clusters=self.k, n_init=5, random_state=SEED).fit(normalize(self.svd.transform(profiles[regular])))
        return self

    def assign(self, profiles, regular: np.ndarray) -> np.ndarray:
        labels = np.full(profiles.shape[0], self.k, dtype=np.int16)         # occasional buyers: own group
        if regular.any():
            labels[regular] = self.kmeans.predict(normalize(self.svd.transform(profiles[regular])))
        return labels


def prepare_cutoff(con, cutoff: str, cat: pd.DataFrame, groups: list[str], tfidf, style: StyleModel, fit_style: bool) -> dict:
    """Everything the candidate generator and the feature builder need at one cutoff."""
    row_of = pd.Index(cat.article_id)
    frame = build_frame(con, cutoff, groups)                       # all customers with a purchase in the last 365 days
    n_all = len(frame)
    users = frame.customer_id
    con.register("users", users.to_frame())                   # replaces the previous cutoff's table

    # purchase history before the cutoff (all time), per customer and article
    hist = con.sql(f"""
        SELECT customer_id, article_id, count(*) n,
               count(*) FILTER (WHERE t_dat >= DATE '{cutoff}' - INTERVAL '365 days') n365,
               date_diff('day', max(t_dat), DATE '{cutoff}') age_days
        FROM events JOIN users USING (customer_id) WHERE t_dat < DATE '{cutoff}' GROUP BY 1, 2""").df()
    hist["u"] = pd.Index(users).get_indexer(hist.customer_id)
    hist["row"] = row_of.get_indexer(hist.article_id)
    hist = hist[hist.row >= 0]
    hist["w"] = (hist.n * np.exp(-hist.age_days / 60)).astype(np.float32)

    # taste profile: TF-IDF rows of the products bought in the last year, weighted by log count (RQ3 -> RQ1)
    recent = hist[hist.n365 > 0]
    H = sparse.csr_matrix((np.log1p(recent.n365.to_numpy(dtype=np.float32)), (recent.u, recent.row)), shape=(n_all, tfidf.shape[0]))
    profiles = normalize(H @ tfidf).tocsr()
    regular = frame.n365.to_numpy() >= MIN_REGULAR
    if fit_style:
        style.fit(profiles, regular)
    frame["cluster"] = style.assign(profiles, regular)

    # last visit: one article (for text similarity) and the garment groups bought (for transitions)
    last = con.sql(f"""
        SELECT customer_id, min(article_id) last_article, list(DISTINCT garment_group_name) last_groups
        FROM (SELECT * FROM events JOIN users USING (customer_id) WHERE t_dat < DATE '{cutoff}'
              QUALIFY t_dat = max(t_dat) OVER (PARTITION BY customer_id))
        GROUP BY 1""").df().set_index("customer_id").reindex(users)
    last_row = row_of.get_indexer(last.last_article.fillna(""))
    last_groups = MultiLabelBinarizer(classes=groups).fit_transform(last.last_groups.map(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])).astype(bool)

    # catalogue side: active products, popularity, age-group and cluster best sellers, co-purchase, colour variants
    stats = item_stats(con, cutoff, row_of)
    active_rows = stats.index.to_numpy()
    stats["rank7_group"] = stats.groupby(cat.group_idx.to_numpy()[active_rows]).sold7.rank(method="first", ascending=False)   # 1 = best seller of its garment group last week
    popular = top_normalised(active_rows, stats.sold7.to_numpy(dtype=np.float32), PER_SOURCE["popular"])
    by_age = age_popularity(con, cutoff, row_of, BEST_SELLER_DAYS, PER_SOURCE["age"])
    earlier = dict(popular=top_normalised(active_rows, stats.sold30.to_numpy(dtype=np.float32), EARLIER_LISTS["n"]),
                   by_age=age_popularity(con, cutoff, row_of, EARLIER_LISTS["days"], EARLIER_LISTS["n"]))
    neighbours = copurchase_neighbours(con, cutoff, active_rows, row_of)
    con.register("user_clusters", frame[["customer_id", "cluster"]])
    cp = con.sql(f"""
        SELECT c.cluster, e.article_id, count(*) n FROM events e JOIN user_clusters c USING (customer_id)
        WHERE e.t_dat >= DATE '{cutoff}' - INTERVAL '30 days' AND e.t_dat < DATE '{cutoff}' GROUP BY 1, 2 ORDER BY 1, 2""").df()
    cp["row"] = row_of.get_indexer(cp.article_id)
    cp = cp[cp.row.isin(active_rows)]
    by_cluster = {int(c): top_normalised(g.row.to_numpy(), g.n.to_numpy(dtype=np.float32), PER_SOURCE["cluster"]) for c, g in cp.groupby("cluster")}
    active_cat = cat.iloc[active_rows]
    code_rows = {code: g.index.to_numpy() for code, g in active_cat.groupby("code_idx")}

    # customers who buy in the label week are the ones we score
    lab = np.flatnonzero(frame.targets.notna().to_numpy())
    frame_lab = frame.iloc[lab].reset_index(drop=True)
    truth_df = con.sql(f"""
        SELECT customer_id, list(DISTINCT article_id) bought FROM events JOIN users USING (customer_id)
        WHERE t_dat >= DATE '{cutoff}' AND t_dat < DATE '{cutoff}' + INTERVAL '{LABEL_DAYS} days' GROUP BY 1""").df().set_index("customer_id").bought
    truth = {}
    for i, cid in enumerate(frame_lab.customer_id):
        rows = row_of.get_indexer(truth_df.get(cid, []))
        truth[i] = set(rows[rows >= 0].tolist())

    # RQ4 signals and the category layer, as customer x group matrices
    rules = mine_rules(con, cutoff, groups)
    rule_mat = rule_features(rules, frame_lab, groups)
    P, base = transition_matrix(con, cutoff, groups)
    lg = last_groups[lab]
    trans = np.where(lg[:, :, None], P[None, :, :], 0).max(axis=1).astype(np.float32)   # max over groups of the last visit
    trans_lift = trans / np.maximum(base[None, :], 1e-6)
    season = season_features(con, cutoff, groups)
    proba_path = CACHE / f"rq2_category_proba_{cutoff}.parquet"
    if not proba_path.exists():
        raise FileNotFoundError(f"{proba_path} missing: run src/rq2_categories.py --mod first")
    proba = pd.read_parquet(proba_path).set_index("customer_id").reindex(frame_lab.customer_id)[groups].to_numpy(dtype=np.float32)
    proba = np.nan_to_num(proba, nan=0.0)

    print(f"    {n_all:,} customers with history, {len(lab):,} buy in the next {LABEL_DAYS} days; {len(active_rows):,} active products; "
          f"{len(rules)} rules; style clusters {np.bincount(frame.cluster, minlength=style.k + 1).tolist()}")
    return dict(cutoff=cutoff, frame=frame_lab, lab=lab, users=frame_lab.customer_id.to_numpy(), truth=truth, k_style=style.k,
                hist=hist[hist.u.isin(lab)].assign(u=lambda d: pd.Index(lab).get_indexer(d.u)).sort_values(["u", "row"]),
                profiles=profiles[lab], last_row=last_row[lab], active_rows=active_rows, stats=stats,
                popular=popular, by_age=by_age, earlier=earlier, by_cluster=by_cluster, neighbours=neighbours, code_rows=code_rows,
                rule_mat=rule_mat, trans=trans, trans_lift=trans_lift, season=season, proba=proba,
                p365=frame_lab[[f"p365:{g}" for g in groups]].to_numpy(dtype=np.float32),
                p90=frame_lab[[f"p90:{g}" for g in groups]].to_numpy(dtype=np.float32))


# ----------------------------------------------------------------------------- candidates and features
def similar_candidates(profiles, tfidf, active_rows: np.ndarray, n: int) -> list[dict]:
    """Per customer: the n active products whose text is closest to the taste vector (RQ3)."""
    M = tfidf[active_rows]
    out = []
    for start in range(0, profiles.shape[0], 1000):
        S = (profiles[start:start + 1000] @ M.T).toarray()
        for row in S:
            if row.max() <= 0:
                out.append({}); continue
            keep = np.argpartition(-row, n - 1)[:n]
            out.append(dict(zip(active_rows[keep].tolist(), (row[keep] / row[keep].max()).tolist())))
    return out


def build_pairs(ctx: dict, cat: pd.DataFrame, tfidf, sample: bool, rng: np.random.Generator, lists: str = "current") -> pd.DataFrame:
    """One row per (customer, candidate) with the source scores and all ranking features.

    `lists` chooses the best-seller sources: "current" (last 7 days) or "earlier" (the 30-day lists of the progress report)."""
    popular, by_age = (ctx["popular"], ctx["by_age"]) if lists == "current" else (ctx["earlier"]["popular"], ctx["earlier"]["by_age"])
    frame, truth = ctx["frame"], ctx["truth"]
    n_users = len(frame)
    hist = ctx["hist"]                                   # sorted by (u, row): ties break the same way every run
    starts = np.searchsorted(hist.u.to_numpy(), np.arange(n_users + 1))
    h_rows, h_w = hist.row.to_numpy(), hist.w.to_numpy()
    code_of = cat.code_idx.to_numpy()
    sims = similar_candidates(ctx["profiles"], tfidf, ctx["active_rows"], PER_SOURCE["similar"])
    age_bucket = np.where(frame.age.notna(), np.digitize(frame.age.fillna(0).to_numpy(dtype=float), AGE_EDGES), -1)
    clusters = frame.cluster.to_numpy()
    profiles, last_row = ctx["profiles"], ctx["last_row"]

    cols = {k: [] for k in ["u", "item", "label"] + [f"s_{s}" for s in SOURCES] + ["cos_taste", "cos_last", "anchor_colour", "anchor_cop"]}
    for i in range(n_users):
        s, e = starts[i], starts[i + 1]
        rows_i, w_i = h_rows[s:e], h_w[s:e]
        bought = set(rows_i.tolist())
        rep = top_normalised(rows_i, w_i, PER_SOURCE["repurchase"])
        colour, anchor_colour, cop, anchor_cop = {}, {}, {}, {}
        for r, w in rep.items():
            for r2 in ctx["code_rows"].get(code_of[r], ()):
                if r2 not in bought and w > colour.get(r2, 0.0):
                    colour[r2], anchor_colour[r2] = w, r
        for r, w in zip(rows_i.tolist(), w_i.tolist()):
            nb = ctx["neighbours"].get(r)
            if nb is None:
                continue
            for j, v in zip(nb[0].tolist(), nb[1].tolist()):
                if j in bought:
                    continue
                cop[j] = cop.get(j, 0.0) + w * v
                if w * v > anchor_cop.get(j, (0.0, -1))[0]:
                    anchor_cop[j] = (w * v, r)
        if colour:
            colour = dict(sorted(colour.items(), key=lambda kv: -kv[1])[:PER_SOURCE["colour"]])
        if cop:
            cop = dict(sorted(cop.items(), key=lambda kv: -kv[1])[:PER_SOURCE["copurchase"]])
            m = max(cop.values()); cop = {k: v / m for k, v in cop.items()}
        sources = (rep, colour, cop, sims[i], ctx["by_cluster"].get(int(clusters[i]), {}) if clusters[i] < ctx["k_style"] else {},
                   by_age.get(int(age_bucket[i]), {}), popular)
        items = np.array(sorted(set().union(*[d.keys() for d in sources])), dtype=np.int64)
        if len(items) == 0:
            continue
        label = np.isin(items, list(truth[i])) if truth[i] else np.zeros(len(items), dtype=bool)
        if sample:
            neg = np.flatnonzero(~label)
            keep = np.concatenate([np.flatnonzero(label), rng.choice(neg, min(NEGATIVES, len(neg)), replace=False)])
            items, label = items[keep], label[keep]
        scores = np.zeros((len(items), len(SOURCES)), dtype=np.float32)
        for k, d in enumerate(sources):
            if d:
                scores[:, k] = [d.get(int(it), 0.0) for it in items]
        T = tfidf[items]
        cos_taste = np.asarray((profiles[i] @ T.T).todense()).ravel().astype(np.float32)
        cos_last = (np.asarray((tfidf[last_row[i]] @ T.T).todense()).ravel().astype(np.float32)
                    if last_row[i] >= 0 else np.full(len(items), np.nan, dtype=np.float32))
        cols["u"].append(np.full(len(items), i, dtype=np.int32)); cols["item"].append(items.astype(np.int32)); cols["label"].append(label)
        for k, src in enumerate(SOURCES):
            cols[f"s_{src}"].append(scores[:, k])
        cols["cos_taste"].append(cos_taste); cols["cos_last"].append(cos_last)
        cols["anchor_colour"].append(np.array([anchor_colour.get(int(it), -1) for it in items], dtype=np.int32))
        cols["anchor_cop"].append(np.array([anchor_cop.get(int(it), (0, -1))[1] for it in items], dtype=np.int32))
    pairs = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})

    # vectorised features: item, customer, and group-dependent lookups
    item, u = pairs.item.to_numpy(), pairs.u.to_numpy()
    g = cat.group_idx.to_numpy()[item]
    st = ctx["stats"].reindex(item)
    pairs["n_sources"] = (pairs[[f"s_{s}" for s in SOURCES]].to_numpy() > 0).sum(axis=1).astype(np.int8)
    pairs["item_group"] = g.astype(np.int16)
    pairs["item_index"] = cat.index_idx.to_numpy()[item]
    pairs["item_price"] = st.price90.to_numpy(dtype=np.float32)
    pairs["item_age_days"] = st.age_days.to_numpy(dtype=np.float32)
    pairs["item_sold30_log"] = np.log1p(st.sold30.fillna(0).to_numpy(dtype=np.float32))
    pairs["item_trend"] = (st.sold7.fillna(0).to_numpy() * 30 / 7) / (st.sold30.fillna(0).to_numpy() + 1)
    # recent sales: last 7 days, the day before the cutoff, and the rank within the garment group last week
    pairs["item_sold7_log"] = np.log1p(st.sold7.fillna(0).to_numpy(dtype=np.float32))
    pairs["item_sold1_log"] = np.log1p(st.sold1.fillna(0).to_numpy(dtype=np.float32))
    pairs["item_rank7_group"] = st.rank7_group.to_numpy(dtype=np.float32)
    f = frame.iloc[u]
    pairs["u_n365_log"] = np.log1p(f.n365.to_numpy(dtype=np.float32))
    pairs["u_n90_log"] = np.log1p(f.n90.to_numpy(dtype=np.float32))
    pairs["u_recency"] = f.recency_days.to_numpy(dtype=np.float32)
    pairs["u_online"] = f.online_share.to_numpy(dtype=np.float32)
    pairs["u_age"] = f.age.to_numpy(dtype=np.float32)
    pairs["u_novelty"] = f.novelty.to_numpy(dtype=np.float32)
    for name, dept in zip(("women", "men", "kids"), DEPARTMENTS):
        pairs[f"u_dept_{name}"] = f[f"dept:{dept}"].to_numpy(dtype=np.float32)
    pairs["cluster"] = f.cluster.to_numpy(dtype=np.int16)
    pairs["season_lift"] = ctx["season"][g]
    pairs["rule_conf"] = ctx["rule_mat"][u, g]
    pairs["trans_prob"] = ctx["trans"][u, g]
    pairs["trans_lift"] = ctx["trans_lift"][u, g]
    pairs["cat_proba"] = ctx["proba"][u, g]
    pairs["u_p365_group"] = ctx["p365"][u, g]
    pairs["u_p90_group"] = ctx["p90"][u, g]
    # repeat timing: days since the customer last bought this article, how often, and days since any variant of its style
    bought_key = pd.MultiIndex.from_arrays([hist.u.to_numpy(), hist.row.to_numpy()])
    pair_key = pd.MultiIndex.from_arrays([u, item])
    pairs["d_item"] = pd.Series(hist.age_days.to_numpy(dtype=np.float32), index=bought_key).reindex(pair_key).to_numpy()   # NaN: never bought
    pairs["n_item"] = pd.Series(hist.n.to_numpy(dtype=np.float32), index=bought_key).reindex(pair_key).fillna(0).to_numpy()
    style_age = hist.assign(code=code_of[hist.row.to_numpy()]).groupby(["u", "code"]).age_days.min()
    pairs["d_code"] = style_age.reindex(pd.MultiIndex.from_arrays([u, code_of[item]])).to_numpy(dtype=np.float32)
    pairs["cutoff"] = ctx["cutoff"]
    return pairs


# ----------------------------------------------------------------------------- ranking and evaluation
def recommend(pairs: pd.DataFrame, scores: np.ndarray, code_of: np.ndarray) -> dict[int, list[int]]:
    """LIST_LEN products per customer, at most MAX_PER_PRODUCT colours of one product."""
    u, items = pairs.u.to_numpy(), pairs.item.to_numpy()
    order = np.lexsort((-scores, u))
    u, items = u[order], items[order]
    bounds = np.flatnonzero(np.diff(u)) + 1
    out = {}
    for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(u)]):
        recs, per_code = [], {}
        for it in items[s:e]:
            c = code_of[it]
            if per_code.get(c, 0) >= MAX_PER_PRODUCT:
                continue
            per_code[c] = per_code.get(c, 0) + 1
            recs.append(int(it))
            if len(recs) == LIST_LEN:
                break
        out[int(u[s])] = recs
    return out


def metrics(recs: dict[int, list[int]], truth: dict[int, set], users: np.ndarray | None = None) -> dict:
    users = range(len(truth)) if users is None else users
    rec10, hit10, ap = [], [], []
    for i in users:
        t, r = truth[i], recs.get(i, [])
        if not t:
            continue
        hits = len(set(r[:TOP_N]) & t)
        rec10.append(hits / len(t)); hit10.append(float(hits > 0))
        n, s = 0, 0.0
        for rank, it in enumerate(r[:LIST_LEN], 1):
            if it in t:
                n += 1; s += n / rank
        ap.append(s / min(len(t), LIST_LEN))
    return dict(customers=len(rec10), recall10=float(np.mean(rec10)), hit10=float(np.mean(hit10)), map12=float(np.mean(ap)))


def hits_by_history(recs: dict[int, list[int]], ctx: dict, code_of: np.ndarray) -> dict:
    """Articles bought in the label week and top-N hits, split by what the customer had bought before the cutoff:
    the same article, another variant (article number) of a style bought before, or a style not bought before."""
    bought_before = ctx["hist"].groupby("u").row.agg(set).to_dict()
    kinds = list(HISTORY_CLASSES)
    bought, hits = dict.fromkeys(kinds, 0), dict.fromkeys(kinds, 0)
    for i, t in ctx["truth"].items():
        if not t:
            continue
        b = bought_before.get(i, set())
        styles = {code_of[r] for r in b}
        top = set(recs.get(i, [])[:TOP_N])
        for it in t:
            kind = "same_article" if it in b else "other_variant" if code_of[it] in styles else "new_style"
            bought[kind] += 1
            hits[kind] += it in top
    return {**{f"bought_{k}": bought[k] for k in kinds}, **{f"hits_{k}": hits[k] for k in kinds}}


def previous_version(pairs: pd.DataFrame, proba: np.ndarray) -> np.ndarray:
    """The weighted-sum recommender of the first version: fixed weights, top-3 category gate on non-repurchase sources."""
    top3 = np.argsort(-proba, axis=1)[:, :3]
    gate_mat = np.zeros_like(proba, dtype=bool)
    gate_mat[np.arange(len(proba))[:, None], top3] = True
    gate = gate_mat[pairs.u.to_numpy(), pairs.item_group.to_numpy()]
    s = WEIGHTS_PREVIOUS["repurchase"] * pairs.s_repurchase.to_numpy()
    for src in ("copurchase", "similar", "cluster", "popular"):
        s = s + gate * WEIGHTS_PREVIOUS[src] * pairs[f"s_{src}"].to_numpy()
    return s.astype(np.float32)


def features_for(groups_in: list[str]) -> list[str]:
    return [c for gname in groups_in for c in FEATURE_GROUPS[gname]]


def fit_ranker(train: pd.DataFrame, cols: list[str]) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200,
                                           l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                           n_iter_no_change=30, random_state=SEED,
                                           categorical_features=[c in CATEGORICAL for c in cols])
    return model.fit(train[cols].to_numpy(dtype=np.float32), train.label.to_numpy())


def score(model, pairs: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = pairs[cols].to_numpy(dtype=np.float32)
    return np.concatenate([model.predict_proba(X[s:s + 500_000])[:, 1] for s in range(0, len(X), 500_000)]).astype(np.float32)


REASON = dict(repurchase="you bought this before", colour="another colour of '{}' you bought",
              copurchase="often bought together with '{}'", similar="similar to '{}' you bought",
              cluster="popular with customers of your style", age="popular in your age group", popular="trending now")


def explain(pairs: pd.DataFrame, idx: np.ndarray, cat: pd.DataFrame, ctx: dict, tfidf) -> list[str]:
    """Reason for the rows `idx` of pairs: the strongest source, with the anchor product where there is one."""
    S = pairs.iloc[idx][[f"s_{s}" for s in SOURCES]].to_numpy()
    best = S.argmax(axis=1)
    names = cat.prod_name.to_numpy()
    hist = ctx["hist"]
    out = []
    for k, (row, b) in enumerate(zip(pairs.iloc[idx].itertuples(), best)):
        src = SOURCES[b]
        if src == "colour" and row.anchor_colour >= 0:
            out.append(REASON[src].format(names[row.anchor_colour]))
        elif src == "copurchase" and row.anchor_cop >= 0:
            out.append(REASON[src].format(names[row.anchor_cop]))
        elif src == "similar":
            h = hist[hist.u == row.u].row.to_numpy()
            sims = np.asarray((tfidf[h] @ tfidf[row.item].T).todense()).ravel() if len(h) else np.zeros(0)
            out.append(REASON[src].format(names[h[sims.argmax()]]) if len(h) else "similar to what you bought")
        else:
            out.append(REASON[src])
    return out


# ----------------------------------------------------------------------------- figures
METHOD_NAMES = {"GlobalPopularity": "Overall best sellers", "RepurchaseOnly": "Repurchase only",
                "WeightedSum (first version)": "Weighted sum (first version)", "Ranker (all RQ features)": "Ranking model"}
VARIANT_NAMES = {"all features": "All features", "without style clusters": "Without style clusters (RQ2)",
                 "without text similarity": "Without text similarity (RQ1)", "without purchase patterns": "Without purchase patterns (RQ3)",
                 "without category layer": "Without category layer", "without repeat timing": "Without repeat timing",
                 "without recent sales": "Without recent sales",
                 "base features only": "Base features only"}
FEATURE_NAMES = {
    "s_repurchase": "Repurchase score", "s_colour": "Colour-variant score", "s_age": "Age-group best-seller score",
    "s_popular": "Best-seller score", "n_sources": "Number of candidate sources", "item_group": "Item garment group",
    "item_index": "Item index group", "item_price": "Item price", "item_age_days": "Days since item first sold",
    "item_sold30_log": "Item sales, last 30 days", "item_trend": "Item sales trend", "u_n365_log": "Customer purchases, last year",
    "u_n90_log": "Customer purchases, last 90 days", "u_recency": "Days since customer's last purchase",
    "u_online": "Customer online share", "u_age": "Customer age", "u_novelty": "Customer's share of new products",
    "u_dept_women": "Customer womenswear share", "u_dept_men": "Customer menswear share", "u_dept_kids": "Customer kidswear share",
    "season_lift": "Seasonal lift of the group", "cluster": "Customer style group", "s_cluster": "Style-group best-seller score",
    "cos_taste": "Text similarity to customer taste", "cos_last": "Text similarity to last purchase", "s_similar": "Similar-product score",
    "rule_conf": "Association-rule confidence", "trans_prob": "Next-visit transition probability",
    "trans_lift": "Next-visit transition lift", "s_copurchase": "Co-purchase score",
    "cat_proba": "Category-layer probability", "u_p365_group": "Customer share of the group, year",
    "u_p90_group": "Customer share of the group, 90 days",
    "d_item": "Days since customer bought this article", "n_item": "Times customer bought this article",
    "d_code": "Days since customer bought this style",
    "item_sold7_log": "Item sales, last 7 days", "item_sold1_log": "Item sales, yesterday",
    "item_rank7_group": "Item rank in its garment group, last 7 days"}
GROUP_NAMES = {"base": "base features", "text similarity": "text similarity (RQ1)", "style clusters": "style clusters (RQ2)",
               "purchase patterns": "purchase patterns (RQ3)", "category layer": "category layer", "repeat timing": "repeat timing", "recent sales": "recent sales"}
STEP_NAMES = {"progress-report model": "Progress-report model", "+ repeat timing": "+ repeat timing",
              "+ last week's best sellers": "+ last week's best sellers"}


def figures(metrics_table: pd.DataFrame, ablation: pd.DataFrame, importance: pd.DataFrame, tiers: pd.DataFrame, steps: pd.DataFrame) -> None:
    viz.style()
    best = "Ranker (all RQ features)"

    # steps: top-10 hits by purchase history for the three versions of the model, validation and test week together
    both = steps.groupby("step", sort=False).sum(numeric_only=True)
    kinds = list(HISTORY_CLASSES)
    fig, axes = viz.plt.subplots(1, len(kinds), figsize=(viz.FULL, 1.9), sharey=True, gridspec_kw=dict(wspace=0.1))
    y = np.arange(len(both))
    total = sum(both[f"bought_{k}"].iloc[0] for k in kinds)
    for ax, kind in zip(axes, kinds):
        hits, bought = both[f"hits_{kind}"].to_numpy(), both[f"bought_{kind}"].iloc[0]
        ax.barh(y, hits, color=[viz.CONTEXT] * (len(both) - 1) + [viz.HIGHLIGHT], height=0.62)
        ax.set_xlim(0, max(hits.max(), 1) * 1.45)
        viz.bar_labels(ax, y, hits, [f"{h:,.0f}  ({h / bought:.1%})" for h in hits])
        ax.grid(axis="y", visible=False); ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
        ax.set_title(f"{textwrap.fill(HISTORY_CLASSES[kind], 24)}\n{bought:,.0f} bought ({bought / total:.1%})", fontsize=7.2)
    axes[0].set_yticks(y); axes[0].set_yticklabels([STEP_NAMES[s] for s in both.index]); axes[0].invert_yaxis()
    fig.text(0.0, 1.27, "Top-10 hits by what the customer had bought before", ha="left", va="bottom",
             fontsize=8.8, fontweight="bold", color=viz.TEXT)
    fig.text(0.0, 1.17, "Validation and test week together; bars: hits in the top 10 (share of the articles bought)", ha="left", va="bottom",
             fontsize=7.2, color=viz.TEXT2)
    viz.save(fig, "rq2_model_steps")

    # methods: two panels, one per metric, the ranking model highlighted
    hold = metrics_table[metrics_table.split == "holdout"].reset_index(drop=True)
    y = np.arange(len(hold))
    colours = [viz.HIGHLIGHT if m == best else viz.CONTEXT for m in hold.method]
    fig, axes = viz.plt.subplots(1, 2, figsize=(viz.FULL, 1.75), sharey=True, gridspec_kw=dict(wspace=0.12))
    for ax, col, fmt, label in ((axes[0], "recall10", "{:.1%}", "Recall@10"), (axes[1], "map12", "{:.4f}", "MAP@12")):
        ax.barh(y, hold[col], color=colours, height=0.62)
        ax.set_xlim(0, hold[col].max() * 1.3)
        viz.bar_labels(ax, y, hold[col], [fmt.format(v) for v in hold[col]])
        ax.grid(axis="y", visible=False); ax.set_xticks([])
        for spine in ("bottom",):
            ax.spines[spine].set_visible(False)
        ax.set_title(label, fontsize=8)
    axes[0].set_yticks(y); axes[0].set_yticklabels([METHOD_NAMES[m] for m in hold.method]); axes[0].invert_yaxis()
    fig.text(0.0, 1.12, "The ranking model beats every baseline on both measures", ha="left", va="bottom",
             fontsize=8.8, fontweight="bold", color=viz.TEXT)
    fig.text(0.0, 1.02, f"Product layer, test week from {HOLDOUT_CUTOFF}, 3,031 customers", ha="left", va="bottom",
             fontsize=7.2, color=viz.TEXT2)
    viz.save(fig, "rq2_products")

    # ablation: MAP@12 in both weeks as dots (the axis does not start at zero, so no bars)
    order = list(VARIANT_NAMES)
    fig, ax = viz.plt.subplots(figsize=(viz.FULL, 2.35))
    yy = np.arange(len(order))
    val = ablation[ablation.split == "validation"].set_index("variant").loc[order].map12.to_numpy()
    hold = ablation[ablation.split == "holdout"].set_index("variant").loc[order].map12.to_numpy()
    for i in yy:
        ax.plot([val[i], hold[i]], [i, i], color=viz.GRID, linewidth=2.4, solid_capstyle="round", zorder=1)
    ax.scatter(val, yy, s=30, color=viz.SEQ[3], edgecolor=viz.SURFACE, linewidth=1, zorder=2, label="validation week")
    ax.scatter(hold, yy, s=30, color=viz.HIGHLIGHT, edgecolor=viz.SURFACE, linewidth=1, zorder=3, label="test week")
    for i in yy:
        lo, hi = sorted((val[i], hold[i]))
        ax.annotate(f"{lo:.4f}", (lo, i), xytext=(-6, 0), textcoords="offset points", ha="right", va="center", fontsize=6.4, color=viz.TEXT2)
        ax.annotate(f"{hi:.4f}", (hi, i), xytext=(6, 0), textcoords="offset points", ha="left", va="center", fontsize=6.4, color=viz.TEXT2)
    span = max(val.max(), hold.max()) - min(val.min(), hold.min())
    ax.set_xlim(min(val.min(), hold.min()) - 0.28 * span, max(val.max(), hold.max()) + 0.28 * span)
    ax.set_yticks(yy); ax.set_yticklabels([VARIANT_NAMES[v] for v in order]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("MAP@12")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, handletextpad=0.2)
    viz.titled(ax, "Ablation study of the ranking model", "MAP@12 after retraining with one feature group removed")
    viz.save(fig, "rq2_product_ablation")

    # feature importance, coloured by source
    top = importance.sort_values("importance", ascending=False).head(14).iloc[::-1]
    palette = {"base": viz.SEQ[3], "text similarity": viz.CAT[2], "style clusters": viz.CAT[3],
               "purchase patterns": viz.CAT[1], "category layer": viz.CAT[4], "repeat timing": viz.CAT[7], "recent sales": viz.CAT[5]}
    fig, ax = viz.plt.subplots(figsize=(viz.FULL * 0.86, 3.1))
    ax.barh(range(len(top)), top.importance, color=[palette[g] for g in top.group], height=0.62)
    ax.set_yticks(range(len(top))); ax.set_yticklabels([FEATURE_NAMES.get(f, f) for f in top.feature], fontsize=7)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=GROUP_NAMES[g]) for g, c in palette.items()], loc="lower right")
    ax.set_xlabel("permutation importance: drop in average precision when the feature is shuffled")
    viz.titled(ax, "Permutation feature importance of the ranking model", f"14 most important of {len(importance)} features, validation sample")
    viz.save(fig, "rq2_feature_importance")

    # by activity tier
    methods = ["GlobalPopularity", "WeightedSum (first version)", best]
    tier_names = list(dict.fromkeys(tiers.tier))
    fig, ax = viz.plt.subplots(figsize=(viz.FULL, 2.2))
    x = np.arange(len(tier_names)); width = 0.26
    for j, (m, colour) in enumerate(zip(methods, (viz.CONTEXT, viz.CONTEXT_DARK, viz.HIGHLIGHT))):
        part = tiers[tiers.method == m].set_index("tier").loc[tier_names]
        ax.bar(x + (j - 1) * width, part.recall10, width=width * 0.92, color=colour, label=METHOD_NAMES[m])
    n = tiers[tiers.method == best].set_index("tier").loc[tier_names].customers
    ax.set_xticks(x); ax.set_xticklabels([f"{t}\n{c:,} customers" for t, c in zip(tier_names, n)])
    ax.yaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:.0%}")); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Recall@10"); ax.legend(loc="upper right", ncol=3)
    viz.titled(ax, "The ranking model helps every activity tier", "Test week, by purchases in the year before the cutoff")
    viz.save(fig, "rq2_product_tiers")


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod", type=int, default=5)
    parser.add_argument("--k-style", type=int, default=6, help="number of RQ1 style clusters")
    args = parser.parse_args()

    con = connect()
    base_views(con)
    sample_events(con, args.mod)
    groups = garment_groups(con)
    cat = load_catalog(con, groups)
    tfidf = sparse.load_npz(CACHE / "rq3_tfidf_all_fields.npz").tocsr()
    assert tfidf.shape[0] == len(cat), "TF-IDF matrix does not match the article table; rerun rq3_text_mining.py"
    style = StyleModel(args.k_style)
    rng = np.random.default_rng(SEED)

    # 1. candidates and features at every cutoff
    # `pairs` uses the current best-seller lists; `earlier_pairs` uses the 30-day lists of the progress report and
    # keeps the random generator of that version, so the first two steps reproduce the earlier results exactly
    pairs, earlier_pairs, ctxs = {}, {}, {}
    rng_current = np.random.default_rng(SEED + 1)
    for c in TRAIN_CUTOFFS + [VALIDATION_CUTOFF, HOLDOUT_CUTOFF]:
        start = time.perf_counter()
        print(f"{c}:")
        ctxs[c] = prepare_cutoff(con, c, cat, groups, tfidf, style, fit_style=c == TRAIN_CUTOFFS[0])
        earlier_pairs[c] = build_pairs(ctxs[c], cat, tfidf, sample=c in TRAIN_CUTOFFS, rng=rng, lists="earlier")
        pairs[c] = build_pairs(ctxs[c], cat, tfidf, sample=c in TRAIN_CUTOFFS, rng=rng_current)
        print(f"    {len(pairs[c]):,} pairs, {int(pairs[c].label.sum()):,} positives; with the earlier lists {len(earlier_pairs[c]):,} pairs, "
              f"{int(earlier_pairs[c].label.sum()):,} positives ({time.perf_counter() - start:.0f}s)")
    train = pd.concat([pairs[c] for c in TRAIN_CUTOFFS], ignore_index=True)
    evals = {"validation": (pairs[VALIDATION_CUTOFF], ctxs[VALIDATION_CUTOFF]), "holdout": (pairs[HOLDOUT_CUTOFF], ctxs[HOLDOUT_CUTOFF])}
    code_of = cat.code_idx.to_numpy()

    # candidate coverage: how much of what customers bought is reachable, per source
    rows = []
    for split, (p, ctx) in evals.items():
        total = sum(len(t) for t in ctx["truth"].values())
        pos = p[p.label]
        rows.append(dict(split=split, truth_items=total, candidates_per_customer=len(p) / len(ctx["truth"]),
                         any_source=len(pos) / total, **{s: float((pos[f"s_{s}"] > 0).sum() / total) for s in SOURCES}))
    coverage = pd.DataFrame(rows)
    save_table(coverage, "rq2_candidate_coverage")
    print(coverage.round(3).to_string(index=False))

    # 2. baselines and the ranker
    all_groups = list(FEATURE_GROUPS)
    results, recs_by = [], {}

    def record(method, scores_by_split):
        for split, (p, ctx) in evals.items():
            recs = recommend(p, scores_by_split[split], code_of)
            recs_by[(method, split)] = recs
            results.append(dict(method=method, split=split, **metrics(recs, ctx["truth"])))
        v = results[-2]; h = results[-1]
        print(f"{method:28s} val recall@10 {v['recall10']:.4f} map@12 {v['map12']:.4f} | hold recall@10 {h['recall10']:.4f} "
              f"hit@10 {h['hit10']:.4f} map@12 {h['map12']:.4f}")

    record("GlobalPopularity", {s: p.s_popular.to_numpy() for s, (p, _) in evals.items()})
    record("RepurchaseOnly", {s: p.s_repurchase.to_numpy() for s, (p, _) in evals.items()})
    record("WeightedSum (first version)", {s: previous_version(p, ctx["proba"]) for s, (p, ctx) in evals.items()})
    start = time.perf_counter()
    cols_all = features_for(all_groups)
    ranker = fit_ranker(train, cols_all)
    print(f"ranker: {len(train):,} training pairs, {len(cols_all)} features, {ranker.n_iter_} iterations ({time.perf_counter() - start:.0f}s)")
    ranker_scores = {s: score(ranker, p, cols_all) for s, (p, _) in evals.items()}
    record("Ranker (all RQ features)", ranker_scores)
    metrics_table = pd.DataFrame(results)
    save_table(metrics_table, "rq2_product_metrics")

    # 3. ablation: remove one feature group at a time, and the base features alone
    rows = []
    variants = [("all features", all_groups)] + [(f"without {g}", [x for x in all_groups if x != g]) for g in all_groups if g != "base"] \
        + [("base features only", ["base"])]
    for name, use in variants:
        cols = features_for(use)
        model = ranker if name == "all features" else fit_ranker(train, cols)
        for split, (p, ctx) in evals.items():
            sc = ranker_scores[split] if name == "all features" else score(model, p, cols)
            rows.append(dict(variant=name, split=split, features=len(cols), **metrics(recommend(p, sc, code_of), ctx["truth"])))
        print(f"ablation {name:32s} val map@12 {rows[-2]['map12']:.4f}  hold map@12 {rows[-1]['map12']:.4f}  recall@10 {rows[-1]['recall10']:.4f}")
    ablation = pd.DataFrame(rows)
    save_table(ablation, "rq2_product_ablation")

    # 3b. the two changes that followed the findings, one at a time: same split, hits split by purchase history
    rows = []
    for name, lists, left_out in MODEL_STEPS:
        final = lists == "current" and not left_out                  # the last step is the ranker fitted above
        source = pairs if lists == "current" else earlier_pairs
        cols = features_for([g for g in all_groups if g not in left_out])
        model = ranker if final else fit_ranker(pd.concat([source[c] for c in TRAIN_CUTOFFS], ignore_index=True), cols)
        for split, cutoff in (("validation", VALIDATION_CUTOFF), ("holdout", HOLDOUT_CUTOFF)):
            p, ctx = source[cutoff], ctxs[cutoff]
            recs = recs_by[("Ranker (all RQ features)", split)] if final else recommend(p, score(model, p, cols), code_of)
            total = sum(len(t) for t in ctx["truth"].values())
            rows.append(dict(step=name, split=split, features=len(cols), candidates_per_customer=len(p) / len(ctx["truth"]),
                             coverage=float(p.label.sum() / total), **metrics(recs, ctx["truth"]), **hits_by_history(recs, ctx, code_of)))
        print(f"step {name:28s} val map@12 {rows[-2]['map12']:.4f}  hold map@12 {rows[-1]['map12']:.4f}  recall@10 {rows[-1]['recall10']:.4f}  "
              f"hit@10 {rows[-1]['hit10']:.4f}")
    steps = pd.DataFrame(rows)
    save_table(steps, "rq2_model_steps")

    # 4. feature importance on a validation sample
    p_val = evals["validation"][0]
    idx = rng.choice(len(p_val), min(200_000, len(p_val)), replace=False)
    imp = permutation_importance(ranker, p_val.iloc[idx][cols_all].to_numpy(dtype=np.float32), p_val.label.to_numpy()[idx],
                                 scoring="average_precision", n_repeats=3, random_state=SEED)
    group_of = {c: g for g, cs in FEATURE_GROUPS.items() for c in cs}
    importance = pd.DataFrame(dict(feature=cols_all, group=[group_of[c] for c in cols_all], importance=imp.importances_mean, std=imp.importances_std))
    save_table(importance.sort_values("importance", ascending=False), "rq2_feature_importance")

    # 5. who benefits: results by activity tier (holdout)
    p_hold, ctx_hold = evals["holdout"]
    n365 = ctx_hold["frame"].n365.to_numpy()
    tier_of = np.select([n365 == 1, n365 <= 4, n365 <= 19], ["1 purchase", "2-4 purchases", "5-19 purchases"], "20+ purchases")
    rows = []
    for method in ("GlobalPopularity", "WeightedSum (first version)", "Ranker (all RQ features)"):
        for tier in ("1 purchase", "2-4 purchases", "5-19 purchases", "20+ purchases"):
            users = np.flatnonzero(tier_of == tier)
            rows.append(dict(method=method, tier=tier, **metrics(recs_by[(method, "holdout")], ctx_hold["truth"], users)))
    tiers = pd.DataFrame(rows)
    save_table(tiers, "rq2_product_tiers")

    # 6. recommendations with reasons (holdout): full list to the cache, eight customers with hits as examples
    recs = recs_by[("Ranker (all RQ features)", "holdout")]
    key = pd.Series(np.arange(len(p_hold)), index=pd.MultiIndex.from_arrays([p_hold.u.to_numpy(), p_hold.item.to_numpy()]))
    rec_rows = [(u, rank, it) for u, lst in recs.items() for rank, it in enumerate(lst[:TOP_N], 1)]
    ru = np.array([r[0] for r in rec_rows]); ritem = np.array([r[2] for r in rec_rows])
    pair_idx = key.loc[list(zip(ru, ritem))].to_numpy()
    reasons = explain(p_hold, pair_idx, cat, ctx_hold, tfidf)
    full = pd.DataFrame(dict(customer_id=ctx_hold["users"][ru], rank=[r[1] for r in rec_rows], article_id=cat.article_id.to_numpy()[ritem],
                             prod_name=cat.prod_name.to_numpy()[ritem], garment_group=cat.garment_group_name.to_numpy()[ritem],
                             score=ranker_scores["holdout"][pair_idx], reason=reasons,
                             bought=[ritem[k] in ctx_hold["truth"][ru[k]] for k in range(len(ru))]))
    full.to_parquet(CACHE / "rq2_recommendations_holdout.parquet", index=False)
    hits_per_user = full.groupby("customer_id").bought.sum()
    chosen = hits_per_user[hits_per_user > 0].index[:8]
    examples = []
    for cid in chosen:
        part = full[full.customer_id == cid]
        u = int(ru[np.flatnonzero(ctx_hold["users"][ru] == cid)[0]])
        examples.append(dict(customer_id=cid, n_purchases_last_year=int(n365[u]), cluster=int(ctx_hold["frame"].cluster.iat[u]),
                             recommendations=" | ".join(f"{r.rank}. {r.prod_name} [{r.reason}]{' *' if r.bought else ''}" for r in part.itertuples()),
                             bought=" | ".join(cat.prod_name.to_numpy()[sorted(ctx_hold["truth"][u])][:8]),
                             hits=int(part.bought.sum())))
    save_table(pd.DataFrame(examples), "rq2_product_examples")
    print(metrics_table.round(4).to_string(index=False))
    print(ablation[ablation.split == "holdout"].round(4).to_string(index=False))
    print(importance.sort_values("importance", ascending=False).head(12).round(4).to_string(index=False))
    print(steps.round(4).to_string(index=False))
    figures(metrics_table, ablation, importance, tiers, steps)
