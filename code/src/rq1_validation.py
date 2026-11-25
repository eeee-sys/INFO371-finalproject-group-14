"""RQ1 customer groups: what does the low silhouette mean, and do the groups hold for later purchases?

The six style groups of rq1_clustering.py have a silhouette of 0.089. This script puts that number next to
two references and then tests the groups on purchases they were not built from. It changes no result of
rq1_clustering.py and nothing in the recommender.

1. Reference for the silhouette. The same K-means (K = 6) is run
     - on other customer vectors: fewer SVD dimensions, and garment-group shares;
     - on shuffled data: every dimension is shuffled over the customers, so each dimension keeps its values
       but the structure between dimensions is gone (average of several shuffles).
   The adjusted Rand index shows how similar the groups are to the groups of the report.
2. Later purchases. At four cutoff dates the groups are built from the year before the cutoff only. The
   purchases of the following four weeks show
     - whether each group still buys its leading garment group more often than the average customer;
     - how much of a customer's purchases fall into the top three garment groups of the group, compared with
       the top three of all customers and the top three of the customer's own history.

Outputs: tables rq1_silhouette_reference, rq1_later_groups, rq1_later_purchases.

Usage:
    python src/rq1_validation.py --mod 20          (after rq3_text_mining.py and rq1_clustering.py)
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import normalize

from common import DATA_END, SEED, TABLES, base_views, connect, garment_groups, sample_events, save_table
from rq1_clustering import customer_features, style_vectors

K = 6                                           # groups of the report
WINDOW_DAYS, MIN_PURCHASES = 365, 5             # regular customers, as in rq1_clustering
SAMPLE_N = 5000                                 # customers in the silhouette sample, as in rq1_clustering
SHUFFLES = 5
SVD_SIZES = [50, 20, 10, 5]
CUTOFFS = ["2019-11-13", "2020-02-12", "2020-06-10", "2020-08-26"]   # the four weeks after each avoid the store closure
LATER_DAYS = 28
LEAD_MIN_SHARE = 0.05                           # a leading garment group holds at least 5% of the group's purchases
TOP_GROUPS = 3


def regular_customers(con, end: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Garment-group shares and the 50-dimension taste vectors of regular customers in the year before `end`."""
    activity, shares = customer_features(con, WINDOW_DAYS, end)
    regular = activity.index[activity.n_purchases >= MIN_PURCHASES]
    Z, _, _ = style_vectors(con, pd.Index(regular))
    return shares.loc[regular, garment_groups(con)], Z


def kmeans_labels(X: np.ndarray) -> np.ndarray:
    return KMeans(n_clusters=K, n_init=5, random_state=SEED).fit_predict(X)


def shuffled(X: np.ndarray, rng: np.random.Generator, unit_length: bool) -> np.ndarray:
    """Every dimension shuffled over the customers on its own."""
    S = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
    return normalize(S) if unit_length else S


def silhouette_reference(shares: pd.DataFrame, Z: np.ndarray) -> pd.DataFrame:
    """Silhouette at K = 6 for several customer vectors, next to the same K-means on shuffled data."""
    idx = np.random.default_rng(SEED).choice(len(Z), min(SAMPLE_N, len(Z)), replace=False)   # same sample as rq1_clustering
    vectors = {f"taste vector, {d} dimensions": (normalize(Z[:, :d]).astype(np.float32), True) for d in SVD_SIZES}
    vectors["garment-group shares (square root)"] = (np.sqrt(shares.to_numpy(dtype=np.float32)), False)
    rng = np.random.default_rng(SEED)
    rows, report_labels = [], None
    for name, (X, unit_length) in vectors.items():
        labels = kmeans_labels(X)
        if report_labels is None:
            report_labels = labels                                   # the first entry is the version of the report
        null = []
        for _ in range(SHUFFLES):
            S = shuffled(X, rng, unit_length)
            null.append(silhouette_score(S[idx], kmeans_labels(S)[idx]))
        sil = silhouette_samples(X[idx], labels[idx])
        rows.append(dict(customer_vector=name, k=K, customers=len(X), silhouette=sil.mean(), positive_share=(sil > 0).mean(),
                         silhouette_shuffled=np.mean(null), shuffled_min=np.min(null), shuffled_max=np.max(null),
                         ari_with_report_groups=adjusted_rand_score(report_labels, labels)))
        print(f"{name:36s} silhouette {rows[-1]['silhouette']:.3f}   shuffled {rows[-1]['silhouette_shuffled']:.3f}   "
              f"ARI {rows[-1]['ari_with_report_groups']:.2f}")
    return pd.DataFrame(rows)


def later_purchases(con, cutoff: str, groups: list[str]) -> tuple[list[dict], list[dict]]:
    """Groups built from the year before the cutoff, tested on the purchases of the four weeks after it."""
    shares, Z = regular_customers(con, cutoff)
    labels = kmeans_labels(Z)
    past = shares.to_numpy()
    overall = past.mean(axis=0)
    by_group = np.vstack([past[labels == c].mean(axis=0) for c in range(K)])

    later = con.sql(f"""SELECT customer_id, garment_group_name g, count(*) n FROM events
                        WHERE t_dat >= DATE '{cutoff}' AND t_dat < DATE '{cutoff}' + INTERVAL {LATER_DAYS} DAY GROUP BY 1, 2""").df()
    later = later[later.customer_id.isin(shares.index)]
    counts = later.pivot(index="customer_id", columns="g", values="n").reindex(columns=groups).fillna(0.0)
    bought = counts.to_numpy()
    bought_share = bought / bought.sum(axis=1, keepdims=True)
    pos = shares.index.get_indexer(counts.index)
    group_of = labels[pos]

    # each group's leading garment group: the highest share relative to the average among those with at least 5%
    all_later = bought.sum(axis=0) / bought.sum()
    rows = []
    for c in range(K):
        lift = np.where(by_group[c] >= LEAD_MIN_SHARE, by_group[c] / overall, 0)
        lead = int(lift.argmax())
        mine = bought[group_of == c].sum(axis=0)
        rows.append(dict(cutoff=cutoff, group=c, customers=int((labels == c).sum()), buyers_later=int((group_of == c).sum()),
                         leading_garment_group=groups[lead], times_average_year_before=lift[lead],
                         times_average_later=(mine[lead] / mine.sum()) / all_later[lead]))

    # share of a customer's later purchases inside the top three garment groups, and the log loss of the shares
    own = past[pos]
    scores = {"all customers": np.tile(overall, (len(counts), 1)), "customer's group": by_group[group_of], "customer's own history": own}
    rows2 = []
    for name, s in scores.items():
        top = np.argsort(-s, axis=1)[:, :TOP_GROUPS]
        q = 0.9 * s + 0.1 * overall if name == "customer's own history" else s       # avoids zero shares
        q = np.maximum(q, 1e-4)
        q = q / q.sum(axis=1, keepdims=True)
        rows2.append(dict(cutoff=cutoff, buyers=len(counts), top_three_of=name,
                          share_of_later_purchases=np.take_along_axis(bought_share, top, 1).sum(axis=1).mean(),
                          log_loss=-(bought_share * np.log(q)).sum(axis=1).mean()))
    return rows, rows2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod", type=int, default=20)
    args = parser.parse_args()
    pd.set_option("display.width", 200, "display.max_columns", 20)
    con = connect()
    base_views(con)
    sample_events(con, args.mod)
    groups = garment_groups(con)

    # the groups rebuilt here must be the groups of the report
    shares, Z = regular_customers(con, DATA_END)
    saved = pd.read_csv(TABLES / "rq1_assignments.csv", dtype={"customer_id": str}).set_index("customer_id").cluster
    same = adjusted_rand_score(saved.reindex(shares.index), kmeans_labels(Z))
    print(f"agreement with rq1_assignments (adjusted Rand index): {same:.3f}")
    assert same > 0.99, "the groups differ from rq1_assignments: run rq1_clustering.py --mod 20 --k-final 6 first"

    reference = silhouette_reference(shares, Z)
    save_table(reference, "rq1_silhouette_reference")

    lead_rows, share_rows = [], []
    for cutoff in CUTOFFS:
        a, b = later_purchases(con, cutoff, groups)
        lead_rows += a
        share_rows += b
    lead, later = pd.DataFrame(lead_rows), pd.DataFrame(share_rows)
    save_table(lead, "rq1_later_groups")
    save_table(later, "rq1_later_purchases")
    print(reference.round(3).to_string(index=False))
    print(lead.round(2).to_string(index=False))
    print(later.round(3).to_string(index=False))
    print(f"leading garment group later, times the average: {lead.times_average_later.min():.2f} to {lead.times_average_later.max():.2f}; "
          f"above 1 in {(lead.times_average_later > 1).sum()} of {len(lead)} cases")
