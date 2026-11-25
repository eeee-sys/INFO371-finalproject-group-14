"""RQ2, category layer: which clothing categories will each customer buy next week?

Multi-label formulation: for every garment group there is one binary target, "the customer
bought this group in the 7 days after the cutoff". The 21 scores are ranked and the top
three groups are shown. Signals come in four groups, added one at a time (an ablation):

    history  purchase shares over the last 365 / 90 days, recency-weighted shares,
             activity numbers and the customer profile (age, club status, news)
    cluster  the customer's K-means preference cluster (clustering as preprocessing)
    rules    for every garment group, the highest confidence of an association rule
             "categories bought recently -> this group" (RQ4 rules, mined before the cutoff)
    season   how much each garment group sold in the same four weeks one year earlier,
             relative to its yearly share (seasonal variation)

Time handling is strict: every signal is computed from purchases before the cutoff, the
targets from the 7 days after it. Weekly cutoffs from 2019-12-25 to 2020-09-02 train the
model, the week of 2020-09-09 selects it, the last full week of data (2020-09-16 .. 09-22)
is the final holdout. For the product layer the final classifier is refitted in an expanding
window (train on every cutoff before the target cutoff) so that its probabilities are
out-of-sample at every date.

Metrics: Hit@1 (the top group was bought), Recall@3 and Precision@3 of the top three
against the set of groups actually bought, and macro average precision over groups.

Usage:
    python src/rq2_categories.py --mod 20
"""
import argparse
import time

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import CACHE, DEPARTMENTS, DEPARTMENT_SQL, SEED, base_views, connect, garment_groups, sample_events, save_table
import viz


def weekly(start: str, end: str) -> list[str]:
    """Dates from `start` to `end`, seven days apart."""
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="7D")]


TRAIN_CUTOFFS = weekly("2019-12-25", "2020-09-02")   # 37 weekly cutoffs, label weeks do not overlap
VALIDATION_CUTOFF = "2020-09-09"
HOLDOUT_CUTOFF = "2020-09-16"   # the last full week of data: 2020-09-16 .. 2020-09-22
PROBA_FROM = "2020-06-17"       # expanding-window probabilities are saved from this cutoff on (product layer)
REFIT_EVERY = 4                 # in the expanding window, refit every fourth week (plus validation and holdout)
LABEL_DAYS = 7
SEASON_DAYS = 28                # window for the seasonal lift, one year earlier
K_CLUSTERS = 4                  # auxiliary clusters for the category layer; K = 6 gave the same results
MIN_REGULAR = 5                 # customers with fewer purchases in the year form their own "occasional" group
CLUSTER_FIT_CUTOFFS = TRAIN_CUTOFFS[:4]   # cluster centroids come from the four earliest cutoffs only
CLUB = ["ACTIVE", "PRE-CREATE", "LEFT CLUB"]
NEWS = ["Regularly", "Monthly"]
BASELINES = ["PopularCategory", "HistoryModal365", "Decay30"]
BETAS = [0.5, 1.0, 1.5, 2.0]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]   # exponent on the base rate when ranking groups for display


# ----------------------------------------------------------------------------- features
def build_frame(con, cutoff: str, groups: list[str]) -> pd.DataFrame:
    """One row per customer with a purchase in the 365 days before `cutoff`: history + profile + targets."""
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW h AS
        SELECT e.*, {DEPARTMENT_SQL} AS department,
               (e.t_dat >= DATE '{cutoff}' - INTERVAL '90 days') AS recent,
               exp(-date_diff('day', e.t_dat, DATE '{cutoff}') / 30.0) AS decay_w
        FROM events e JOIN articles a USING (article_id)
        WHERE e.t_dat >= DATE '{cutoff}' - INTERVAL '365 days' AND e.t_dat < DATE '{cutoff}'""")
    totals = con.sql(f"""
        SELECT customer_id,
               count(*) n365,
               sum(CASE WHEN recent THEN 1 ELSE 0 END) n90,
               count(DISTINCT CASE WHEN recent THEN t_dat END) days90,
               count(DISTINCT CASE WHEN recent THEN garment_group_name END) categories90,
               sum(CASE WHEN recent THEN price ELSE 0 END) spend90,
               avg(price) mean_price,
               date_diff('day', max(t_dat), DATE '{cutoff}') recency_days,
               avg(CASE WHEN sales_channel_id = 2 THEN 1.0 ELSE 0.0 END) online_share,
               count(DISTINCT product_code) * 1.0 / count(*) novelty
        FROM h GROUP BY 1 ORDER BY 1""").df().set_index("customer_id")   # fixed row order: K-means and boosting depend on it
    per_cat = con.sql("""
        SELECT customer_id, garment_group_name,
               count(*) n365, sum(CASE WHEN recent THEN 1 ELSE 0 END) n90,
               sum(CASE WHEN recent THEN decay_w ELSE 0 END) decay
        FROM h GROUP BY 1, 2""").df()
    per_dept = con.sql("SELECT customer_id, department, count(*) n FROM h GROUP BY 1, 2").df()

    def shares(table, value, column, names, prefix):
        wide = table.pivot(index="customer_id", columns=column, values=value)
        wide = wide.reindex(index=totals.index, columns=names).fillna(0.0)
        return wide.div(wide.sum(axis=1).replace(0, 1.0), axis=0).add_prefix(prefix)

    frame = pd.concat([totals,
                       shares(per_cat, "n365", "garment_group_name", groups, "p365:"),
                       shares(per_cat, "n90", "garment_group_name", groups, "p90:"),
                       shares(per_cat, "decay", "garment_group_name", groups, "d30:"),
                       shares(per_dept, "n", "department", DEPARTMENTS, "dept:")], axis=1)

    customers = con.sql("""
        SELECT customer_id, TRY_CAST(age AS DOUBLE) age, club_member_status, fashion_news_frequency,
               CASE WHEN FN IS NULL OR FN = '' THEN 0 ELSE 1 END fn,
               CASE WHEN Active IS NULL OR Active = '' THEN 0 ELSE 1 END active
        FROM customers""").df().set_index("customer_id").reindex(frame.index)
    frame["age_missing"] = customers.age.isna().astype(int)
    frame["age"] = customers.age
    for v in CLUB:
        frame[f"club:{v}"] = (customers.club_member_status == v).astype(int)
    for v in NEWS:
        frame[f"news:{v}"] = (customers.fashion_news_frequency == v).astype(int)
    frame["fn"], frame["active"] = customers.fn, customers.active

    targets = con.sql(f"""
        SELECT customer_id, list(DISTINCT garment_group_name) AS targets FROM events
        WHERE t_dat >= DATE '{cutoff}' AND t_dat < DATE '{cutoff}' + INTERVAL '{LABEL_DAYS} days'
        GROUP BY 1""").df().set_index("customer_id").targets
    frame["targets"] = targets.reindex(frame.index)
    frame["cutoff"] = cutoff
    return frame.reset_index()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ("customer_id", "targets", "cutoff")]


def season_features(con, cutoff: str, groups: list[str]) -> np.ndarray:
    """Seasonal lift per garment group: share in the same four weeks one year earlier,
    divided by the share over the whole preceding year. Computed on all transactions."""
    last_year = con.sql(f"""
        SELECT a.garment_group_name g, count(*) n FROM transactions_train t JOIN articles a USING (article_id)
        WHERE t.t_dat >= DATE '{cutoff}' - INTERVAL '365 days' AND t.t_dat < DATE '{cutoff}' GROUP BY 1""").df().set_index("g").n
    same_window = con.sql(f"""
        SELECT a.garment_group_name g, count(*) n FROM transactions_train t JOIN articles a USING (article_id)
        WHERE t.t_dat >= DATE '{cutoff}' - INTERVAL '365 days'
          AND t.t_dat < DATE '{cutoff}' - INTERVAL '365 days' + INTERVAL '{SEASON_DAYS} days' GROUP BY 1""").df().set_index("g").n
    year = last_year.reindex(groups).fillna(0) + 1
    window = same_window.reindex(groups).fillna(0) + 1
    return ((window / window.sum()) / (year / year.sum())).to_numpy(dtype=np.float32)


def mine_rules(con, cutoff: str, groups: list[str]) -> pd.DataFrame:
    """Category rules from same-customer, same-day baskets in the year before the cutoff."""
    baskets = con.sql(f"""
        SELECT list(DISTINCT garment_group_name) cats FROM events
        WHERE t_dat >= DATE '{cutoff}' - INTERVAL '365 days' AND t_dat < DATE '{cutoff}'
        GROUP BY customer_id, t_dat""").df()
    matrix = pd.DataFrame(MultiLabelBinarizer(classes=groups).fit_transform(baskets.cats).astype(bool), columns=groups)
    frequent = fpgrowth(matrix, min_support=0.005, use_colnames=True, max_len=3)
    rules = association_rules(frequent, num_itemsets=len(matrix), metric="confidence", min_threshold=0.1)
    rules = rules[(rules.lift > 1.0) & (rules.consequents.map(len) == 1)]
    return rules[["antecedents", "consequents", "confidence", "lift"]].reset_index(drop=True)


def rule_features(rules: pd.DataFrame, frame: pd.DataFrame, groups: list[str]) -> np.ndarray:
    """For each customer and garment group: the highest confidence of a rule whose antecedent
    is contained in the categories the customer bought in the last 90 days (or the year, if none)."""
    recent = frame[[f"p90:{g}" for g in groups]].to_numpy() > 0
    none = ~recent.any(axis=1)
    recent[none] = frame.loc[none, [f"p365:{g}" for g in groups]].to_numpy() > 0
    index = {g: i for i, g in enumerate(groups)}
    score = np.zeros((len(frame), len(groups)), dtype=np.float32)
    for r in rules.itertuples():
        ante = [index[g] for g in r.antecedents]
        cons = index[next(iter(r.consequents))]
        hit = recent[:, ante].all(axis=1)
        score[hit, cons] = np.maximum(score[hit, cons], r.confidence)
    return score


def fit_clusters(train: pd.DataFrame, groups: list[str]) -> KMeans:
    cols = [f"p365:{g}" for g in groups] + [f"dept:{d}" for d in DEPARTMENTS]
    regular = train[train.n365 >= MIN_REGULAR]
    return KMeans(n_clusters=K_CLUSTERS, n_init=5, random_state=SEED).fit(np.sqrt(regular[cols].to_numpy(dtype=np.float32)))


def cluster_features(model: KMeans, frame: pd.DataFrame, groups: list[str]) -> tuple[np.ndarray, np.ndarray]:
    cols = [f"p365:{g}" for g in groups] + [f"dept:{d}" for d in DEPARTMENTS]
    labels = model.predict(np.sqrt(frame[cols].to_numpy(dtype=np.float32)))
    labels[frame.n365.to_numpy() < MIN_REGULAR] = K_CLUSTERS          # occasional buyers
    onehot = np.zeros((len(frame), K_CLUSTERS + 1), dtype=np.float32)
    onehot[np.arange(len(frame)), labels] = 1
    return onehot, labels


# ----------------------------------------------------------------------------- scoring
def baseline_scores(name: str, frame: pd.DataFrame, groups: list[str], popular: list[str]) -> np.ndarray:
    if name == "PopularCategory":                     # the most common groups in training, same order for everyone
        s = np.zeros((len(frame), len(groups)), dtype=np.float32)
        for rank, g in enumerate(popular):
            s[:, groups.index(g)] = len(groups) - rank
        return s
    prefix = {"HistoryModal365": "p365:", "Decay30": "d30:"}[name]
    s = frame[[prefix + g for g in groups]].to_numpy(dtype=np.float32).copy()
    empty = s.sum(axis=1) == 0
    s[empty] = frame.loc[empty, ["p365:" + g for g in groups]].to_numpy(dtype=np.float32)
    return s


def season_adjust(scores: np.ndarray, lift: np.ndarray, beta: float) -> np.ndarray:
    """Multiply each garment group's score by its seasonal lift ** beta (beta = 0 means no adjustment)."""
    return scores * (lift[None, :] ** beta)


def lift_adjust(scores: np.ndarray, base: np.ndarray, alpha: float) -> np.ndarray:
    """Divide each group's score by its base rate ** alpha: alpha = 0 ranks by probability (the most
    common groups win), alpha = 1 ranks by lift (groups the customer is unusually likely to buy)."""
    return scores / (np.maximum(base, 1e-6)[None, :] ** alpha)


def evaluate(Y: np.ndarray, scores: np.ndarray) -> dict:
    order = np.argsort(-scores, axis=1, kind="stable")
    rows = np.arange(len(Y))[:, None]
    top3 = Y[rows, order[:, :3]]
    n_true = Y.sum(axis=1)
    in_top3 = np.zeros_like(Y, dtype=bool)
    in_top3[rows, order[:, :3]] = True
    per_group_recall = [in_top3[Y[:, j] == 1, j].mean() for j in range(Y.shape[1]) if Y[:, j].any()]
    ap = [average_precision_score(Y[:, j], scores[:, j]) for j in range(Y.shape[1]) if Y[:, j].any()]
    return dict(hit1=float(Y[np.arange(len(Y)), order[:, 0]].mean()),
                recall3=float((top3.sum(axis=1) / n_true).mean()),
                precision3=float(top3.mean()),
                macro_recall3=float(np.mean(per_group_recall)),
                macro_ap=float(np.mean(ap)))


def per_group(Y: np.ndarray, scores: np.ndarray, groups: list[str]) -> pd.DataFrame:
    order = np.argsort(-scores, axis=1, kind="stable")
    in_top3 = np.zeros_like(Y, dtype=bool)
    in_top3[np.arange(len(Y))[:, None], order[:, :3]] = True
    rows = []
    for j, g in enumerate(groups):
        bought, shown = Y[:, j].astype(bool), in_top3[:, j]
        rows.append(dict(garment_group=g, customers_bought=int(bought.sum()), share_bought=float(bought.mean()),
                         recall3=float(shown[bought].mean()) if bought.any() else np.nan,
                         precision3=float(bought[shown].mean()) if shown.any() else np.nan,
                         times_shown=int(shown.sum()),
                         average_precision=float(average_precision_score(Y[:, j], scores[:, j])) if bought.any() else np.nan))
    return pd.DataFrame(rows)


def ovr(estimator, parallel: bool = True) -> OneVsRestClassifier:
    return OneVsRestClassifier(estimator, n_jobs=-1 if parallel else 1)


def logistic():
    return ovr(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)))


def classifier_choices() -> dict:
    return {
        "LogisticRegression": logistic(),
        "DecisionTree": ovr(DecisionTreeClassifier(max_depth=10, min_samples_leaf=50, random_state=SEED)),
        # all signals are non-negative shares, counts, lifts or flags, so the multinomial variant applies
        "NaiveBayes": ovr(MultinomialNB(alpha=1.0)),
        "Boosting": ovr(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, early_stopping=True,
                                                       validation_fraction=0.1, random_state=SEED), parallel=False),
    }


# ----------------------------------------------------------------------------- figures
READABLE = {"PopularCategory": "Most popular groups", "HistoryModal365": "Customer's most frequent groups",
            "Decay30": "Recently bought groups", "history + profile": "History and profile",
            "+ cluster": "+ customer cluster", "+ rules": "+ association rules", "+ season": "+ seasonal signal",
            "LogisticRegression": "Logistic regression", "DecisionTree": "Decision tree", "NaiveBayes": "Naive Bayes",
            "Boosting": "Gradient boosting"}


def figures(ablation: pd.DataFrame, groups_table: pd.DataFrame, baseline_groups: pd.DataFrame, final_name: str) -> None:
    viz.style()
    hold = ablation[ablation.split == "holdout"].set_index("step")
    rows = BASELINES + ["history + profile", "+ cluster", "+ rules", "+ season"]
    vals = hold.loc[rows].recall3.to_numpy()
    y = np.array([0, 1, 2, 3.6, 4.6, 5.6, 6.6])                 # a gap between the baselines and the model
    colours = [viz.CONTEXT] * 3 + [viz.SEQ[3]] * 3 + [viz.HIGHLIGHT]
    fig, ax = viz.plt.subplots(figsize=(viz.FULL, 2.55))
    ax.barh(y, vals, color=colours, height=0.62)
    ax.set_xlim(0, 0.52)
    viz.bar_labels(ax, y, vals, [f"{v:.1%}" for v in vals])
    ax.set_yticks(y); ax.set_yticklabels([READABLE[r] for r in rows]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    for yy, text in ((-0.75, "Baselines"), (2.85, "Logistic regression, signal groups added one at a time")):
        ax.text(0.005, yy, text, fontsize=6.8, color=viz.TEXT2, style="italic", va="center")
    ax.set_ylim(7.2, -1.2)
    ax.set_xlabel("Recall@3: share of the groups each customer bought that appear in the top three")
    viz.titled(ax, "History beats every baseline, and the seasonal signal adds most",
               f"Category layer, test week from {HOLDOUT_CUTOFF}, 3,031 customers")
    viz.save(fig, "rq2_ablation")

    g = groups_table.merge(baseline_groups[["garment_group", "recall3"]].rename(columns={"recall3": "baseline"}), on="garment_group")
    g = g[g.customers_bought >= 15].sort_values("customers_bought", ascending=False).reset_index(drop=True)
    yy = np.arange(len(g))
    fig, ax = viz.plt.subplots(figsize=(viz.FULL, 3.9))
    for i, r in g.iterrows():
        ax.plot([r.baseline, r.recall3], [i, i], color=viz.GRID, linewidth=2.4, solid_capstyle="round", zorder=1)
    ax.scatter(g.baseline, yy, s=26, color=viz.CONTEXT_DARK, edgecolor=viz.SURFACE, linewidth=1, zorder=2, label="Customer's most frequent groups (baseline)")
    ax.scatter(g.recall3, yy, s=30, color=viz.HIGHLIGHT, edgecolor=viz.SURFACE, linewidth=1, zorder=3, label=final_name)
    for i, r in g.iterrows():
        if r.garment_group in ("Knitwear", "Jersey Fancy", "Swimwear"):
            ax.annotate(f"{r.recall3:.0%}", (r.recall3, i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=6.8, color=viz.TEXT)
    ax.set_yticks(yy); ax.set_yticklabels([f"{r.garment_group}  ({r.customers_bought:,})" for r in g.itertuples()], fontsize=7)
    ax.invert_yaxis(); ax.set_xlim(-0.02, 1.0); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Recall@3: share of the group's buyers who saw it among their top three")
    ax.legend(loc="lower right", handletextpad=0.2)
    viz.titled(ax, "The model now finds knitwear; small summer groups are rarely shown",
               "Test week by garment group, groups with at least 15 buyers (number of buyers in brackets)")
    viz.save(fig, "rq2_group_recall")


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod", type=int, default=5)
    args = parser.parse_args()

    con = connect()
    base_views(con)
    sample_events(con, args.mod)
    groups = garment_groups(con)
    cutoffs = TRAIN_CUTOFFS + [VALIDATION_CUTOFF, HOLDOUT_CUTOFF]
    mlb = MultiLabelBinarizer(classes=groups)

    # 1. frames, rules and seasonal lifts per cutoff (all from data before the cutoff)
    frames, rule_sets, seasons = {}, {}, {}
    for c in cutoffs:
        start = time.perf_counter()
        frames[c] = build_frame(con, c, groups)
        rule_sets[c] = mine_rules(con, c, groups)
        seasons[c] = season_features(con, c, groups)
        n_lab = int(frames[c].targets.notna().sum())
        print(f"{c}: {len(frames[c]):,} customers, {n_lab:,} labelled, {len(rule_sets[c])} rules ({time.perf_counter() - start:.0f}s)")
    coverage = pd.DataFrame([dict(cutoff=c, split="train" if c in TRAIN_CUTOFFS else ("validation" if c == VALIDATION_CUTOFF else "holdout"),
                                  customers_with_history=len(f), customers_with_label=int(f.targets.notna().sum()),
                                  label_coverage=f.targets.notna().mean(),
                                  mean_groups_bought=float(f.targets.dropna().map(len).mean()), rules=len(rule_sets[c]))
                             for c, f in frames.items()])
    save_table(coverage, "rq2_label_coverage")

    # 2. cluster model fitted on the earliest training cutoffs only
    kmeans = fit_clusters(pd.concat([frames[c] for c in CLUSTER_FIT_CUTOFFS], ignore_index=True), groups)

    # 3. feature blocks for every frame. Training-only cutoffs keep just their labelled customers;
    #    cutoffs from PROBA_FROM on keep everyone, because the product layer needs all their probabilities.
    history_cols = feature_columns(frames[VALIDATION_CUTOFF])
    blocks, labelled = {}, {}
    for c, f in frames.items():
        onehot, labels = cluster_features(kmeans, f, groups)
        f["cluster"] = labels
        lab = np.flatnonzero(f.targets.notna().to_numpy())
        keep = np.arange(len(f)) if c >= PROBA_FROM else lab
        blocks[c] = dict(history=f[history_cols].to_numpy(dtype=np.float32)[keep],
                         cluster=onehot[keep],
                         rules=rule_features(rule_sets[c], f.iloc[keep], groups),
                         season=np.repeat(seasons[c][None, :], len(keep), axis=0))
        labelled[c] = (np.searchsorted(keep, lab), mlb.fit_transform(f.targets.iloc[lab]))   # rows index the kept block
        if c not in (VALIDATION_CUTOFF, HOLDOUT_CUTOFF):                                     # the wide columns are no longer needed
            frames[c] = f[["customer_id", "targets", "age", "cluster"]]
    age_fill = float(pd.concat([frames[c].age for c in TRAIN_CUTOFFS]).median())
    log_cols = [history_cols.index(col) for col in ("n365", "n90", "days90", "spend90", "mean_price", "recency_days")]

    def matrix(c: str, use: list[str], rows: np.ndarray) -> np.ndarray:
        hist = blocks[c]["history"][rows].copy()
        hist[:, history_cols.index("age")] = np.nan_to_num(hist[:, history_cols.index("age")], nan=age_fill)
        hist[:, log_cols] = np.log1p(hist[:, log_cols])
        return np.hstack([hist] + [blocks[c][b][rows] for b in use if b != "history"])

    val_rows, Y_val = labelled[VALIDATION_CUTOFF]
    hold_rows, Y_hold = labelled[HOLDOUT_CUTOFF]
    val_frame = frames[VALIDATION_CUTOFF].iloc[val_rows].reset_index(drop=True)
    hold_frame = frames[HOLDOUT_CUTOFF].iloc[hold_rows].reset_index(drop=True)
    Y_train = np.vstack([labelled[c][1] for c in TRAIN_CUTOFFS])
    popular = [groups[j] for j in np.argsort(-Y_train.sum(axis=0))]
    print(f"train {len(Y_train):,} rows, validation {len(Y_val):,}, holdout {len(Y_hold):,}; "
          f"{Y_train.sum(axis=1).mean():.2f} groups bought per labelled customer")

    def stack(use: list[str], train_cutoffs: list[str]):
        return (np.vstack([matrix(c, use, labelled[c][0]) for c in train_cutoffs]),
                matrix(VALIDATION_CUTOFF, use, val_rows), matrix(HOLDOUT_CUTOFF, use, hold_rows))

    # 4. ablation with one-vs-rest logistic regression: add one signal group at a time
    steps = [("history + profile", ["history"]), ("+ cluster", ["history", "cluster"]),
             ("+ rules", ["history", "cluster", "rules"]), ("+ season", ["history", "cluster", "rules", "season"])]
    rows, scorers, n_features = [], {}, {}
    for name in BASELINES:
        scorers[name] = dict(validation=baseline_scores(name, val_frame, groups, popular), holdout=baseline_scores(name, hold_frame, groups, popular))
        for split_name, Y in (("validation", Y_val), ("holdout", Y_hold)):
            rows.append(dict(step=name, split=split_name, features=0, **evaluate(Y, scorers[name][split_name])))
    for step, use in steps:
        X_tr, X_val, X_hold = stack(use, TRAIN_CUTOFFS)
        n_features[step] = X_tr.shape[1]
        start = time.perf_counter()
        model = logistic().fit(X_tr, Y_train)
        scorers[step] = dict(validation=model.predict_proba(X_val), holdout=model.predict_proba(X_hold))
        for split_name, Y in (("validation", Y_val), ("holdout", Y_hold)):
            rows.append(dict(step=step, split=split_name, features=X_tr.shape[1], **evaluate(Y, scorers[step][split_name])))
        v = rows[-2]
        print(f"{step:20s} {X_tr.shape[1]:3d} features  val hit@1 {v['hit1']:.3f}  recall@3 {v['recall3']:.3f}  "
              f"precision@3 {v['precision3']:.3f}  macro-recall@3 {v['macro_recall3']:.3f}  ({time.perf_counter() - start:.0f}s)")
    lifts = dict(validation=seasons[VALIDATION_CUTOFF], holdout=seasons[HOLDOUT_CUTOFF])
    for base in ("HistoryModal365", "+ rules"):
        best = max(BETAS, key=lambda b: evaluate(Y_val, season_adjust(scorers[base]["validation"], lifts["validation"], b))["recall3"])
        step = f"{base} x season (beta {best})"
        scorers[step] = {s: season_adjust(scorers[base][s], lifts[s], best) for s in ("validation", "holdout")}
        for split_name, Y in (("validation", Y_val), ("holdout", Y_hold)):
            rows.append(dict(step=step, split=split_name, features=n_features.get(base, 0), **evaluate(Y, scorers[step][split_name])))
        v = rows[-2]
        print(f"{step:34s} val hit@1 {v['hit1']:.3f}  recall@3 {v['recall3']:.3f}  precision@3 {v['precision3']:.3f}  macro-AP {v['macro_ap']:.3f}")
    ablation = pd.DataFrame(rows)
    save_table(ablation, "rq2_ablation")

    # 5. classifier choice on the full signal set, each also tried with the seasonal prior
    ALL = ["history", "cluster", "rules", "season"]
    X_tr, X_val, X_hold = stack(ALL, TRAIN_CUTOFFS)
    rows, betas_of = [], {}
    for name, model in classifier_choices().items():
        start = time.perf_counter()
        model.fit(X_tr, Y_train)
        raw = dict(validation=model.predict_proba(X_val), holdout=model.predict_proba(X_hold))
        best = max([0.0] + BETAS, key=lambda b: evaluate(Y_val, season_adjust(raw["validation"], lifts["validation"], b))["recall3"])
        betas_of[name] = best
        scorers[name] = {s: season_adjust(raw[s], lifts[s], best) for s in ("validation", "holdout")}
        for split_name, Y in (("validation", Y_val), ("holdout", Y_hold)):
            rows.append(dict(classifier=name, season_beta=best, split=split_name, seconds=time.perf_counter() - start,
                             **evaluate(Y, scorers[name][split_name])))
        v = rows[-2]
        print(f"{name:20s} beta {best:<4} val hit@1 {v['hit1']:.3f}  recall@3 {v['recall3']:.3f}  precision@3 {v['precision3']:.3f}  "
              f"macro-AP {v['macro_ap']:.3f}  ({v['seconds']:.0f}s)")
    classifiers = pd.DataFrame(rows)
    save_table(classifiers, "rq2_classifier_choice")
    final = classifiers[classifiers.split == "validation"].sort_values("recall3", ascending=False).iloc[0].classifier
    print(f"final classifier (best validation Recall@3): {final}, seasonal beta {betas_of[final]}")
    print(ablation[ablation.split == "holdout"].round(4).to_string(index=False))
    print(classifiers[classifiers.split == "holdout"].round(4).to_string(index=False))

    # 6. what to show: ranking by probability favours the most common groups, ranking by lift favours
    #     personal ones; alpha trades the two off and is chosen on validation (mean of micro and macro Recall@3)
    base_train = Y_train.mean(axis=0)
    rows = []
    for a in ALPHAS:
        for split_name, Y in (("validation", Y_val), ("holdout", Y_hold)):
            rows.append(dict(alpha=a, split=split_name, **evaluate(Y, lift_adjust(scorers[final][split_name], base_train, a))))
    display = pd.DataFrame(rows)
    save_table(display, "rq2_display_alpha")
    v = display[display.split == "validation"].assign(objective=lambda d: (d.recall3 + d.macro_recall3) / 2)
    alpha = float(v.sort_values("objective", ascending=False).iloc[0].alpha)
    print(display.round(4).to_string(index=False))
    print(f"display alpha (best mean of Recall@3 and macro Recall@3 on validation): {alpha}")

    # 7. expanding window: the final classifier is trained on every cutoff before the target cutoff (refitted every
    #    REFIT_EVERY weeks and for validation / holdout; a model fitted on earlier weeks stays out-of-sample for later
    #    ones), probabilities are saved for all customers of that cutoff (the product layer uses them), and the
    #    labelled customers' metrics are reported
    rows, model, fitted_on, base_rate, n_train = [], None, [], None, 0
    for i, c in enumerate([x for x in cutoffs if x >= PROBA_FROM]):
        start = time.perf_counter()
        if model is None or i % REFIT_EVERY == 0 or c in (VALIDATION_CUTOFF, HOLDOUT_CUTOFF):
            fitted_on = [x for x in cutoffs if x < c]
            X_tr = np.vstack([matrix(x, ALL, labelled[x][0]) for x in fitted_on])
            Y_tr = np.vstack([labelled[x][1] for x in fitted_on])
            model = classifier_choices()[final].fit(X_tr, Y_tr)
            base_rate, n_train = Y_tr.mean(axis=0), len(Y_tr)
        all_rows = np.arange(len(blocks[c]["history"]))
        proba = season_adjust(model.predict_proba(matrix(c, ALL, all_rows)), seasons[c], betas_of[final]).astype(np.float32)
        table = pd.DataFrame(proba, columns=groups)
        table.insert(0, "customer_id", frames[c].customer_id.to_numpy())
        table.to_parquet(CACHE / f"rq2_category_proba_{c}.parquet", index=False)
        lab_rows, Y_c = labelled[c]
        shown = lift_adjust(proba[lab_rows], base_rate, alpha)
        rows.append(dict(cutoff=c, trained_on_cutoffs=len(fitted_on), last_train_cutoff=fitted_on[-1], train_rows=n_train,
                         customers_scored=len(table), labelled=len(lab_rows), **evaluate(Y_c, proba[lab_rows]),
                         **{f"display_{k}": v for k, v in evaluate(Y_c, shown).items()}))
        print(f"expanding {c}: model trained up to {fitted_on[-1]} ({len(fitted_on)} cutoffs, {n_train:,} rows)  "
              f"hit@1 {rows[-1]['hit1']:.3f}  recall@3 {rows[-1]['recall3']:.3f}  | shown (alpha {alpha}) "
              f"recall@3 {rows[-1]['display_recall3']:.3f}  macro {rows[-1]['display_macro_recall3']:.3f}  ({time.perf_counter() - start:.0f}s)")
        if c < VALIDATION_CUTOFF:                     # scored: keep only the labelled rows for later refits
            blocks[c] = {k: v[lab_rows] for k, v in blocks[c].items()}
            labelled[c] = (np.arange(len(lab_rows)), Y_c)
        if c in (VALIDATION_CUTOFF, HOLDOUT_CUTOFF):
            split_name = "validation" if c == VALIDATION_CUTOFF else "holdout"
            order = np.argsort(-shown, axis=1, kind="stable")
            f = frames[c].iloc[lab_rows]
            save_table(pd.DataFrame(dict(customer_id=f.customer_id.to_numpy(), bought=f.targets.map(lambda g: "; ".join(sorted(g))).to_numpy(),
                                         top1=np.asarray(groups)[order[:, 0]], top2=np.asarray(groups)[order[:, 1]],
                                         top3=np.asarray(groups)[order[:, 2]])), f"rq2_top3_{split_name}")
            if split_name == "holdout":
                groups_table = per_group(Y_c, shown, groups)
                save_table(groups_table, "rq2_holdout_per_group")
                baseline_groups = per_group(Y_c, scorers["HistoryModal365"]["holdout"], groups)
    expanding = pd.DataFrame(rows)
    save_table(expanding, "rq2_expanding_window")
    figures(ablation, groups_table, baseline_groups, f"{READABLE[final]}, shown with \u03b1 = {alpha}")
