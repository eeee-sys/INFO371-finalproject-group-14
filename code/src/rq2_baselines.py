"""RQ2: the recommendation model against baselines, on the products it finally recommends.

The category layer and the product layer together form one model, so all methods are compared on the same
final task: which products does a customer buy in the next seven days (Recall@10, Hit@10, MAP@12). The split,
the customers and the metrics are those of rq2_products.py; nothing that script saved is changed.

Simple rules
    popularity        best sellers of all time
    trending          best sellers of the last 7 days
    trending by age   best sellers of the last 7 days in the customer's age group
    co-purchase       products that share baskets with what the customer bought (item-based neighbours)
    re-buy            products the customer bought before, recent and frequent ones first
Classifiers from the course
    naive Bayes, decision tree and logistic regression score the same candidate products as our model. They use
    the 21 base features (customer activity, product sales and price, source scores), without the signals of the
    other research questions, the category layer and the two feature groups that came from the findings.
    Each is also run with all 39 features, to see how much is due to the method and how much to the information.
Our model
    gradient boosting with all 39 features; refitted here on the same pairs, it must reproduce rq2_product_metrics.

Every list follows the same display rule: at most two colours of one product.

Outputs: table rq2_baselines, figure rq2_baselines.

Usage:
    python src/rq2_baselines.py --mod 20        (after rq2_products.py; about 15 minutes)
    python src/rq2_baselines.py --figure        (only draws the figure from the saved tables)
"""
import argparse
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import CACHE, SEED, TABLES, base_views, connect, garment_groups, sample_events, save_table
from rq2_categories import HOLDOUT_CUTOFF, VALIDATION_CUTOFF
from rq2_products import (BEST_SELLER_DAYS, CATEGORICAL, FEATURE_GROUPS, LIST_LEN, MAX_PER_PRODUCT, TRAIN_CUTOFFS, AGE_EDGES, StyleModel,
                          age_popularity, build_pairs, features_for, fit_ranker, load_catalog, metrics, prepare_cutoff, recommend, score)
import viz

EVAL = {"validation": VALIDATION_CUTOFF, "holdout": HOLDOUT_CUTOFF}
TREND_POOL = 100            # best sellers kept per list; more than enough for a list of 12
NEVER_DAYS = 1000.0         # stands for "never bought" in the two day-count features of the linear and Bayes models
DAY_COUNTS = ["d_item", "d_code"]
OUR_MODEL = "Our model (two layers, gradient boosting)"


# ----------------------------------------------------------------------------- simple rules
def show(ranked, code_of: np.ndarray) -> list[int]:
    """The first LIST_LEN products of a ranked sequence, at most MAX_PER_PRODUCT colours of one product."""
    out, per_code = [], {}
    for it in ranked:
        c = code_of[it]
        if per_code.get(c, 0) >= MAX_PER_PRODUCT:
            continue
        per_code[c] = per_code.get(c, 0) + 1
        out.append(int(it))
        if len(out) == LIST_LEN:
            break
    return out


def simple_rules(con, cutoff: str, ctx: dict, cat: pd.DataFrame) -> dict[str, dict[int, list[int]]]:
    """Recommendations of the simple rules for the customers of one cutoff."""
    row_of, code_of = pd.Index(cat.article_id), cat.code_idx.to_numpy()
    frame, stats, active = ctx["frame"], ctx["stats"], ctx["active_rows"]
    hist = ctx["hist"].sort_values(["u", "w", "row"], ascending=[True, False, True])        # w = count x recency, as in the re-buy source
    own = {u: (g.row.to_numpy(), g.w.to_numpy()) for u, g in hist.groupby("u")}
    trending = active[np.lexsort((active, -stats.sold7.to_numpy()))][:TREND_POOL]
    ever = con.sql(f"""SELECT article_id, count(*) n FROM transactions_train WHERE t_dat < DATE '{cutoff}'
                       GROUP BY 1 ORDER BY n DESC, 1 LIMIT {TREND_POOL}""").df()
    popular = row_of.get_indexer(ever.article_id)
    by_age = age_popularity(con, cutoff, row_of, BEST_SELLER_DAYS, TREND_POOL)
    by_age = {b: [it for it, _ in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))] for b, d in by_age.items()}
    bucket = np.where(frame.age.notna(), np.digitize(frame.age.fillna(0).to_numpy(dtype=float), AGE_EDGES), -1)

    def co_purchase(u: int) -> list[int]:
        items, w = own[u]
        mine, total = set(items.tolist()), {}
        for r, wr in zip(items.tolist(), w.tolist()):
            nb = ctx["neighbours"].get(r)
            if nb is not None:
                for j, v in zip(nb[0].tolist(), nb[1].tolist()):
                    if j not in mine:
                        total[j] = total.get(j, 0.0) + wr * v
        return sorted(total, key=lambda j: (-total[j], j))

    users = range(len(frame))
    return {"Popularity (all time)": {u: show(popular, code_of) for u in users},
            "Trending (last 7 days)": {u: show(trending, code_of) for u in users},
            "Trending in the age group": {u: show(by_age.get(int(bucket[u]), trending), code_of) for u in users},
            "Co-purchase (item neighbours)": {u: show(co_purchase(u), code_of) for u in users},
            "Re-buy": {u: show(own[u][0], code_of) for u in users}}


# ----------------------------------------------------------------------------- classifiers from the course
def prepared(cols: list[str], one_hot: bool) -> ColumnTransformer:
    """Fill missing values (a day count that is missing means 'never bought'), scale numbers, one-hot encode the two product codes."""
    codes = [i for i, c in enumerate(cols) if c in CATEGORICAL]
    days = [i for i, c in enumerate(cols) if c in DAY_COUNTS]
    rest = [i for i, c in enumerate(cols) if c not in CATEGORICAL + DAY_COUNTS]
    parts = [("numbers", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), rest)]
    if days:
        parts.append(("days", make_pipeline(SimpleImputer(strategy="constant", fill_value=NEVER_DAYS), StandardScaler()), days))
    parts.append(("codes", OneHotEncoder(handle_unknown="ignore", sparse_output=False), codes) if one_hot else ("codes", "passthrough", codes))
    return ColumnTransformer(parts)


def course_classifiers(cols: list[str]) -> dict:
    return {"Naive Bayes": make_pipeline(prepared(cols, one_hot=True), GaussianNB(var_smoothing=1e-3)),
            "Decision tree": make_pipeline(prepared(cols, one_hot=False), DecisionTreeClassifier(max_depth=10, min_samples_leaf=200, random_state=SEED)),
            "Logistic regression": make_pipeline(prepared(cols, one_hot=True), LogisticRegression(C=1.0, max_iter=1000))}


# ----------------------------------------------------------------------------- figure
def figure() -> None:
    """Test-week Hit@10 and MAP@12 of the simple rules, the classifiers with base features, the initial and the final model."""
    viz.style()
    t = pd.read_csv(TABLES / "rq2_baselines.csv").query("split == 'holdout'")
    steps = pd.read_csv(TABLES / "rq2_model_steps.csv").query("split == 'holdout'").iloc[[0]]
    rows = pd.concat([t[t.family == "simple rule"], t[t.family == "classifier, base features"],
                      steps.assign(method="Initial model"), t[t.method == OUR_MODEL].assign(method="Final model")])
    y = np.array([0, 1, 2, 3, 4, 5.6, 6.6, 7.6, 9.2, 10.2])
    colours = [viz.CONTEXT] * 5 + [viz.CONTEXT_DARK] * 3 + [viz.SEQ[3], viz.HIGHLIGHT]
    fig, axes = viz.plt.subplots(1, 2, figsize=(viz.FULL, 2.75), sharey=True, gridspec_kw=dict(wspace=0.1))
    for ax, m, title, text in ((axes[0], "hit10", "Hit@10", lambda v: f"{v:.1%}"), (axes[1], "map12", "MAP@12", lambda v: f"{v:.4f}")):
        ax.barh(y, rows[m], color=colours, height=0.68)
        ax.set_xlim(0, rows[m].max() * 1.25)
        viz.bar_labels(ax, y, rows[m], [text(v) for v in rows[m]])
        ax.set_title(title, fontsize=7.6); ax.grid(axis="y", visible=False); ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
    axes[0].set_yticks(y); axes[0].set_yticklabels(rows.method); axes[0].invert_yaxis()
    for yy, label in ((-0.85, "Simple rules"), (4.8, "Classifiers of the course"), (8.4, "Our model")):
        axes[0].text(0, yy, label, fontsize=6.6, color=viz.TEXT2, style="italic", va="center")
    axes[0].set_ylim(10.8, -1.3)
    fig.suptitle("Recommendation performance of the baselines and the proposed model (test week)", x=0.02, y=1.04, ha="left", fontsize=9.6, fontweight="bold", color=viz.TEXT)
    viz.save(fig, "rq2_baselines")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod", type=int, default=20)
    parser.add_argument("--k-style", type=int, default=6)
    parser.add_argument("--figure", action="store_true", help="only draw the figure from the saved tables")
    args = parser.parse_args()
    if args.figure:
        figure()
        raise SystemExit
    pd.set_option("display.width", 200, "display.max_columns", 20)
    con = connect()
    base_views(con)
    sample_events(con, args.mod)
    groups = garment_groups(con)
    cat = load_catalog(con, groups)
    code_of = cat.code_idx.to_numpy()
    tfidf = sparse.load_npz(CACHE / "rq3_tfidf_all_fields.npz").tocsr()
    style = StyleModel(args.k_style)
    rng = np.random.default_rng(SEED + 1)                  # the generator of the final model in rq2_products, so the pairs are the same

    # 1. candidate pairs at every cutoff, as in rq2_products; the rules are scored on the two evaluation weeks
    rows, train, evals = [], [], {}
    for c in TRAIN_CUTOFFS + list(EVAL.values()):
        start = time.perf_counter()
        ctx = prepare_cutoff(con, c, cat, groups, tfidf, style, fit_style=c == TRAIN_CUTOFFS[0])
        p = build_pairs(ctx, cat, tfidf, sample=c in TRAIN_CUTOFFS, rng=rng)
        if c in TRAIN_CUTOFFS:
            train.append(p)
        else:
            split = "validation" if c == VALIDATION_CUTOFF else "holdout"
            evals[split] = (p, ctx["truth"])
            for name, recs in simple_rules(con, c, ctx, cat).items():
                rows.append(dict(family="simple rule", method=name, features=0, split=split, **metrics(recs, ctx["truth"])))
        print(f"{c}: {len(p):,} pairs ({time.perf_counter() - start:.0f}s)", flush=True)
    train = pd.concat(train, ignore_index=True)
    print(f"{len(train):,} training pairs, {int(train.label.sum()):,} of them bought")
    print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)

    # 2. classifiers from the course and our model on the same pairs
    def evaluate_model(family, name, cols, model, predict):
        for split, (p, truth) in evals.items():
            rows.append(dict(family=family, method=name, features=len(cols), split=split, **metrics(recommend(p, predict(model, p), code_of), truth)))
        v, h = rows[-2], rows[-1]
        print(f"{name:44s} {len(cols):2d} features  val map@12 {v['map12']:.4f}  test recall@10 {h['recall10']:.4f}  hit@10 {h['hit10']:.4f}  "
              f"map@12 {h['map12']:.4f}", flush=True)

    y = train.label.to_numpy()
    for use, family in ((["base"], "classifier, base features"), (list(FEATURE_GROUPS), "classifier, all features")):
        cols = features_for(use)
        for name, model in course_classifiers(cols).items():
            model.fit(train[cols].to_numpy(dtype=np.float32), y)
            evaluate_model(family, name, cols, model, lambda m, p: m.predict_proba(p[cols].to_numpy(dtype=np.float32))[:, 1].astype(np.float32))
        evaluate_model("our method, base features" if use == ["base"] else "our model", "Gradient boosting" if use == ["base"] else OUR_MODEL,
                       cols, fit_ranker(train, cols), lambda m, p: score(m, p, cols))

    table = pd.DataFrame(rows)
    save_table(table, "rq2_baselines")
    saved = pd.read_csv(TABLES / "rq2_product_metrics.csv").query("method == 'Ranker (all RQ features)' and split == 'holdout'").map12.iloc[0]
    ours = table[(table.method == OUR_MODEL) & (table.split == "holdout")].map12.iloc[0]
    print(f"our model, test MAP@12: {ours:.6f} here, {saved:.6f} in rq2_product_metrics")
    print(table[table.split == "holdout"].round(4).to_string(index=False))
    figure()
