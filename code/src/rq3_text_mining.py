"""RQ3: How can I find similar alternatives?

Every article's name, description, product type and garment group are turned into a
TF-IDF vector; alternatives are the most cosine-similar articles. Three rules keep the
results useful for a shopper:

* one result per product code: colour variants of the same garment are collapsed,
  and the query's own product code is excluded;
* same department: alternatives come from the same index group (Ladieswear,
  Menswear, Baby/Children, ...);
* attributes count: the final score is 0.7 x text similarity + 0.3 x the share of
  three catalogue attributes the candidate has in common with the query
  (product type, pattern, colour lightness), so two dresses with the same wording
  but a different pattern no longer tie.

Usage:
    python src/rq3_text_mining.py
"""
import argparse
import re

import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from common import CACHE, DEPARTMENT_SQL, base_views, connect, save_table
import viz

VARIANTS = {
    "all_fields": ["prod_name", "detail_desc", "product_type_name", "garment_group_name"],
    "name_description": ["prod_name", "detail_desc"],
}
TEXT_WEIGHT = 0.7
ATTRIBUTES = ["product_type_name", "graphical_appearance_name", "perceived_colour_value_name"]
MODES = [  # name, collapse colour variants / same names, same department only, attribute term
    ("text_only_naive", False, False, False),
    ("text_only_deduped", True, True, False),
    ("text_plus_attributes", True, True, True),
]


def clean(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(text).lower())


def load_articles(con) -> pd.DataFrame:
    # department (same rule as RQ1 and RQ2): a shopper looking at a women's dress should not be offered a boys' item
    df = con.sql(f"""
        SELECT article_id, product_code, prod_name, product_type_name, garment_group_name,
               index_group_name, colour_group_name, graphical_appearance_name,
               perceived_colour_value_name, coalesce(detail_desc, '') AS detail_desc,
               {DEPARTMENT_SQL} AS department
        FROM articles a ORDER BY article_id""").df()
    # Key used to collapse near-duplicates: same product code, or the same name in a different code.
    df["name_key"] = df.prod_name.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    return df


def pick_queries(df: pd.DataFrame, n_types: int) -> list[int]:
    """One query per common product type: the first article with a real description."""
    queries = []
    for typ in df.product_type_name.value_counts().index[:n_types]:
        part = df[(df.product_type_name == typ) & (df.detail_desc.str.len() > 20)]
        queries.append(int(part.index[0]) if len(part) else int(df[df.product_type_name == typ].index[0]))
    return queries


def search(matrix, terms, df: pd.DataFrame, attrs: list[np.ndarray], i: int, k: int,
           dedupe: bool, same_department: bool, attributes: bool) -> list[dict]:
    """Top-k alternatives for article row i."""
    cosine = np.clip((matrix @ matrix[i].T).toarray().ravel(), 0, 1)
    agree = np.zeros(len(df), dtype=np.float32)
    for codes in attrs:
        agree += codes == codes[i]
    agree /= len(attrs)
    scores = TEXT_WEIGHT * cosine + (1 - TEXT_WEIGHT) * agree if attributes else cosine.copy()
    scores[i] = -1
    if dedupe:
        scores[(df.product_code == df.product_code.iat[i]).to_numpy()] = -1
        scores[(df.name_key == df.name_key.iat[i]).to_numpy()] = -1
    if same_department:
        scores[(df.department != df.department.iat[i]).to_numpy()] = -1
    results, seen = [], set()
    for j in np.argsort(-scores, kind="stable")[:300]:
        if scores[j] < 0:
            break
        code, name_key = df.product_code.iat[j], df.name_key.iat[j]
        if dedupe and (code in seen or name_key in seen):
            continue
        seen.update({code, name_key})
        shared = np.intersect1d(matrix[i].indices, matrix[j].indices)
        shared = sorted(shared, key=lambda t: float(matrix[i, t] * matrix[j, t]), reverse=True)[:8]
        results.append(dict(
            rank=len(results) + 1, article_id=df.article_id.iat[j], product_code=code, prod_name=df.prod_name.iat[j],
            product_type=df.product_type_name.iat[j], garment_group=df.garment_group_name.iat[j],
            colour=df.colour_group_name.iat[j], pattern=df.graphical_appearance_name.iat[j],
            score=float(scores[j]), similarity=float(cosine[j]), attributes_in_common=int(round(agree[j] * len(attrs))),
            same_product_type=df.product_type_name.iat[j] == df.product_type_name.iat[i],
            same_garment_group=df.garment_group_name.iat[j] == df.garment_group_name.iat[i],
            same_pattern=df.graphical_appearance_name.iat[j] == df.graphical_appearance_name.iat[i],
            same_colour_value=df.perceived_colour_value_name.iat[j] == df.perceived_colour_value_name.iat[i],
            identical_description=df.detail_desc.iat[j] == df.detail_desc.iat[i],
            shared_terms="; ".join(terms[shared]),
        ))
        if len(results) == k:
            break
    return results


def example_figure(results: pd.DataFrame, queries: pd.DataFrame, names: list[str]) -> None:
    viz.style()
    chosen = [q for q in names if q in set(queries.prod_name)][:3]
    fig, axes = viz.plt.subplots(len(chosen), 1, figsize=(viz.FULL, 1.55 * len(chosen) + 0.3), sharex=True,
                                 gridspec_kw=dict(hspace=0.75))
    axes = np.atleast_1d(axes)
    for ax, qname in zip(axes, chosen):
        q = queries[queries.prod_name == qname].iloc[0]
        part = results[(results.variant == "all_fields") & (results.query_id == q.article_id)].sort_values("rank")
        y = np.arange(len(part))
        ax.barh(y, part.score, color=viz.HIGHLIGHT, height=0.62)
        ax.barh(y, (1 - TEXT_WEIGHT) * part.attributes_in_common / len(ATTRIBUTES), left=TEXT_WEIGHT * part.similarity,
                color=viz.SEQ[1], height=0.62)                      # the attribute part of the score, lighter
        viz.bar_labels(ax, y, part.score, [f"{r.score:.2f}   shared: {r.shared_terms.split('; ')[0]}" for r in part.itertuples()])
        ax.set_yticks(y); ax.set_yticklabels([f"{r.prod_name}" for r in part.itertuples()])
        ax.invert_yaxis(); ax.set_xlim(0, 1.0); ax.grid(axis="y", visible=False)
        desc = q.detail_desc if len(q.detail_desc) <= 88 else q.detail_desc[:85].rsplit(" ", 1)[0] + " ..."
        viz.titled(ax, f"Query: {q.prod_name}  ({q.product_type_name}, {q.graphical_appearance_name.lower()})", desc)
    axes[-1].set_xlabel("score = 0.7 x text similarity (dark) + 0.3 x share of matching attributes (light)")
    viz.save(fig, "rq3_examples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-types", type=int, default=30, help="number of query product types")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    con = connect()
    base_views(con)
    df = load_articles(con)
    queries = pick_queries(df, args.n_types)
    save_table(df.loc[queries, ["article_id", "product_code", "prod_name", "product_type_name", "garment_group_name",
                                "graphical_appearance_name", "perceived_colour_value_name", "department", "detail_desc"]], "rq3_queries")
    print(f"{len(df):,} articles, {len(queries)} queries")
    print("departments:", df.department.value_counts().to_dict())
    attrs = [df[a].fillna("").astype("category").cat.codes.to_numpy() for a in ATTRIBUTES]

    all_rows, summary = [], []
    for variant, fields in VARIANTS.items():
        texts = df[fields].agg(" ".join, axis=1).map(clean)
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_features=20000, dtype=np.float32)
        matrix = vec.fit_transform(texts)
        terms = vec.get_feature_names_out()
        if variant == "all_fields":                      # reused by RQ1 (style clusters) and RQ2 (similar products)
            save_npz(CACHE / "rq3_tfidf_all_fields.npz", matrix)
            (CACHE / "rq3_tfidf_terms.txt").write_text("\n".join(terms), encoding="utf8")
        for mode, dedupe, dept, attributes in MODES:
            rows = []
            for i in queries:
                for r in search(matrix, terms, df, attrs, i, args.k, dedupe, dept, attributes):
                    rows.append(dict(variant=variant, mode=mode, query_id=df.article_id.iat[i], query_name=df.prod_name.iat[i],
                                     query_type=df.product_type_name.iat[i], query_code=df.product_code.iat[i], **r))
            part = pd.DataFrame(rows)
            summary.append(dict(
                variant=variant, mode=mode, results=len(part),
                same_product_code_share=(part.product_code == part.query_code).mean(),
                identical_description_share=part.identical_description.mean(),
                same_product_type_share=part.same_product_type.mean(),
                same_garment_group_share=part.same_garment_group.mean(),
                same_pattern_share=part.same_pattern.mean(),
                same_colour_value_share=part.same_colour_value.mean(),
                mean_text_similarity=part.similarity.mean(),
            ))
            all_rows.extend(rows)
            s = summary[-1]
            print(f"{variant:18s} {mode:22s} same-code {s['same_product_code_share']:.0%}  identical-desc {s['identical_description_share']:.0%}  "
                  f"same-type {s['same_product_type_share']:.0%}  same-pattern {s['same_pattern_share']:.0%}  "
                  f"same-lightness {s['same_colour_value_share']:.0%}  text-sim {s['mean_text_similarity']:.2f}")

    results = pd.DataFrame(all_rows)
    save_table(results, "rq3_results")
    save_table(pd.DataFrame(summary), "rq3_summary")
    save_table(results[(results.variant == "all_fields") & (results["mode"] == "text_plus_attributes")]
               .drop(columns=["variant", "mode"]), "rq3_alternatives_final")
    example_figure(results[results["mode"] == "text_plus_attributes"], df.loc[queries], ["Knit dress", "Kevin softshell jacket (1)", "Fix it tote bag"])
