"""RQ4: What other products might interest me based on purchasing patterns?

Approximate baskets: all purchases by one customer on one day (there is no order ID).
Each basket is the set of garment groups it contains. FP-growth finds frequent
category combinations and the rules between them; a category network shows which
groups are bought together more often than chance.

A second view looks across visits instead of within one visit: for consecutive
shopping days of the same customer (at most MAX_GAP_DAYS apart), how likely is garment
group B on the next visit given group A on this one, compared with B's base rate.

Usage:
    python src/rq4_association.py --mod 20
    python src/rq4_association.py --mod 5
"""
import argparse
import math

import networkx as nx
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from sklearn.preprocessing import MultiLabelBinarizer

from common import DATA_END, base_views, connect, garment_groups, sample_events, save_table
import viz

MIN_SUPPORT = 0.005      # a combination must appear in at least 0.5% of baskets
MIN_CONFIDENCE = 0.10
MAX_LEN = 3
EDGE_LIFT, EDGE_SUPPORT, EDGE_COUNT = 1.2, 0.002, 500
MAX_GAP_DAYS = 90        # "next visit" must happen within this many days


def next_visit_transitions(con, groups: list[str]) -> tuple[pd.DataFrame, dict]:
    """P(group B on the next visit | group A on this visit) and its lift over B's base rate."""
    visits = con.sql(f"""
        WITH v AS (SELECT customer_id, t_dat, list(DISTINCT garment_group_name) cats FROM events GROUP BY 1, 2)
        SELECT cats, lead(cats) OVER w AS next_cats, date_diff('day', t_dat, lead(t_dat) OVER w) AS gap_days
        FROM v WINDOW w AS (PARTITION BY customer_id ORDER BY t_dat)
        QUALIFY next_cats IS NOT NULL""").df()
    pairs = visits[visits.gap_days <= MAX_GAP_DAYS]
    mlb = MultiLabelBinarizer(classes=groups)
    V1 = mlb.fit_transform(pairs.cats).astype(np.int32)
    V2 = mlb.transform(pairs.next_cats).astype(np.int32)
    counts = V1.T @ V2                       # pairs with A now and B next
    n_a = V1.sum(axis=0)
    base = V2.mean(axis=0)                   # share of next visits containing B
    rows = []
    for i, a in enumerate(groups):
        for j, b in enumerate(groups):
            p = counts[i, j] / n_a[i] if n_a[i] else 0.0
            rows.append(dict(this_visit=a, next_visit=b, pairs=int(counts[i, j]), visits_with_a=int(n_a[i]),
                             prob_next=p, base_rate=base[j], lift=p / base[j] if base[j] else 0.0, same_group=a == b))
    info = dict(visit_pairs=len(pairs), all_pairs=len(visits), median_gap_days=float(pairs.gap_days.median()),
                share_within_gap=len(pairs) / len(visits))
    return pd.DataFrame(rows), info


def transition_figures(trans: pd.DataFrame, groups: list[str], info: dict) -> None:
    viz.style()
    prob = trans.pivot(index="this_visit", columns="next_visit", values="prob_next").loc[groups, groups]
    fig, ax = viz.plt.subplots(figsize=(viz.FULL, 6.0))
    im = ax.imshow(prob.values, cmap=viz.SEQ_CMAP, vmin=0, vmax=0.4)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, rotation=90, fontsize=6.6)
    ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups, fontsize=6.6); ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(groups)):
        for j in range(len(groups)):
            v = prob.values[i, j]
            if v >= 0.08:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=5.4, color="white" if v > 0.24 else viz.TEXT2)
    ax.set_xlabel("garment group on the next visit"); ax.set_ylabel("garment group on this visit")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.4)
    viz.titled(ax, "What the next visit contains, given this visit",
               f"{info['visit_pairs']:,} consecutive visits at most {MAX_GAP_DAYS} days apart; median gap {info['median_gap_days']:.0f} days")
    viz.save(fig, "rq4_next_visit_heatmap")

    noise = {"Unknown", "Special Offers"}       # catalogue leftovers, kept in the table but not in the chart
    top = trans[(~trans.same_group) & (trans.pairs >= 300) & ~trans.this_visit.isin(noise) & ~trans.next_visit.isin(noise)]
    top = top.sort_values("lift", ascending=False).head(12).reset_index(drop=True)
    fig, ax = viz.plt.subplots(figsize=(viz.FULL, 3.0))
    y = np.arange(len(top))
    ax.barh(y, top.lift - 1, left=1, color=viz.HIGHLIGHT, height=0.6)
    ax.axvline(1, color=viz.CONTEXT_DARK, linewidth=0.8)
    ax.set_xlim(1, 1 + (top.lift.max() - 1) * 1.45)
    viz.bar_labels(ax, y, top.lift, [f"{r.lift:.2f}   {r.prob_next:.1%} of next visits vs {r.base_rate:.1%} overall" for r in top.itertuples()], size=6.6)
    ax.set_yticks(y); ax.set_yticklabels([f"{r.this_visit}  \u2192  {r.next_visit}" for r in top.itertuples()])
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("lift: how much more often the next visit contains the second group (1 = no association)")
    viz.titled(ax, "Summer groups lead the transitions between different garment groups",
               "Strongest next-visit transitions with at least 300 visit pairs")
    viz.save(fig, "rq4_next_visit_top")


def basket_matrix(con, groups: list[str], period: tuple[str, str] | None = None) -> pd.DataFrame:
    where = "" if period is None else f"WHERE t_dat >= DATE '{period[0]}' AND t_dat < DATE '{period[1]}'"
    baskets = con.sql(f"""
        SELECT customer_id, t_dat, list(DISTINCT garment_group_name) AS cats
        FROM events {where} GROUP BY 1, 2""").df()
    mlb = MultiLabelBinarizer(classes=groups)
    matrix = pd.DataFrame(mlb.fit_transform(baskets.cats).astype(bool), columns=groups)
    return matrix


def mine_rules(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frequent = fpgrowth(matrix, min_support=MIN_SUPPORT, use_colnames=True, max_len=MAX_LEN)
    rules = association_rules(frequent, num_itemsets=len(matrix), metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules = rules[rules.lift > 1.0].copy()
    rules["antecedents"] = rules.antecedents.map(lambda s: " + ".join(sorted(s)))
    rules["consequents"] = rules.consequents.map(lambda s: " + ".join(sorted(s)))
    rules["baskets"] = (rules.support * len(matrix)).round().astype(int)
    rules = rules[["antecedents", "consequents", "support", "confidence", "lift", "baskets"]]
    rules = rules.sort_values(["lift", "support", "antecedents", "consequents"], ascending=[False, False, True, True]).reset_index(drop=True)
    frequent["itemsets"] = frequent.itemsets.map(lambda s: " + ".join(sorted(s)))
    frequent["size"] = frequent.itemsets.str.count(r" \+ ") + 1
    return frequent.sort_values(["support", "itemsets"], ascending=[False, True]).reset_index(drop=True), rules   # ties in a fixed order


def pair_edges(matrix: pd.DataFrame) -> pd.DataFrame:
    """Co-occurrence, support, lift and phi for every pair of garment groups."""
    m = matrix.to_numpy(dtype=np.int32)
    n = len(m)
    co = m.T @ m
    counts = np.diag(co)
    rows = []
    for i in range(len(matrix.columns)):
        for j in range(i + 1, len(matrix.columns)):
            pa, pb, pab = counts[i] / n, counts[j] / n, co[i, j] / n
            den = np.sqrt(pa * (1 - pa) * pb * (1 - pb))
            rows.append(dict(category_a=matrix.columns[i], category_b=matrix.columns[j], co_count=int(co[i, j]),
                             support=pab, lift=pab / (pa * pb) if pa * pb else 0.0, phi=(pab - pa * pb) / den if den else 0.0,
                             count_a=int(counts[i]), count_b=int(counts[j])))
    return pd.DataFrame(rows)


def build_network(edges: pd.DataFrame, groups: list[str], purchases: pd.Series) -> tuple[nx.Graph, pd.DataFrame]:
    keep = edges[(edges.lift >= EDGE_LIFT) & (edges.support >= EDGE_SUPPORT) & (edges.co_count >= EDGE_COUNT)]
    g = nx.Graph()
    g.add_nodes_from(groups)
    for e in keep.itertuples():
        g.add_edge(e.category_a, e.category_b, weight=float(e.phi), lift=float(e.lift), co_count=int(e.co_count))
    communities = nx.community.greedy_modularity_communities(g, weight="weight") if g.number_of_edges() else []
    member = {n: i for i, c in enumerate(communities) for n in c}
    nodes = pd.DataFrame([dict(category=n, purchases=int(purchases[n]), degree=g.degree(n),
                               weighted_degree=round(g.degree(n, weight="weight"), 3), community=member.get(n, -1))
                          for n in g.nodes]).sort_values(["community", "degree"], ascending=[True, False])
    return g, nodes


def community_layout(members_of: dict[int, list[str]]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Each community sits in its own quadrant, its members on a small circle around the community centre.
    Returns node positions and, per node, the outward direction used to place its label."""
    centres = [(-1.4, 0.95), (1.4, 0.95), (-1.4, -0.95), (1.4, -0.95)]
    pos, out = {}, {}
    for k, (c, members) in enumerate(members_of.items()):
        centre = np.array(centres[k] if k < 4 else (2.8 * math.cos(k), 2.8 * math.sin(k)))
        radius = 0.28 + 0.07 * len(members)
        for i, n in enumerate(members):
            angle = math.pi / 2 - 2 * math.pi * i / len(members)
            direction = np.array([math.cos(angle), math.sin(angle)])
            pos[n] = centre + (radius * direction * np.array([1.35, 1.0]) if len(members) > 1 else 0)
            out[n] = direction if len(members) > 1 else np.array([0.0, -1.0])
    return pos, out


def figures(rules: pd.DataFrame, g: nx.Graph, nodes: pd.DataFrame, raw_edges: int, sizes: pd.DataFrame) -> None:
    import matplotlib.patheffects as pe
    viz.style()

    # top rules by lift
    top = rules.head(12).reset_index(drop=True)
    fig, ax = viz.plt.subplots(figsize=(viz.FULL, 3.0))
    y = np.arange(len(top))
    ax.barh(y, top.lift - 1, left=1, color=viz.HIGHLIGHT, height=0.6)
    ax.axvline(1, color=viz.CONTEXT_DARK, linewidth=0.8)
    ax.set_xlim(1, 1 + (top.lift.max() - 1) * 1.5)
    viz.bar_labels(ax, y, top.lift, [f"{r.lift:.2f}   confidence {r.confidence:.0%}, {r.baskets:,} baskets" for r in top.itertuples()], size=6.6)
    ax.set_yticks(y); ax.set_yticklabels([f"{r.antecedents}  \u2192  {r.consequents}" for r in top.itertuples()], fontsize=7)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlabel("lift (1 = no association)")
    viz.titled(ax, "The strongest rules describe womenswear outfits", "Same-customer, same-day baskets of garment groups")
    viz.save(fig, "rq4_top_rules")

    # basket size distribution
    fig, ax = viz.plt.subplots(figsize=(viz.FULL * 0.62, 2.1))
    s = sizes[sizes.categories_in_basket <= 8]
    ax.bar(s.categories_in_basket, s.share, color=[viz.HIGHLIGHT] + [viz.SEQ[2]] * (len(s) - 1), width=0.62)
    for x, v in zip(s.categories_in_basket, s.share):
        ax.text(x, v + 0.008, f"{v:.0%}", ha="center", va="bottom", fontsize=6.8, color=viz.TEXT2)
    ax.set_xticks(s.categories_in_basket); ax.set_xlabel("garment groups in the basket"); ax.set_ylabel("share of baskets")
    ax.yaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:.0%}")); ax.grid(axis="x", visible=False)
    viz.titled(ax, "Half of all baskets contain one garment group")
    viz.save(fig, "rq4_basket_sizes")

    # category network: each community in its own region, links inside a community in its colour
    connected = [n for n in g.nodes if g.degree(n) > 0]
    sub = g.subgraph(connected)
    info = nodes.set_index("category")
    comm, size = info.community, info.purchases
    members_of = {c: sorted([n for n in connected if comm[n] == c], key=lambda n: -size[n]) for c in sorted(comm.unique()) if c >= 0}
    pos, out = community_layout(members_of)
    colour = {c: viz.CAT[k % len(viz.CAT)] for k, c in enumerate(members_of)}
    fig, ax = viz.plt.subplots(figsize=(viz.FULL, 4.9))
    for same in (False, True):                       # links between communities first, underneath
        for a, b, d in sub.edges(data=True):
            if (comm[a] == comm[b]) != same:
                continue
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], solid_capstyle="round",
                    linewidth=0.5 + 16 * d["weight"] if same else 0.4 + 6 * d["weight"],
                    color=colour[comm[a]] if same else viz.CONTEXT, alpha=0.75 if same else 0.45, zorder=1 if not same else 2)
    for c, members in members_of.items():
        xy = np.array([pos[n] for n in members])
        ax.scatter(xy[:, 0], xy[:, 1], s=[40 + 360 * size[n] / size.max() for n in members], color=colour[c],
                   edgecolor=viz.SURFACE, linewidth=1.2, zorder=3)
        for n in members:
            r = math.sqrt(40 + 360 * size[n] / size.max()) / 2 + 2.5          # node radius in points, plus a gap
            dx, dy = out[n]
            label = n if len(n) <= 18 else n.replace("/", "/\n", 1).replace("Knitted mix", "Knitted\nmix") if n.startswith("Woven") else n.replace("/", "/\n", 1)
            ax.annotate(label, pos[n], xytext=(dx * r, dy * r), textcoords="offset points", linespacing=0.95,
                        ha="left" if dx > 0.35 else ("right" if dx < -0.35 else "center"),
                        va="bottom" if dy > 0.35 else ("top" if dy < -0.35 else "center"),
                        fontsize=6.6, color=viz.TEXT, zorder=4,
                        path_effects=[pe.withStroke(linewidth=2.2, foreground=viz.SURFACE)])
        named = [n for n in members if n not in ("Unknown", "Special Offers")][:3]
        ax.scatter([], [], s=30, color=colour[c], label=", ".join(named))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, columnspacing=1.6, handletextpad=0.3)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.margins(x=0.16, y=0.1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    isolated = [n for n in g.nodes if g.degree(n) == 0]
    viz.titled(ax, "Co-purchase network of garment groups",
               f"{sub.number_of_edges()} of {raw_edges} pairs kept (lift \u2265 {EDGE_LIFT}, support \u2265 {EDGE_SUPPORT:.1%}, \u2265 {EDGE_COUNT} baskets); "
               f"link width = phi, node size = purchases; grey links join communities" + (f"; unlinked: {', '.join(isolated)}" if isolated else ""))
    viz.save(fig, "rq4_category_network")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod", type=int, default=5)
    args = parser.parse_args()

    con = connect()
    base_views(con)
    sample_events(con, args.mod)
    groups = garment_groups(con)

    matrix = basket_matrix(con, groups)
    sizes = matrix.sum(axis=1).value_counts().sort_index().rename_axis("categories_in_basket").reset_index(name="baskets")
    sizes["share"] = sizes.baskets / sizes.baskets.sum()
    save_table(sizes, "rq4_basket_sizes")
    print(f"{len(matrix):,} baskets; {sizes.share.iloc[0]:.0%} contain a single garment group")

    frequent, rules = mine_rules(matrix)
    save_table(frequent, "rq4_frequent_itemsets")
    save_table(rules, "rq4_rules")
    print(f"{len(frequent)} frequent itemsets, {len(rules)} rules with lift > 1")
    print(rules.head(12).round(3).to_string(index=False))

    edges = pair_edges(matrix)
    save_table(edges.sort_values("phi", ascending=False), "rq4_pair_edges")
    g, nodes = build_network(edges, groups, matrix.sum())
    save_table(nodes, "rq4_network_nodes")
    print(f"network: {g.number_of_edges()} edges kept of {len(edges)} pairs "
          f"({int((edges.co_count >= 100).sum())} pairs co-occur 100+ times); {nodes.community.nunique()} communities")
    print(nodes.to_string(index=False))

    # year-to-year stability of the top rules
    top_all = set(zip(rules.head(20).antecedents, rules.head(20).consequents))
    stability = []
    for label, period in (("2019", ("2019-01-01", "2020-01-01")), ("2020", ("2020-01-01", DATA_END))):
        _, r = mine_rules(basket_matrix(con, groups, period))
        top = set(zip(r.head(20).antecedents, r.head(20).consequents))
        stability.append(dict(period=label, rules=len(r), top20_overlap_with_full=len(top & top_all),
                              top20_jaccard=len(top & top_all) / len(top | top_all)))
    save_table(pd.DataFrame(stability), "rq4_rule_stability")
    print(pd.DataFrame(stability).round(3).to_string(index=False))

    figures(rules, g, nodes, len(edges), sizes)

    # across visits: what comes next
    trans, info = next_visit_transitions(con, groups)
    save_table(trans, "rq4_next_visit_transitions")
    print(f"next-visit: {info['visit_pairs']:,} visit pairs within {MAX_GAP_DAYS} days "
          f"({info['share_within_gap']:.0%} of all consecutive visits), median gap {info['median_gap_days']:.0f} days")
    same = trans[trans.same_group].sort_values("prob_next", ascending=False)
    print("repeat the same group next visit (top 5):")
    print(same[["this_visit", "prob_next", "base_rate", "lift"]].head(5).round(3).to_string(index=False))
    print("strongest transitions between different groups (>= 300 pairs):")
    print(trans[(~trans.same_group) & (trans.pairs >= 300)].sort_values("lift", ascending=False)
          [["this_visit", "next_visit", "pairs", "prob_next", "base_rate", "lift"]].head(12).round(3).to_string(index=False))
    transition_figures(trans, groups, info)
