"""Additional report figures built from saved tables and caches (no model is retrained).

    eda_calendar.png          daily purchases as a calendar heatmap (overview_daily)
    eda_seasonality.png       seasonal index of every garment group by calendar month (overview_monthly_groups)
    eda_lorenz.png            concentration of purchases across customers (overview_purchase_distribution)
    rq3_product_map.png       UMAP map of the product text vectors with three queries and their alternatives (RQ1)
    rq1_cluster_bubbles.png   garment-group share and over-representation for each customer group (RQ2)
    rq4_chord.png             chord diagram of the category co-purchase links, ordered by community (RQ3)
    rq4_rules_scatter.png     support, confidence and lift of all association rules (RQ3)
    rq2_roc_calibration.png   ROC curve and calibration of the category layer in the test week (RQ4)

Usage:
    python src/advanced_figures.py --mod 20      (after the other scripts)
"""
import argparse
import math

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch, PathPatch, Wedge
from matplotlib.path import Path
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import MultiLabelBinarizer, normalize

from common import CACHE, SEED, TABLES, base_views, connect, garment_groups, sample_events
import viz

# diverging scale for "below / above average": blue <- grey -> orange
DIVERGING = LinearSegmentedColormap.from_list("below_above", [viz.CAT[0], "#a9c9ef", "#efefec", "#f5b896", viz.CAT[1]])


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


# ----------------------------------------------------------------------------- exploration
def calendar() -> None:
    d = read("overview_daily"); d["day"] = pd.to_datetime(d.day)
    start = d.day.min() - pd.Timedelta(days=d.day.min().weekday())
    d["week"] = (d.day - start).dt.days // 7
    d["weekday"] = d.day.dt.weekday
    grid = np.full((7, d.week.max() + 1), np.nan)
    grid[d.weekday, d.week] = d.purchases / 1000
    fig, ax = viz.plt.subplots(figsize=(viz.FULL, 1.75))
    im = ax.imshow(grid, aspect="auto", cmap=viz.SEQ_CMAP, vmin=0, vmax=np.nanquantile(grid, 0.99), interpolation="nearest")
    months = pd.date_range(d.day.min().replace(day=1) + pd.offsets.MonthBegin(1), d.day.max(), freq="MS")
    months = [m for m in months if m.month in (1, 4, 7, 10)]
    ticks = [((m - start).days // 7, f"{m:%b}\n{m:%Y}" if m.month == 1 or m == months[0] else f"{m:%b}") for m in months]
    ax.set_xticks([t for t, _ in ticks]); ax.set_xticklabels([l for _, l in ticks], fontsize=6.4)
    ax.set_yticks([0, 3, 6]); ax.set_yticklabels(["Mon", "Thu", "Sun"], fontsize=6.4); ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    top = d.nlargest(3, "purchases")
    for r in top.itertuples():
        ax.annotate(f"{r.day:%d %b %Y}: {r.purchases / 1000:.0f}K", (r.week, r.weekday), xytext=(8, -4 if r.weekday > 3 else 10),
                    textcoords="offset points", fontsize=6.3, color=viz.TEXT, arrowprops=dict(arrowstyle="-", color=viz.TEXT2, linewidth=0.6),
                    path_effects=[pe.withStroke(linewidth=2, foreground=viz.SURFACE)])
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6)
    cb.set_label("thousand purchases", fontsize=6.4)
    viz.titled(ax, "Every day of the two years: summer peaks and single-day sales spikes",
               "Purchases per day, all 31.8 million transactions; one column per week")
    viz.save(fig, "eda_calendar")


def seasonality() -> None:
    m = read("overview_monthly_groups"); m["purchase_month"] = pd.to_datetime(m.purchase_month)
    m = m[(m.purchase_month >= "2018-10-01") & (m.purchase_month < "2020-09-01")]          # full months only
    m["month"] = m.purchase_month.dt.month
    by = m.pivot_table(index="garment_group_name", columns="month", values="purchases", aggfunc="sum").fillna(0)
    share_month = by / by.sum(axis=0)                                  # share of the group within each calendar month
    overall = by.sum(axis=1) / by.values.sum()
    index = share_month.div(overall, axis=0)
    order = overall.sort_values(ascending=False).index
    index, overall = index.loc[order], overall.loc[order]
    fig, (ax, side) = viz.plt.subplots(1, 2, figsize=(viz.FULL, 4.3), gridspec_kw=dict(width_ratios=[4.2, 1], wspace=0.04), sharey=True)
    norm = TwoSlopeNorm(vmin=-1.6, vcenter=0, vmax=1.6)
    im = ax.imshow(np.log2(index.values), cmap=DIVERGING, norm=norm, aspect="auto")
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            v = index.values[i, j]
            if v >= 1.5 or v <= 0.6:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.6, color="white" if abs(math.log2(v)) > 1.1 else viz.TEXT)
    ax.set_xticks(range(12)); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=6.6)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6.8); ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    side.barh(range(len(order)), overall.values, color=viz.SEQ[3], height=0.66)
    for i, v in enumerate(overall.values):
        side.text(v, i, f" {v:.0%}", va="center", fontsize=6.2, color=viz.TEXT2)
    side.set_xlim(0, overall.max() * 1.6); side.set_xticks([]); side.grid(False)
    for spine in side.spines.values():
        spine.set_visible(False)
    side.set_xlabel("share of all purchases", fontsize=6.6)
    cax = fig.add_axes([0.2, -0.02, 0.42, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=np.log2([1 / 3, 0.5, 1, 2, 3]))
    cb.ax.set_xticklabels(["1/3", "1/2", "1", "2", "3"], fontsize=6); cb.outline.set_visible(False)
    cb.set_label("seasonal index: the group's share in that month / its share over the whole year", fontsize=6.4)
    fig.subplots_adjust(bottom=0.12)
    viz.titled(ax, "Seasonal index of garment groups by calendar month",
               "Calendar months October 2018 to August 2020; numbers shown where the index is at least 1.5 or at most 0.6")
    viz.save(fig, "eda_seasonality")


def lorenz() -> None:
    d = read("overview_purchase_distribution").sort_values("purchases", ascending=False)
    cust = np.r_[0, np.cumsum(d.customers.to_numpy()) / d.customers.sum()]
    purch = np.r_[0, np.cumsum((d.customers * d.purchases).to_numpy()) / (d.customers * d.purchases).sum()]
    asc_c, asc_p = 1 - cust[::-1], 1 - purch[::-1]
    gini = 1 - 2 * np.trapezoid(asc_p, asc_c)
    fig, ax = viz.plt.subplots(figsize=(viz.FULL * 0.56, 2.75))
    ax.fill_between(cust, purch, cust, color=viz.HIGHLIGHT, alpha=0.08, linewidth=0)
    ax.plot([0, 1], [0, 1], color=viz.CONTEXT_DARK, linewidth=0.8)
    ax.plot(cust, purch, color=viz.HIGHLIGHT, linewidth=1.8)
    for share in (0.1, 0.2, 0.5):
        y = float(np.interp(share, cust, purch))
        ax.plot(share, y, "o", markersize=4.5, color=viz.HIGHLIGHT, markeredgecolor=viz.SURFACE, markeredgewidth=1.1)
        ax.annotate(f"top {share:.0%} of customers:\n{y:.0%} of purchases", (share, y), xytext=(8, -14), textcoords="offset points",
                    fontsize=6.3, color=viz.TEXT)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fmt = viz.plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel("customers, from the most to the least active"); ax.set_ylabel("share of all purchases")
    ax.text(0.97, 0.06, f"Gini coefficient {gini:.2f}", transform=ax.transAxes, ha="right", fontsize=6.8, color=viz.TEXT2)
    viz.titled(ax, "Purchases are concentrated", "1.36 million buying customers, two years")
    viz.save(fig, "eda_lorenz")
    print(f"    Gini {gini:.3f}; top 10% -> {np.interp(0.1, cust, purch):.3f}, top 20% -> {np.interp(0.2, cust, purch):.3f}")


# ----------------------------------------------------------------------------- RQ1 (text)
def product_map(con) -> None:
    import umap
    art = con.sql("SELECT article_id, prod_name, product_group_name FROM articles ORDER BY article_id").df()
    tfidf = load_npz(CACHE / "rq3_tfidf_all_fields.npz").tocsr()
    Z = normalize(TruncatedSVD(50, random_state=SEED).fit_transform(tfidf))
    alt = read("rq3_alternatives_final")
    queries = ["Knit dress", "Kevin softshell jacket (1)", "Fix it tote bag"]
    row_of = pd.Index(art.article_id)
    special = []
    for q in queries:
        part = alt[alt.query_name == q].sort_values("rank")
        qid = part.query_id.iloc[0]
        special.append((q, row_of.get_loc(str(qid).zfill(10)) if not isinstance(qid, str) else row_of.get_loc(qid),
                        [row_of.get_loc(str(a).zfill(10) if not isinstance(a, str) else a) for a in part.article_id]))
    rng = np.random.default_rng(SEED)
    keep = set(rng.choice(len(art), 9000, replace=False).tolist())
    for _, q, alts in special:
        keep.update([q] + alts)
    idx = np.array(sorted(keep))
    xy = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=SEED).fit_transform(Z[idx])
    pos = {int(i): xy[k] for k, i in enumerate(idx)}
    names = art.product_group_name.value_counts().index[:7].tolist()
    groups = np.where(art.product_group_name.iloc[idx].isin(names), art.product_group_name.iloc[idx], "Other")
    names = names + ["Other"]
    colour = {g: (viz.CAT[k] if g != "Other" else viz.CONTEXT) for k, g in enumerate(names)}
    fig, ax = viz.plt.subplots(figsize=(viz.FULL, 3.9))
    for g in names:
        msk = groups == g
        ax.scatter(xy[msk, 0], xy[msk, 1], s=1.6, color=colour[g], alpha=0.35, edgecolors="none", rasterized=True)
        ax.scatter([], [], s=16, color=colour[g], label=g)
    for name, q, alts in special:
        for a in alts:
            ax.plot([pos[q][0], pos[a][0]], [pos[q][1], pos[a][1]], color=viz.TEXT, linewidth=0.6, zorder=3)
        ax.scatter([pos[a][0] for a in alts], [pos[a][1] for a in alts], s=16, facecolor=viz.SURFACE, edgecolor=viz.TEXT, linewidth=0.9, zorder=4)
        ax.scatter([pos[q][0]], [pos[q][1]], s=46, marker="*", color=viz.TEXT, zorder=5)
        ax.annotate(name.replace(" (1)", ""), pos[q], xytext=(7, 6), textcoords="offset points", fontsize=6.8, color=viz.TEXT, zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.4, foreground=viz.SURFACE)])
    ax.scatter([], [], s=40, marker="*", color=viz.TEXT, label="query product")
    ax.scatter([], [], s=16, facecolor=viz.SURFACE, edgecolor=viz.TEXT, linewidth=0.9, label="its five alternatives")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handletextpad=0.3)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    viz.titled(ax, "UMAP projection of product text vectors with three query products",
               "TF-IDF vectors of 9,000 sampled products, coloured by product group")
    viz.save(fig, "rq3_product_map")


# ----------------------------------------------------------------------------- RQ2 (customer groups)
def cluster_bubbles() -> None:
    shares = read("rq1_cluster_category_shares")
    shares = shares[~shares.garment_group.str.endswith("dept.")].sort_values("overall_share", ascending=False)
    shares = shares[shares.overall_share >= 0.005].reset_index(drop=True)
    profiles = read("rq1_cluster_profiles").set_index("cluster")
    cols = [c for c in shares.columns if c.startswith("cluster_")]
    S = shares[cols].to_numpy(); L = S / shares.overall_share.to_numpy()[:, None]
    fig, ax = viz.plt.subplots(figsize=(viz.FULL * 0.86, 4.6))
    yy, xx = np.meshgrid(np.arange(len(shares)), np.arange(len(cols)), indexing="ij")
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    sc = ax.scatter(xx.ravel(), yy.ravel(), s=900 * S.ravel(), c=np.log2(np.clip(L.ravel(), 0.25, 4)), cmap=DIVERGING, norm=norm,
                    edgecolor=viz.SURFACE, linewidth=0.6)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if L[i, j] >= 2:
                ax.annotate(f"×{L[i, j]:.1f}", (j, i), xytext=(math.sqrt(900 * S[i, j]) / 2 + 2, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=6.0, color=viz.TEXT,
                            path_effects=[pe.withStroke(linewidth=2, foreground=viz.SURFACE)])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"C{int(c.split('_')[1])}\n" + ", ".join(profiles.loc[int(c.split('_')[1]), "name"].split(", ")[:2]) for c in cols], fontsize=6.4)
    ax.set_yticks(range(len(shares))); ax.set_yticklabels(shares.garment_group, fontsize=6.8)
    ax.set_xlim(-0.6, len(cols) - 0.2); ax.set_ylim(len(shares) - 0.4, -0.7); ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for s, label in ((0.05, "5%"), (0.2, "20%"), (0.4, "40%")):
        ax.scatter([], [], s=900 * s, color=viz.CONTEXT, label=label)
    leg = ax.legend(title="share of the group's purchases", loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3,
                    title_fontsize=6.4, labelspacing=1.2, borderpad=0.8)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, ticks=[-2, -1, 0, 1, 2])
    cb.ax.set_yticklabels(["1/4", "1/2", "1", "2", "4"], fontsize=6); cb.outline.set_visible(False)
    cb.set_label("times the average regular customer", fontsize=6.4)
    viz.titled(ax, "Each customer group is defined by one or two garment groups",
               "Bubble area: share of the group's purchases; colour: compared with all regular customers")
    viz.save(fig, "rq1_cluster_bubbles")


# ----------------------------------------------------------------------------- RQ3 (purchase patterns)
def chord() -> None:
    from rq4_association import EDGE_COUNT, EDGE_LIFT, EDGE_SUPPORT
    edges = read("rq4_pair_edges")
    edges = edges[(edges.lift >= EDGE_LIFT) & (edges.support >= EDGE_SUPPORT) & (edges.co_count >= EDGE_COUNT)]
    nodes = read("rq4_network_nodes")
    nodes = nodes[nodes.category.isin(set(edges.category_a) | set(edges.category_b))]
    flow = pd.concat([edges.groupby("category_a").co_count.sum(), edges.groupby("category_b").co_count.sum()], axis=1).fillna(0).sum(axis=1)
    nodes = nodes.assign(flow=nodes.category.map(flow)).sort_values(["community", "flow"], ascending=[True, False]).reset_index(drop=True)
    communities = list(dict.fromkeys(nodes.community))
    colour = {c: viz.CAT[k % len(viz.CAT)] for k, c in enumerate(communities)}
    gap_node, gap_comm, min_arc = 0.012, 0.06, 0.075
    total = 2 * math.pi - gap_node * len(nodes) - gap_comm * len(communities)
    scale = total / nodes.flow.sum()
    for _ in range(20):                                               # small groups get a minimum arc so their labels do not overlap
        arcs = np.maximum(nodes.flow * scale, min_arc)
        scale *= total / arcs.sum()
    arc_of = dict(zip(nodes.category, np.maximum(nodes.flow * scale, min_arc)))
    link_scale = {n: arc_of[n] / f for n, f in zip(nodes.category, nodes.flow)}
    start, theta, prev = {}, math.pi / 2, None
    for r in nodes.itertuples():
        theta -= gap_comm if r.community != prev else gap_node
        start[r.category] = (theta - arc_of[r.category], theta)       # clockwise arcs
        theta -= arc_of[r.category]; prev = r.community
    order = {n: i for i, n in enumerate(nodes.category)}
    cursor = {n: start[n][1] for n in nodes.category}
    seg = {}
    for n in nodes.category:                                          # sub-arc per link, in the order of the partner
        mine = edges[(edges.category_a == n) | (edges.category_b == n)].copy()
        mine["other"] = np.where(mine.category_a == n, mine.category_b, mine.category_a)
        mine["k"] = mine.other.map(lambda o: (order[o] - order[n]) % len(order))
        for e in mine.sort_values("k").itertuples():
            w = e.co_count * link_scale[n]
            seg[(n, e.other)] = (cursor[n] - w, cursor[n]); cursor[n] -= w
    comm = dict(zip(nodes.category, nodes.community))
    fig, ax = viz.plt.subplots(figsize=(viz.FULL * 0.86, viz.FULL * 0.86))
    R = 1.0
    arc = lambda a0, a1, n=24: [(R * math.cos(t), R * math.sin(t)) for t in np.linspace(a0, a1, n)]
    for e in edges.sort_values("co_count").itertuples():
        a0, a1 = seg[(e.category_a, e.category_b)]; b0, b1 = seg[(e.category_b, e.category_a)]
        pts = arc(a0, a1) + [(0, 0), (R * math.cos(b0), R * math.sin(b0))] + arc(b0, b1)[1:] + [(0, 0), (R * math.cos(a0), R * math.sin(a0))]
        codes = [Path.MOVETO] + [Path.LINETO] * 23 + [Path.CURVE3, Path.CURVE3] + [Path.LINETO] * 23 + [Path.CURVE3, Path.CURVE3]
        same = comm[e.category_a] == comm[e.category_b]
        ax.add_patch(PathPatch(Path(pts, codes), facecolor=colour[comm[e.category_a]] if same else viz.CONTEXT,
                               alpha=0.62 if same else 0.32, edgecolor=viz.SURFACE, linewidth=0.3, zorder=2 if same else 1))
    for r in nodes.itertuples():
        a0, a1 = start[r.category]
        ax.add_patch(Wedge((0, 0), 1.07, math.degrees(a0), math.degrees(a1), width=0.06, color=colour[r.community], linewidth=0))
        mid = (a0 + a1) / 2; deg = math.degrees(mid) % 360
        flip = 90 < deg < 270
        ax.text(1.11 * math.cos(mid), 1.11 * math.sin(mid), r.category, rotation=deg + 180 if flip else deg, rotation_mode="anchor",
                ha="right" if flip else "left", va="center", fontsize=6.4, color=viz.TEXT)
    for c in communities:
        members = [n for n in nodes.category if comm[n] == c and n not in ("Unknown", "Special Offers")][:3]
        ax.scatter([], [], s=24, color=colour[c], label=", ".join(members))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, columnspacing=1.2, handletextpad=0.3)
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.75, 1.75); ax.set_aspect("equal"); ax.axis("off")
    viz.titled(ax, "Garment groups bought together, ordered by community",
               f"{len(edges)} links with lift ≥ {EDGE_LIFT} and ≥ {EDGE_COUNT} shared baskets; ribbon width = shared baskets; grey ribbons join communities")
    viz.save(fig, "rq4_chord")


def rules_scatter() -> None:
    r = read("rq4_rules")
    r["items"] = r.antecedents.str.count(r"\+") + r.consequents.str.count(r"\+") + 2
    fig, ax = viz.plt.subplots(figsize=(viz.FULL, 2.9))
    sc = ax.scatter(r.support, r.confidence, s=[10, 10, 18, 30][0] + 8 * (r["items"] - 2), c=r.lift, cmap=viz.SEQ_CMAP,
                    vmin=1, vmax=r.lift.max(), edgecolor=viz.CONTEXT_DARK, linewidth=0.3, alpha=0.9)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([0.005, 0.01, 0.02, 0.05, 0.1])); ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:.1%}" if v < 0.01 else f"{v:.0%}"))
    ax.yaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    seen = set()
    for x in r.sort_values("lift", ascending=False).itertuples():
        key = frozenset([x.antecedents, x.consequents])
        if key in seen or x.confidence < 0.2:
            continue
        seen.add(key)
        ax.annotate(f"{x.antecedents} → {x.consequents}  (lift {x.lift:.2f})", (x.support, x.confidence),
                    xytext=(0.6, 0.93 - 0.085 * len(seen)), textcoords="axes fraction", fontsize=6.2, color=viz.TEXT,
                    arrowprops=dict(arrowstyle="-", color=viz.TEXT2, linewidth=0.5, shrinkB=3),
                    path_effects=[pe.withStroke(linewidth=2, foreground=viz.SURFACE)])
        if len(seen) == 4:
            break
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6)
    cb.set_label("lift", fontsize=6.6)
    ax.set_xlabel("support: share of baskets containing all groups of the rule (log scale)"); ax.set_ylabel("confidence")
    viz.titled(ax, f"All {len(r)} rules: strong lift comes with low support",
               "Each dot is one rule; larger dots contain three garment groups; labelled: highest lift with confidence ≥ 20%")
    viz.save(fig, "rq4_rules_scatter")


# ----------------------------------------------------------------------------- RQ4 (category layer)
def roc_calibration(con) -> None:
    import rq2_categories as cat
    groups = garment_groups(con)
    f = cat.build_frame(con, cat.HOLDOUT_CUTOFF, groups)
    f = f[f.targets.notna()].reset_index(drop=True)
    Y = MultiLabelBinarizer(classes=groups).fit_transform(f.targets)
    proba = pd.read_parquet(CACHE / f"rq2_category_proba_{cat.HOLDOUT_CUTOFF}.parquet").set_index("customer_id").loc[f.customer_id, groups].to_numpy()
    base = cat.baseline_scores("HistoryModal365", f, groups, groups)
    fig, (a1, a2) = viz.plt.subplots(1, 2, figsize=(viz.FULL, 2.75), gridspec_kw=dict(wspace=0.32))
    for scores, name, colour, lw in ((base, "Customer's most frequent groups", viz.CONTEXT_DARK, 1.3), (proba, "Gradient boosting", viz.HIGHLIGHT, 1.8)):
        fpr, tpr, _ = roc_curve(Y.ravel(), scores.ravel())
        auc = roc_auc_score(Y.ravel(), scores.ravel())
        a1.plot(fpr, tpr, color=colour, linewidth=lw, label=f"{name}  AUC {auc:.3f}")
        print(f"    {name}: micro AUC {auc:.3f}")
    a1.plot([0, 1], [0, 1], color=viz.GRID, linewidth=0.8)
    a1.set_xlabel("false positive rate"); a1.set_ylabel("true positive rate"); a1.set_xlim(0, 1); a1.set_ylim(0, 1.01)
    a1.legend(loc="lower right", bbox_to_anchor=(1.02, 0.02), fontsize=6.6)
    viz.titled(a1, "ROC: the model ranks groups better", "All customer and garment-group pairs, test week")
    p, y = proba.ravel(), Y.ravel()
    edges = np.quantile(p, np.linspace(0, 1, 13))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 11)
    mp = np.array([p[b == k].mean() for k in range(12)]); oy = np.array([y[b == k].mean() for k in range(12)])
    top = max(mp.max(), oy.max()) * 1.08
    a2.plot([0, top], [0, top], color=viz.GRID, linewidth=0.8)
    a2.plot(mp, oy, color=viz.HIGHLIGHT, linewidth=1.6, marker="o", markersize=4, markeredgecolor=viz.SURFACE, markeredgewidth=1)
    a2.set_xlim(0, top); a2.set_ylim(0, top)
    fmt = viz.plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    a2.xaxis.set_major_formatter(fmt); a2.yaxis.set_major_formatter(fmt)
    a2.set_xlabel("predicted probability (12 equal-size bins)"); a2.set_ylabel("observed share bought")
    viz.titled(a2, "Calibration: probabilities are close to reality", "Gradient boosting; diagonal = perfect calibration")
    viz.save(fig, "rq2_roc_calibration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod", type=int, default=20)
    args = parser.parse_args()
    viz.style()
    calendar(); seasonality(); lorenz()
    cluster_bubbles(); chord(); rules_scatter()
    con = connect(); base_views(con); sample_events(con, args.mod)
    product_map(con)
    roc_calibration(con)
