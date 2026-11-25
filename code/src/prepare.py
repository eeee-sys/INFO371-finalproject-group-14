"""Step 0: convert the raw CSV files to Parquet, profile them, and draw a customer sample.

Usage:
    python src/prepare.py --mod 20     # quick test: 1 customer in 20
    python src/prepare.py --mod 5      # main experiment: 1 customer in 5

The Parquet conversion runs once; later calls only build the sample.
"""
import argparse
import time

import pandas as pd

from common import CACHE, RAW, SEED, base_views, connect, save_table


def convert_to_parquet(con) -> None:
    """Read each CSV as text (so IDs keep leading zeros) and store a typed Parquet copy."""
    casts = {
        "articles": "*",
        "customers": "*",
        "transactions_train": (
            "* REPLACE (CAST(t_dat AS DATE) AS t_dat, CAST(price AS DOUBLE) AS price, "
            "CAST(sales_channel_id AS INTEGER) AS sales_channel_id)"
        ),
    }
    for name, select in casts.items():
        target = CACHE / f"{name}.parquet"
        if target.exists():
            print(f"parquet exists: {target.name}")
            continue
        start = time.perf_counter()
        source = (RAW / f"{name}.csv").as_posix()
        con.execute(
            f"COPY (SELECT {select} FROM read_csv('{source}', all_varchar=true, header=true)) "
            f"TO '{target.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        print(f"converted {name}.csv in {time.perf_counter() - start:.0f}s")


def profile(con) -> None:
    """Full-data statistics for the Data Overview section."""
    counts = []
    for name in ("articles", "customers", "transactions_train"):
        n = con.sql(f"SELECT count(*) FROM {name}").fetchall()[0][0]
        cols = ", ".join(f'"{c}"' for c in con.sql(f"DESCRIBE {name}").df().column_name)
        dup = con.sql(
            f"SELECT coalesce(sum(n - 1), 0) FROM (SELECT count(*) n FROM {name} GROUP BY {cols} HAVING count(*) > 1)"
        ).fetchall()[0][0]
        counts.append(dict(table=name, rows=n, exact_duplicate_rows=int(dup)))
    save_table(pd.DataFrame(counts), "overview_counts")

    save_table(con.sql("""
        SELECT min(t_dat) first_date, max(t_dat) last_date,
               count(DISTINCT customer_id) buying_customers,
               count(DISTINCT article_id) sold_articles,
               avg(CASE WHEN sales_channel_id = 2 THEN 1 ELSE 0 END) online_share
        FROM transactions_train""").df(), "overview_dates")

    save_table(con.sql("""
        SELECT date_trunc('month', t_dat) AS purchase_month, count(*) AS purchases
        FROM transactions_train GROUP BY 1 ORDER BY 1""").df(), "overview_monthly")

    save_table(con.sql("""
        SELECT a.garment_group_name, count(*) purchases, count(DISTINCT t.article_id) articles
        FROM transactions_train t JOIN articles a USING (article_id)
        GROUP BY 1 ORDER BY 2 DESC""").df(), "overview_garment_groups")

    save_table(con.sql("""
        SELECT quantile_cont(n, [0.25, 0.5, 0.75, 0.9, 0.99]) AS purchases_quantiles,
               avg(n) mean_purchases, max(n) max_purchases
        FROM (SELECT customer_id, count(*) n FROM transactions_train GROUP BY 1)""").df(),
        "overview_customer_activity")

    # for the report figures: daily purchases, monthly purchases by garment group, purchases per customer
    save_table(con.sql("SELECT t_dat AS day, count(*) AS purchases FROM transactions_train GROUP BY 1 ORDER BY 1").df(), "overview_daily")
    save_table(con.sql("""
        SELECT date_trunc('month', t.t_dat) AS purchase_month, a.garment_group_name, count(*) AS purchases
        FROM transactions_train t JOIN articles a USING (article_id) GROUP BY 1, 2 ORDER BY 1, 2""").df(), "overview_monthly_groups")
    save_table(con.sql("""
        SELECT n AS purchases, count(*) AS customers
        FROM (SELECT customer_id, count(*) n FROM transactions_train GROUP BY 1) GROUP BY 1 ORDER BY 1""").df(),
        "overview_purchase_distribution")

    missing = []
    for name in ("articles", "customers"):
        total = con.sql(f"SELECT count(*) FROM {name}").fetchall()[0][0]
        for col in con.sql(f"DESCRIBE {name}").df().column_name:
            m = con.sql(f"""SELECT count(*) FROM {name} WHERE "{col}" IS NULL OR CAST("{col}" AS VARCHAR) = ''""").fetchall()[0][0]
            if m:
                missing.append(dict(table=name, field=col, missing=m, share=round(m / total, 4)))
    save_table(pd.DataFrame(missing), "overview_missing")


def build_sample(con, mod: int) -> None:
    """Keep every purchase of one customer in `mod`, chosen by a seeded hash of the customer ID."""
    target = CACHE / f"events_mod{mod}.parquet"
    if target.exists():
        print(f"sample exists: {target.name}")
    else:
        con.execute(f"""
            COPY (
                SELECT t.t_dat, t.customer_id, t.article_id, t.price, t.sales_channel_id,
                       a.product_code, a.garment_group_name, a.product_type_name
                FROM transactions_train t JOIN articles a USING (article_id)
                WHERE hash(t.customer_id || '{SEED}') % {mod} = 0
            ) TO '{target.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    scope = con.sql(f"""
        SELECT {mod} AS modulus, count(*) AS n_rows, count(DISTINCT customer_id) AS customers,
               min(t_dat) first_date, max(t_dat) last_date
        FROM read_parquet('{target.as_posix()}')""").df()
    save_table(scope, f"sample_scope_mod{mod}")
    print(scope.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod", type=int, default=5, help="keep one customer in MOD")
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()

    con = connect()
    convert_to_parquet(con)
    base_views(con)
    if not args.skip_profile:
        profile(con)
    build_sample(con, args.mod)
