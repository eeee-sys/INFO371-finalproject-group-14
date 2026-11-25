"""Shared helpers for the INFO371 Group 14 analysis.

Data layout
-----------
data/raw/        the three H&M CSV files (articles, customers, transactions_train)
data/cache/      Parquet copies of the raw files and the customer samples
results/tables/  CSV tables used in the report
results/figures/ PNG figures used in the report
"""
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

SEED = 37114              # same seed as the first experiment round
DATA_END = "2020-09-23"   # exclusive end: the last transaction is 2020-09-22

for folder in (CACHE, TABLES, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)


def connect(memory_limit: str = "6GB", threads: int = 8) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with a bounded memory budget."""
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    (CACHE / "duckdb_tmp").mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{(CACHE / 'duckdb_tmp').as_posix()}'")
    return con


def base_views(con: duckdb.DuckDBPyConnection) -> None:
    """Register the three H&M tables as views on their Parquet copies.

    IDs are kept as text so that leading zeros in article_id survive.
    """
    for name in ("articles", "customers", "transactions_train"):
        parquet = CACHE / f"{name}.parquet"
        if not parquet.exists():
            raise FileNotFoundError(f"{parquet} missing: run src/prepare.py first")
        con.execute(
            f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{parquet.as_posix()}')"
        )


def sample_events(con: duckdb.DuckDBPyConnection, mod: int) -> None:
    """Register `events`: every purchase of the sampled customers, joined to product fields.

    `mod` is the sampling modulus: 5 keeps one customer in five, 20 keeps one in twenty.
    """
    parquet = CACHE / f"events_mod{mod}.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"{parquet} missing: run `src/prepare.py --mod {mod}` first")
    con.execute(f"CREATE OR REPLACE VIEW events AS SELECT * FROM read_parquet('{parquet.as_posix()}')")


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLES / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"table  -> {path.relative_to(ROOT)}  ({len(df)} rows)")
    return path


# Department of an article (who it is sold to), derived from the index group and section name.
DEPARTMENT_SQL = """
    CASE WHEN a.index_group_name = 'Menswear'
              OR (a.section_name LIKE '%Men%' AND a.section_name NOT LIKE '%Women%') THEN 'Menswear dept.'
         WHEN a.index_group_name = 'Baby/Children'
              OR regexp_matches(coalesce(a.section_name, ''), 'Kids|Baby|Girl|Boy|Young') THEN 'Kids dept.'
         ELSE 'Womenswear dept.' END"""
DEPARTMENTS = ["Womenswear dept.", "Menswear dept.", "Kids dept."]


def garment_groups(con: duckdb.DuckDBPyConnection) -> list[str]:
    """The 21 garment groups, in a fixed alphabetical order."""
    return con.sql("SELECT DISTINCT garment_group_name FROM articles ORDER BY 1").df().iloc[:, 0].tolist()
