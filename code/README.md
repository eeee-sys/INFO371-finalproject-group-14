# INFO371 Group 14: code of the final project

Project: **Data Mining for Online Clothing Retail**

Group 14: Tian Zijie, Ma Shang, Zheng Chenye, Fang Yan.

This folder holds the Python code that produced the tables and figures of the final report and the figures of the slides. All comments are in English.

## Dataset

H&M Personalized Fashion Recommendations (Kaggle):
<https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data>

Three files are used: `articles.csv` (105,542 products), `customers.csv` (1.37 million customers) and `transactions_train.csv` (31.8 million purchases, 20 September 2018 to 22 September 2020). They take 3.7 GB and are not included here. The link is also in `data/dataset_link.txt`. To run the code, download the files, create the folder `data/raw/` and place them there.

## Requirements

Python 3.12 and the packages in `requirements.txt`:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## How to run

Run the scripts from this folder, in this order. `--mod 20` keeps one customer in twenty, with all of their purchases; this is the sample used in the report.

```bash
.venv/bin/python src/prepare.py --mod 20
.venv/bin/python src/rq3_text_mining.py
.venv/bin/python src/rq1_clustering.py --mod 20 --k-final 6
.venv/bin/python src/rq1_validation.py --mod 20
.venv/bin/python src/rq2_categories.py --mod 20
.venv/bin/python src/rq2_products.py --mod 20
.venv/bin/python src/rq2_baselines.py --mod 20
.venv/bin/python src/rq4_association.py --mod 20
.venv/bin/python src/insight_repeat.py
.venv/bin/python src/insight_first_time.py --mod 20
.venv/bin/python src/insight_checks.py --mod 20
.venv/bin/python src/eda_figures.py
.venv/bin/python src/diagrams.py
.venv/bin/python src/advanced_figures.py --mod 20
.venv/bin/python src/nugget_figures.py --mod 20
.venv/bin/python src/slide_figures.py --mod 20
```

On a laptop with 16 GB of memory, `rq2_categories.py` takes about 5 minutes, `rq2_products.py` about 18 minutes, `rq2_baselines.py` about 15 minutes, and every other script less than two minutes. Tables are written to `results/tables/` (CSV) and figures to `results/figures/` (PNG); the results of our run are included. `nugget_figures.py`, `slide_figures.py` and `rq2_baselines.py` accept `--figure` to redraw from the saved tables. Every random step uses the same seed and every query returns its rows in a fixed order.

The folder `website/` holds the website prototype shown in the report (sample products, not connected to the model); see `website/README.md`.

## Scripts

The file names follow the numbering of the project proposal; the table gives the research question in the report.

| Script | Research question in the report | What it does |
|---|---|---|
| `common.py` | all | Paths, random seed, DuckDB connection, the sampled purchase view, department rule |
| `viz.py` | all | Shared figure style and colours |
| `prepare.py` | Data understanding | Converts the CSV files to Parquet, profiles the full data and draws the customer sample |
| `rq3_text_mining.py` | RQ1 Similar products | TF-IDF vectors of product texts, cosine similarity, removal of colour variants, attribute agreement |
| `rq1_clustering.py` | RQ2 Customer groups | Taste vector per customer, SVD, K-means, silhouette by K, group profiles |
| `rq1_validation.py` | RQ2 Customer groups | Reference for the silhouette (shuffled data), other customer vectors, check on later purchases |
| `rq4_association.py` | RQ3 Purchase patterns | FP-growth rules, co-purchase network with communities, transitions to the next visit |
| `rq2_categories.py` | RQ4 Category layer | Predicts which garment groups a customer buys in the next seven days |
| `rq2_products.py` | RQ4 Product layer | Seven candidate sources, gradient-boosting ranking, ablation, permutation importance, steps from the initial to the final model |
| `rq2_baselines.py` | RQ4 Baselines | Simple rules and course classifiers on the same final products |
| `insight_repeat.py` | Nugget 2 | Duplicate purchases by channel, product type and time since the purchase |
| `insight_first_time.py` | Nugget 3 | Coverage of first-time purchases by best-seller window, list age and customer history |
| `insight_checks.py` | Nuggets 2 and 3 | Product families, comparison of the 7-day and the 30-day list, categories bought before |
| `nugget_figures.py` | Nuggets 1 to 3 | Figures of the three nuggets |
| `slide_figures.py` | Slides | Maps, network and baseline figure of the presentation |
| `eda_figures.py`, `diagrams.py`, `advanced_figures.py` | several | Exploratory figures, time-split diagram and further figures from the result tables |
