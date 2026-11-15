# Optimization techniques in a linear regression framework for latency prediction from web server traffic data

Code, data and results accompanying the paper

> B. Hudoyorov. *Comparative Analysis of Optimization Techniques in a Linear Regression Framework for Latency Prediction from Web Server Traffic Data.*

The study compares three ways of fitting the same linear regression model - closed-form linear least squares (LLS),
mini-batch gradient descent (GAM) and mini-batch Adam - on traffic aggregated per second from the first one million
entries of a public web server access log.

## Important: what is real and what is simulated

| Column | Source |
|---|---|
| `RPS` (requests per second) | **real** - counted from the access log |
| `Payload_Size_KB` (mean bytes / 1024) | **real** - from the log's bytes field (15 missing values -> 500 B) |
| `Active_DB_Connections`, `Memory_Usage_MB`, `CPU_Load_percent` | **simulated** from `RPS` (heuristic formulas + noise) |
| `Latency_ms` | **simulated**: 15 ms + 0.5 ms/KB x payload + N(0, 5 ms) per request, averaged over the second; +12 ms per CPU percentage point above 80 % (1 interval) |

The log contains no server-side metrics and no response-time field, so the prediction task is a **controlled
benchmark for comparing optimizers, not a model of real server behaviour**. See the paper, Sections 2 and 3.1.

## Repository layout

```
data/dataset.csv              frozen dataset used in the paper (36,385 one-second intervals)
data/dataset_meta.json        provenance and counts for the frozen dataset
data/SHA256SUM                checksum of dataset.csv
src/train_models.py           fits LLS / GAM / Adam, writes all tables, figures and checks to results/
src/prepare_data.py           builds a dataset from access.log (see "Reproducibility" below)
scripts/verify_against_log.py checks the real columns (RPS, payload) of dataset.csv against the raw log
results/                      Figure_1..12.png, table1.csv, results.json (output of train_models.py)
```

## Reproduce the paper's numbers

```bash
pip install -r requirements.txt
python src/train_models.py
```

The script reads `data/dataset.csv`, prints Table 1 and the validation block (Table 2), and writes
`results/Figure_1..12.png`, `results/table1.csv` and `results/results.json`. It reports whether `dataset.csv`
matches the frozen checksum:

`fe8a398bc705e503c80cda6070ceaaabe0d57ebc5e40acd07ed433718d5033d8`

Seed 42 is used for the 80/20 split and the mini-batch order. Metrics reproduce to the printed precision.
Two harmless differences are expected: the training-set mean error of LLS is a floating-point rounding residue
(about 1e-15, its last digits depend on the NumPy/BLAS version), and the PNG files differ at byte level between
Matplotlib versions.

Tested with Python 3.12, NumPy 2.5, pandas 3.0, Matplotlib 3.11, seaborn 0.13.

## Reproducibility of the dataset

* `data/dataset.csv` is the **frozen** file all paper numbers come from. Treat it as the canonical artifact.
* `src/prepare_data.py` rebuilds a dataset from the raw log with seed 42. Its output goes to
  `data/dataset_regenerated.csv` (never over the frozen file). The **real** columns (`RPS`, `Payload_Size_KB`)
  are reproduced exactly (apart from the last row, see below); the **simulated** columns are a new random draw from
  the same generator, so numbers will move slightly, and the outlier filter (which depends on the random latency
  draw) may drop a couple of intervals: a test run gave 36,383 rows instead of 36,385, with the payload-latency
  correlation 0.959 instead of 0.960. The frozen file kept all 36,385 intervals.
* The frozen file's last row has `RPS = 44` where the log gives 43 (one extra line was read by an earlier version
  of the preparation script). No reported result is affected.

### Verify the real columns against the raw log

1. Download the log (not included here: 3.3 GB, contains visitor IP addresses):
   * Zaker, F. (2019). *Online Shopping Store - Web Server Logs*. Harvard Dataverse, V1.
     https://doi.org/10.7910/DVN/3QBYB5
   * Mirror: https://www.kaggle.com/datasets/eliasdabbas/web-server-access-logs
2. `python scripts/verify_against_log.py path/to/access.log`

Expected: 1,000,000 lines, 36,385 distinct seconds, max RPS 264, 15 missing byte counts, no numeric response-time
field, and one known mismatching row (see above).

## Data and licence notes

* Code: MIT (see `LICENSE`).
* `data/dataset.csv` contains only per-second aggregates and simulated values (no IP addresses, URLs or user
  agents). The original log belongs to its authors; please check the licence on the Harvard Dataverse page and cite
  the original source when reusing the real columns.

## Citation

See `CITATION.cff`.
