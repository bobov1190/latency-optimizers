"""
train_models.py  (v4, verified)

Fits three linear regression solvers on dataset.csv and writes everything the paper needs:
  * Figure_1 ... Figure_12 (.png), results.json, table1.csv
  * sanity checks that show WHERE the numbers come from (see section "Validation" below)

Run:  python src/train_models.py      (from any folder; reads data/dataset.csv, writes to results/)
Needs: numpy, pandas, matplotlib, seaborn
"""
import hashlib
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_CSV = os.path.join(ROOT, "data", "dataset.csv")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)
os.chdir(RESULTS)

SEED = 42
# SHA-256 of the dataset.csv the paper's numbers were computed from (frozen copy, 36,385 rows)
FROZEN_SHA256 = "fe8a398bc705e503c80cda6070ceaaabe0d57ebc5e40acd07ed433718d5033d8"

# Parameters of the latency generator in prepare_data.py (used only for the validation block)
BASE_MS, SLOPE_MS_PER_KB, NOISE_SD_MS = 15.0, 0.5, 5.0

FEATURES = ["RPS", "Active_DB_Connections", "Memory_Usage_MB", "Payload_Size_KB", "CPU_Load_percent"]
WEIGHT_LABELS = ["Bias", "RPS", "Active DB", "Memory MB", "Payload KB", "CPU %"]

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.figsize": (8, 5)})

# ------------------------------------------------------------------ data
print("Loading data/dataset.csv ...")
sha = hashlib.sha256(open(DATA_CSV, "rb").read()).hexdigest()
if sha != FROZEN_SHA256:
    print("NOTE: dataset.csv differs from the frozen copy used in the paper.\n"
          "      Numbers below are valid for THIS file; update the paper tables from results.json.")
df = pd.read_csv(DATA_CSV)
X_raw = df[FEATURES].values
y_raw = df["Latency_ms"].values

# ------------------------------------------------------------------ Figure 1
plt.figure(figsize=(7.5, 6))
short = {"RPS": "RPS", "Active_DB_Connections": "DB connections", "Memory_Usage_MB": "Memory (MB)",
         "Payload_Size_KB": "Payload (KB)", "CPU_Load_percent": "CPU load (%)", "Latency_ms": "Latency (ms)"}
cm = df.corr().rename(index=short, columns=short)
sns.heatmap(cm, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, fmt=".2f", square=True,
            linewidths=0.5, cbar_kws={"label": "Pearson r"})
plt.title("Correlation matrix of the features and latency")
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.savefig("Figure_1_Correlation_Matrix.png", dpi=300)
plt.close()

# ------------------------------------------------------------------ split + z-score (Eq. 1), intercept column
np.random.seed(SEED)
idx = np.random.permutation(len(X_raw))
cut = int(len(X_raw) * 0.8)
tr, te = idx[:cut], idx[cut:]
Xtr_raw, Xte_raw, y_train, y_test = X_raw[tr], X_raw[te], y_raw[tr], y_raw[te]

mu, sigma = Xtr_raw.mean(axis=0), Xtr_raw.std(axis=0)          # training statistics only
X_train = np.hstack([np.ones((len(Xtr_raw), 1)), (Xtr_raw - mu) / sigma])
X_test = np.hstack([np.ones((len(Xte_raw), 1)), (Xte_raw - mu) / sigma])


def metrics(y, p):
    e = y - p
    return {"mu_e": float(e.mean()), "sigma_e": float(e.std()), "mse": float((e ** 2).mean()),
            "r2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())}


def plots(w, ytr_pred, yte_pred, title, n0):
    # weights: bar chart (features are categories, not an ordered axis)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cols = ["tab:gray"] + ["tab:blue"] * (len(w) - 1)
    bars = ax.bar(WEIGHT_LABELS, w, color=cols)
    for r, v in zip(bars, w):
        ax.annotate(f"{v:.2f}", (r.get_x() + r.get_width() / 2, v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=10, xytext=(0, 3 if v >= 0 else -3), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{title}: fitted weights (standardized features)")
    ax.set_ylabel("weight (bias in ms, others in ms per 1 SD)")
    ax.set_ylim(min(0, w.min()) - 2, w.max() * 1.12)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(f"Figure_{n0}_Weights.png", dpi=300); plt.close(fig)

    # predicted vs observed (test)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, yte_pred, alpha=0.3, color="tab:blue", s=12, label="test observations")
    lo, hi = min(y_test.min(), yte_pred.min()), max(y_test.max(), yte_pred.max())
    ax.plot([lo, hi], [lo, hi], color="tab:red", linewidth=2, label="ideal: predicted = observed")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title}: predicted vs observed latency (test)")
    ax.set_xlabel("observed latency y (ms)"); ax.set_ylabel("predicted latency \u0177 (ms)")
    ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig(f"Figure_{n0 + 1}_Scatter.png", dpi=300); plt.close(fig)

    # error histogram: shared fixed bins, bar height = share of observations
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(-10, 10, 41)
    for e, lab, c in ((y_train - ytr_pred, "training", "tab:blue"), (y_test - yte_pred, "test", "tab:orange")):
        outside = float(np.mean((e < -10) | (e > 10)))
        ax.hist(e, bins=bins, weights=np.full(len(e), 1.0 / len(e)), alpha=0.55, color=c,
                label=f"{lab} (|e| > 10 ms: {outside * 100:.2f}%)")
    ax.set_title(f"{title}: error distribution")
    ax.set_xlabel("error e = y \u2212 \u0177 (ms)"); ax.set_ylabel("share of observations per 0.5 ms bin")
    ax.legend(); fig.tight_layout(); fig.savefig(f"Figure_{n0 + 2}_Histogram.png", dpi=300); plt.close(fig)


# ------------------------------------------------------------------ 1) LLS, Eq. (7)
print("1/3 Linear Least Squares ...")
w_lls = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train

# ------------------------------------------------------------------ 2) mini-batch GD, Eqs. (12)-(13)
print("2/3 Mini-batch gradient descent (B=64) ...")
H_GAM = {"lr": 0.01, "batch": 64, "epochs": 80, "init": "zeros"}
w_gam = np.zeros(X_train.shape[1])
for _ in range(H_GAM["epochs"]):
    p = np.random.permutation(len(X_train))
    Xs, ys = X_train[p], y_train[p]
    for i in range(0, len(Xs), H_GAM["batch"]):
        Xb, yb = Xs[i:i + H_GAM["batch"]], ys[i:i + H_GAM["batch"]]
        w_gam -= H_GAM["lr"] * (-2 * (Xb.T @ (yb - Xb @ w_gam)) / len(Xb))

# ------------------------------------------------------------------ 3) Adam, Eqs. (8)-(11)
print("3/3 Adam (B=32) ...")
H_ADAM = {"lr": 0.01, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "batch": 32, "epochs": 50, "init": "zeros"}
w_adam = np.zeros(X_train.shape[1]); m = np.zeros_like(w_adam); v = np.zeros_like(w_adam); t = 0
for _ in range(H_ADAM["epochs"]):
    p = np.random.permutation(len(X_train))
    Xs, ys = X_train[p], y_train[p]
    for i in range(0, len(Xs), H_ADAM["batch"]):
        Xb, yb = Xs[i:i + H_ADAM["batch"]], ys[i:i + H_ADAM["batch"]]
        t += 1
        g = -2 * (Xb.T @ (yb - Xb @ w_adam)) / len(Xb)
        m = H_ADAM["beta1"] * m + (1 - H_ADAM["beta1"]) * g
        v = H_ADAM["beta2"] * v + (1 - H_ADAM["beta2"]) * g ** 2
        mh, vh = m / (1 - H_ADAM["beta1"] ** t), v / (1 - H_ADAM["beta2"] ** t)
        w_adam -= H_ADAM["lr"] * mh / (np.sqrt(vh) + H_ADAM["eps"])

models = {"LLS": w_lls, "GAM": w_gam, "SGD (Adam)": w_adam}
titles = {"LLS": ("LLS", 2), "GAM": ("Mini-batch GD (B = 64)", 5), "SGD (Adam)": ("SGD with Adam (B = 32)", 8)}
res = {"models": {}}
for name, w in models.items():
    ptr, pte = X_train @ w, X_test @ w
    plots(w, ptr, pte, *titles[name])
    res["models"][name] = {"weights_standardized": w.tolist(), "train": metrics(y_train, ptr), "test": metrics(y_test, pte)}

# ------------------------------------------------------------------ Table 1
rows = [(n, ds, res["models"][n][k]) for n in models for ds, k in (("Training", "train"), ("Test", "test"))]
with open("table1.csv", "w", encoding="utf-8") as f:
    f.write("Model,Dataset,mu_e,sigma_e,MSE,R2\n")
    for n, ds, r in rows:
        f.write(f"{n},{ds},{r['mu_e']:.3e},{r['sigma_e']:.4f},{r['mse']:.2f},{r['r2']:.3f}\n")
print("\nTABLE 1")
print(f"{'Model':<11}|{'Dataset':<9}|{'mu_e':<11}|{'sigma_e':<9}|{'MSE':<6}|R2")
for n, ds, r in rows:
    print(f"{n:<11}|{ds:<9}|{r['mu_e']:<11.3e}|{r['sigma_e']:<9.4f}|{r['mse']:<6.2f}|{r['r2']:.3f}")

# ------------------------------------------------------------------ Validation (Table 2)
# (a) how far apart are the three solutions?  (standardized weights, and test R2)
d_gam = float(np.max(np.abs(w_gam - w_lls))); d_adam = float(np.max(np.abs(w_adam - w_lls)))
r2s = [res["models"][n]["test"]["r2"] for n in models]
res["solver_agreement"] = {"max_abs_weight_diff_GAM_vs_LLS": d_gam, "max_abs_weight_diff_Adam_vs_LLS": d_adam,
                           "test_r2_spread": float(max(r2s) - min(r2s))}

# (b) coefficients in raw units (full data, ordinary least squares) -> recover the generator
A = np.hstack([np.ones((len(X_raw), 1)), X_raw])
b_raw = np.linalg.lstsq(A, y_raw, rcond=None)[0]
res["raw_unit_ols"] = dict(zip(["intercept"] + FEATURES, b_raw.tolist()))

# (c) payload-only baseline (fit on train, evaluate on test)
pay = FEATURES.index("Payload_Size_KB")
A1tr = np.column_stack([np.ones(len(Xtr_raw)), Xtr_raw[:, pay]]); A1te = np.column_stack([np.ones(len(Xte_raw)), Xte_raw[:, pay]])
b1 = np.linalg.lstsq(A1tr, y_train, rcond=None)[0]
res["payload_only"] = {"intercept": float(b1[0]), "slope_ms_per_kb": float(b1[1]), "test": metrics(y_test, A1te @ b1)}

# (d) noise ceiling: each one-second latency is the mean of RPS noisy draws, so noise variance = sigma^2 / RPS.
#     R2 cannot exceed signal_var / (signal_var + expected_noise_var); computed per split and on all data.
def noise_ceiling(rows):
    d = df.iloc[rows]
    sv = float((SLOPE_MS_PER_KB * d["Payload_Size_KB"]).var())
    nv = float((NOISE_SD_MS ** 2 / d["RPS"]).mean())
    return {"signal_var": sv, "expected_noise_var": nv, "r2_ceiling": sv / (sv + nv)}


res["noise_ceiling"] = {"full": noise_ceiling(np.arange(len(df))), "train": noise_ceiling(tr), "test": noise_ceiling(te)}

# (d2) share of test squared error coming from the few high-latency observations (> 60 ms), LLS
e_te = y_test - X_test @ w_lls
hi = y_test > 60
res["upper_tail"] = {"n_test_over_60ms": int(hi.sum()), "share_of_sse": float((e_te[hi] ** 2).sum() / (e_te ** 2).sum())}

# (e) multicollinearity diagnostics
Z = (X_raw - X_raw.mean(0)) / X_raw.std(0)
vif = {}
for j, nme in enumerate(FEATURES):
    others = np.hstack([np.ones((len(Z), 1)), np.delete(Z, j, axis=1)])
    bb = np.linalg.lstsq(others, Z[:, j], rcond=None)[0]
    vif[nme] = float(1 / (1 - (1 - ((Z[:, j] - others @ bb) ** 2).sum() / (Z[:, j] ** 2).sum())))
res["vif"] = vif
res["condition_number_standardized_X"] = float(np.linalg.cond(Z))

# (f) is the CPU saturation penalty ever active?
res["dataset"] = {"rows": int(len(df)), "train_rows": int(len(tr)), "test_rows": int(len(te)),
                  "rows_cpu_over_80": int((df["CPU_Load_percent"] > 80).sum()),
                  "max_rps": float(df["RPS"].max()), "mean_latency": float(df["Latency_ms"].mean()),
                  "sha256": sha, "matches_frozen_copy": sha == FROZEN_SHA256}
res["hyperparameters"] = {"seed": SEED, "split": "80/20", "GAM": H_GAM, "Adam": H_ADAM}

# (g) residual spread by request-rate bucket vs the sigma/sqrt(RPS) law  -> Figure 11
res_te = y_test - X_test @ w_lls
rps_te = Xte_raw[:, 0]
buckets = [(1, 1), (2, 5), (6, 20), (21, 60), (61, 10 ** 6)]
labels, obs, th, cnt = [], [], [], []
for lo, hi_ in buckets:
    mk = (rps_te >= lo) & (rps_te <= hi_)
    labels.append("1" if lo == hi_ else (f"{lo}-{hi_}" if hi_ < 10 ** 6 else f">{lo - 1}"))
    obs.append(float(res_te[mk].std())); th.append(float(np.sqrt((NOISE_SD_MS ** 2 / rps_te[mk]).mean()))); cnt.append(int(mk.sum()))
res["residual_by_rps"] = {"bucket": labels, "observed_sd": obs, "theory_sd": th, "n": cnt}
plt.figure(figsize=(7, 4.5))
plt.plot(labels, obs, marker="o", linewidth=2, color="tab:blue", label="observed residual SD (LLS, test)")
plt.plot(labels, th, marker="s", linestyle="--", linewidth=2, color="tab:red", label="theory: 5 / \u221aRPS")
plt.title("Residual spread vs request rate"); plt.xlabel("requests per second (bucket)"); plt.ylabel("SD of error e (ms)")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig("Figure_11_Residual_vs_RPS.png", dpi=300); plt.close()


# ------------------------------------------------------------------ Figure 12: confusion matrices
# Regression has no native confusion matrix; latency is binned into three classes using the
# 33rd/67th percentiles of the TRAINING latency, and the true and predicted values are binned the same way.
q1, q2 = np.percentile(y_train, [100 / 3, 200 / 3])
CLS = ["Low", "Medium", "High"]


def to_cls(v):
    return np.digitize(v, [q1, q2])


yt_c = to_cls(y_test)
conf_res = {}
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, (name, w) in zip(axes, models.items()):
    pc = to_cls(X_test @ w)
    M = np.zeros((3, 3), dtype=int)
    for a_, b_ in zip(yt_c, pc):
        M[a_, b_] += 1
    rown = M / M.sum(axis=1, keepdims=True)
    acc = float(np.trace(M) / M.sum())
    f1 = []
    for k in range(3):
        p_ = M[k, k] / max(M[:, k].sum(), 1); r_ = M[k, k] / max(M[k, :].sum(), 1)
        f1.append(2 * p_ * r_ / max(p_ + r_, 1e-12))
    conf_res[name] = {"matrix": M.tolist(), "accuracy": acc, "macro_f1": float(np.mean(f1)),
                      "within_one_class": float((np.abs(yt_c - pc) <= 1).mean())}
    sns.heatmap(rown, annot=False, cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax, linewidths=0.5, square=True)
    for i in range(3):
        for j in range(3):
            ax.text(j + 0.5, i + 0.5, f"{M[i, j]}\n({rown[i, j] * 100:.0f}%)", ha="center", va="center",
                    color="white" if rown[i, j] > 0.5 else "black", fontsize=10)
    ax.set_xticklabels(CLS); ax.set_yticklabels(CLS, rotation=0)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    ax.set_title(f"{name}\naccuracy {acc * 100:.1f}%, macro-F1 {np.mean(f1):.3f}")
fig.suptitle(f"Confusion matrices on the test set (Low < {q1:.1f} ms \u2264 Medium < {q2:.1f} ms \u2264 High; "
             "cells: count and row share)", fontsize=11)
fig.tight_layout(); fig.savefig("Figure_12_Confusion_Matrix.png", dpi=300, bbox_inches="tight"); plt.close(fig)
res["confusion"] = {"thresholds_ms": [float(q1), float(q2)], "classes": CLS, "models": conf_res}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)

print("\nVALIDATION")
print(f"  max |w_GAM-w_LLS| = {d_gam:.3f}   max |w_Adam-w_LLS| = {d_adam:.3f}   test-R2 spread = {max(r2s)-min(r2s):.4f}")
print("  raw-unit OLS:", {k: round(v_, 4) for k, v_ in res["raw_unit_ols"].items()})
print(f"  payload-only test R2 = {res['payload_only']['test']['r2']:.4f}  (5 features: {res['models']['LLS']['test']['r2']:.4f})")
for _k in ("train", "test"):
    _c = res["noise_ceiling"][_k]
    print(f"  noise ceiling ({_k}): R2 <= {_c['r2_ceiling']:.4f}; expected noise var = {_c['expected_noise_var']:.2f}; "
          f"observed MSE = {res['models']['LLS'][_k]['mse']:.2f}; observed R2 = {res['models']['LLS'][_k]['r2']:.4f}")
print(f"  test observations > 60 ms: {res['upper_tail']['n_test_over_60ms']} -> {100 * res['upper_tail']['share_of_sse']:.1f} % of squared error")
print("  VIF:", {k: round(v_, 1) for k, v_ in vif.items()}, " cond(X) = %.1f" % res["condition_number_standardized_X"])
print(f"  rows with CPU > 80 %: {res['dataset']['rows_cpu_over_80']} of {len(df)}   (dataset matches frozen copy: {sha == FROZEN_SHA256})")
print("\nCONFUSION (latency binned by training terciles: %.2f / %.2f ms)" % (q1, q2))
for n_, r_ in conf_res.items():
    print(f"  {n_:<11} acc={r_['accuracy']:.4f} macroF1={r_['macro_f1']:.4f} within-one-class={r_['within_one_class']:.4f}  M={r_['matrix']}")
print("Done: Figure_1..12.png, table1.csv, results.json")
