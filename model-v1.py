# combined_model_and_plot.py
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt


# =========================
# Easy-to-edit settings
# =========================
TARGET = "recession"  # <- change this in one place (or pass --target)
N_BINS = 10           # calibration bins

IN_PATH = Path("data/features/features.csv")

OUT_DIR = Path("data/results-v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PRED_PATH = OUT_DIR / "predictions.csv"
OUT_METRICS_PATH = OUT_DIR / "metrics_by_target.csv"

# If you want plots to read from the newly written predictions file, keep this as OUT_PRED_PATH.
PRED_PATH_FOR_PLOTS = OUT_PRED_PATH

FEATURE_COLS = ["ew_intro_rate", "ew_qna_rate", "since_last_any_log", "trend_any"]
LABEL_COL = "y_any"

MIN_TRAIN = 10  # start predicting only after at least this many meetings of history

# Baseline EWMA on y_any only (not your model feature)
BASELINE_ALPHA = 0.75


# =========================
# Helpers (model section)
# =========================
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def load_rows():
    rows = []
    with IN_PATH.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No rows found in {IN_PATH}")
    return rows


def group_by_target(rows):
    by_t = defaultdict(list)
    for row in rows:
        by_t[row["target"]].append(row)
    # Ensure chronological ordering within each target by date
    for t in by_t:
        by_t[t].sort(key=lambda x: x["date"])
    return by_t


def to_float(row, k):
    return float(row[k])


def to_int(row, k):
    return int(row[k])


# =========================
# Helpers (plot section)
# =========================
def load_target_rows(pred_path: Path, target: str):
    rows = []
    with pred_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["target"] == target:
                rows.append(row)
    rows.sort(key=lambda x: x["date"])
    if not rows:
        raise RuntimeError(f"No rows found for target={target}. Check {pred_path}.")
    return rows


def reliability_curve(p, y, n_bins=10):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        xs.append(p[mask].mean())
        ys.append(y[mask].mean())
        ns.append(mask.sum())
    return np.array(xs), np.array(ys), np.array(ns)


def plot_target(pred_path: Path, out_dir: Path, target: str, n_bins: int = 10):
    rows = load_target_rows(pred_path, target)

    dates = [r["date"] for r in rows]
    x = np.arange(len(dates))

    y = np.array([int(r["y"]) for r in rows], dtype=int)
    p_model = np.array([float(r["p_model"]) for r in rows], dtype=float)
    p_hist = np.array([float(r["p_hist_mean"]) for r in rows], dtype=float)
    p_ewma = np.array([float(r["p_ewma_y"]) for r in rows], dtype=float)

    feats = {k: np.array([float(r[k]) for r in rows], dtype=float) for k in FEATURE_COLS}

    # -------------------------
    # Plot 1: Probabilities vs actual
    # -------------------------
    plt.figure()
    plt.plot(x, p_model, label="p_model")
    plt.plot(x, p_hist, label="p_hist_mean")
    plt.plot(x, p_ewma, label="p_ewma_y")

    # actual y markers at 0/1 (slight vertical jitter so 0s don't hide axis line)
    y_plot = y.astype(float) * 0.98 + 0.01
    plt.scatter(x, y_plot, label="actual y (0/1)", marker="o")

    # reduce x tick clutter
    step = max(1, len(x) // 12)
    plt.xticks(
        x[::step],
        [dates[i] for i in range(0, len(dates), step)],
        rotation=45,
        ha="right",
    )

    plt.ylim(-0.05, 1.05)
    plt.title(f"Probabilities over time: {target}")
    plt.xlabel("Meeting date")
    plt.ylabel("Probability / Actual")
    plt.legend()
    plt.tight_layout()
    out1 = out_dir / f"probabilities_{target.replace(' ', '_')}.png"
    plt.savefig(out1, dpi=200)
    plt.close()

    # -------------------------
    # Plot 2: Feature series
    # -------------------------
    plt.figure()
    for k in FEATURE_COLS:
        plt.plot(x, feats[k], label=k)

    plt.xticks(
        x[::step],
        [dates[i] for i in range(0, len(dates), step)],
        rotation=45,
        ha="right",
    )
    plt.title(f"Feature values over time: {target}")
    plt.xlabel("Meeting date")
    plt.ylabel("Feature value")
    plt.legend()
    plt.tight_layout()
    out2 = out_dir / f"features_{target.replace(' ', '_')}.png"
    plt.savefig(out2, dpi=200)
    plt.close()

    # -------------------------
    # Plot 3: Reliability / calibration
    # -------------------------
    xs, ys, ns = reliability_curve(p_model, y, n_bins=n_bins)

    plt.figure()
    plt.plot([0, 1], [0, 1], label="ideal")
    plt.scatter(xs, ys, label="model bins")
    for i in range(len(xs)):
        plt.annotate(str(int(ns[i])), (xs[i], ys[i]), textcoords="offset points", xytext=(4, 4))

    plt.title(f"Calibration (reliability): {target}  (labels show bin counts)")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical frequency")
    plt.legend()
    plt.tight_layout()
    out3 = out_dir / f"calibration_{target.replace(' ', '_')}.png"
    plt.savefig(out3, dpi=200)
    plt.close()

    print(f"[OK] Saved:\n - {out1}\n - {out2}\n - {out3}")


# =========================
# Main: train + predict + metrics + plot
# =========================
def run_model_and_write_outputs():
    rows = load_rows()
    by_target = group_by_target(rows)

    pred_fields = [
        "date",
        "target",
        "y",
        "p_model",
        "p_hist_mean",
        "p_ewma_y",
        *FEATURE_COLS,
    ]

    metrics_fields = [
        "target",
        "n_total",
        "n_predicted",
        "pos_rate",
        "logloss_model",
        "logloss_hist_mean",
        "logloss_ewma_y",
        "brier_model",
        "brier_hist_mean",
        "brier_ewma_y",
    ]

    metrics_rows = []

    with OUT_PRED_PATH.open("w", newline="", encoding="utf-8") as fp:
        wp = csv.DictWriter(fp, fieldnames=pred_fields)
        wp.writeheader()

        for target, seq in by_target.items():
            X_all = np.array([[to_float(r, c) for c in FEATURE_COLS] for r in seq], dtype=float)
            y_all = np.array([to_int(r, LABEL_COL) for r in seq], dtype=int)
            dates = [r["date"] for r in seq]

            p_model_list = []
            p_mean_list = []
            p_ewma_list = []
            y_list = []

            # Baseline EWMA state for y only
            ewma_y = 0.0

            for t in range(len(seq)):
                # baselines must be "past only"
                if t == 0:
                    p_hist = 0.0
                else:
                    p_hist = float(np.mean(y_all[:t]))

                p_ewma = float(ewma_y)

                # update EWMA AFTER using it (so it's past-only for time t)
                ewma_y = BASELINE_ALPHA * ewma_y + (1 - BASELINE_ALPHA) * float(y_all[t])

                # Only start model predictions after MIN_TRAIN
                if t < MIN_TRAIN:
                    continue

                X_train = X_all[:t]
                y_train = y_all[:t]
                X_test = X_all[t:t + 1]

                # If y_train is all 0s or all 1s, logistic regression can't fit.
                # In that case, fall back to historical mean for p_model.
                if len(np.unique(y_train)) < 2:
                    p_model = p_hist
                else:
                    clf = LogisticRegression(
                        solver="lbfgs",
                        max_iter=2000,
                    )
                    clf.fit(X_train, y_train)
                    p_model = float(clf.predict_proba(X_test)[0, 1])

                wp.writerow({
                    "date": dates[t],
                    "target": target,
                    "y": int(y_all[t]),
                    "p_model": f"{p_model:.6f}",
                    "p_hist_mean": f"{p_hist:.6f}",
                    "p_ewma_y": f"{p_ewma:.6f}",
                    **{c: f"{X_all[t, i]:.6f}" for i, c in enumerate(FEATURE_COLS)},
                })

                p_model_list.append(p_model)
                p_mean_list.append(p_hist)
                p_ewma_list.append(p_ewma)
                y_list.append(int(y_all[t]))

            # Metrics for this target (only over predicted timesteps)
            if len(y_list) == 0:
                continue

            p_model_arr = np.array(p_model_list, dtype=float)
            p_mean_arr = np.array(p_mean_list, dtype=float)
            p_ewma_arr = np.array(p_ewma_list, dtype=float)
            y_arr = np.array(y_list, dtype=int)

            metrics_rows.append({
                "target": target,
                "n_total": len(seq),
                "n_predicted": len(y_arr),
                "pos_rate": f"{float(np.mean(y_arr)):.4f}",
                "logloss_model": f"{logloss(p_model_arr, y_arr):.6f}",
                "logloss_hist_mean": f"{logloss(p_mean_arr, y_arr):.6f}",
                "logloss_ewma_y": f"{logloss(p_ewma_arr, y_arr):.6f}",
                "brier_model": f"{brier(p_model_arr, y_arr):.6f}",
                "brier_hist_mean": f"{brier(p_mean_arr, y_arr):.6f}",
                "brier_ewma_y": f"{brier(p_ewma_arr, y_arr):.6f}",
            })

    with OUT_METRICS_PATH.open("w", newline="", encoding="utf-8") as fm:
        wm = csv.DictWriter(fm, fieldnames=metrics_fields)
        wm.writeheader()
        for row in sorted(metrics_rows, key=lambda r: float(r["logloss_model"])):
            wm.writerow(row)

    print(f"[OK] Wrote predictions: {OUT_PRED_PATH}")
    print(f"[OK] Wrote metrics:     {OUT_METRICS_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=TARGET, help="Target name to plot (default: TARGET constant)")
    parser.add_argument("--no-plot", action="store_true", help="Only write predictions/metrics, skip plots")
    parser.add_argument("--bins", type=int, default=N_BINS, help="Number of bins for calibration plot")
    args = parser.parse_args()

    run_model_and_write_outputs()

    if not args.no_plot:
        plot_target(PRED_PATH_FOR_PLOTS, OUT_DIR, args.target, n_bins=args.bins)


if __name__ == "__main__":
    main()
