#!/usr/bin/env python3
"""
Sweep parameters for mentions/score-and-test.py, find robust "stable maximum" params,
then re-run once with the recommended params and print the familiar backtest summary.

How "robust" is defined:
- For each parameter combo, look at a local neighborhood (small +/- perturbations)
- Compute the worst-case (min) Total PnL in that neighborhood
- Pick the combo that maximizes this worst-case (maximin robustness)
- Tie-breakers: higher profit_factor, lower max_drawdown, more trades

This makes it very hard to pick a knife-edge solution that collapses if params shift slightly.

Outputs:
- sweep_results.csv (full grid results)
- prints recommended params + neighborhood stability report
- runs one final backtest with recommended params and prints the same summary
"""

import csv
import itertools
import math
import sys
from pathlib import Path
import importlib.util
import io
import contextlib

# -----------------------------
# Paths
# -----------------------------
TEST_FILE = Path("mentions/score-and-test.py")  # <-- your scoring/backtest file
TRADE_LOG = Path("data/scoring-v1/trade_log.csv")
SWEEP_OUT = Path("data/scoring-v1/sweep_results.csv")

# -----------------------------
# Sweep grid (edit as you like)
# -----------------------------
GRID = {
    "EV_MIN": [0.02, 0.03, 0.04, 0.05], # cap at 0.02 to account for fees
    "PMKT_MIN": [0.15, 0.20, 0.25, 0.30, 0.35],
    "PMKT_MAX": [0.65, 0.70, 0.75, 0.80, 0.85],
    "MAX_ENTRY_PRICE": [0.40, 0.45, 0.50, 0.55], # cap at 0.55 to minimize drawdown
}

# Hard sanity constraints during selection (not during sweep)
MIN_TRADES_FOR_SELECTION = 10

# Neighborhood definition for robustness:
# (these are "small changes" — if you shift within these, performance should not crater)
NEIGHBOR_DELTAS = {
    "EV_MIN": 0.01,
    "PMKT_MIN": 0.05,
    "PMKT_MAX": 0.05,
    "MAX_ENTRY_PRICE": 0.05,
}

# Robust scoring weights (only used as tie-breaks after maximin)
DRAW_DOWN_PENALTY = 0.10  # penalize drawdown a bit in final tie-break score


# -----------------------------
# Helpers
# -----------------------------
def ffloat(x, default=float("nan")):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def load_trade_log(trade_log_path: Path):
    if not trade_log_path.exists():
        return []
    with trade_log_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compute_stats_from_trade_log(rows):
    """
    Recompute summary stats from trade_log.csv so we don't rely on parsing stdout.
    Matches your printed metrics:
    - trades, total_pnl, winrate, avg_pnl, profit_factor, max_drawdown
    """
    pnls = [ffloat(r.get("pnl", "0")) for r in rows]
    n = len(pnls)
    if n == 0:
        return {
            "trades": 0,
            "total_pnl": 0.0,
            "winrate": 0.0,
            "avg_pnl": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
        }

    wins = sum(1 for x in pnls if x > 0)
    total = sum(pnls)
    avg = total / n

    gross_profit = sum(x for x in pnls if x > 0)
    gross_loss = -sum(x for x in pnls if x < 0)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # max drawdown on equity curve generated from pnls in order
    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    for x in pnls:
        eq += x
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd

    return {
        "trades": n,
        "total_pnl": total,
        "winrate": wins / n,
        "avg_pnl": avg,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }


def load_test_module(py_path: Path):
    if not py_path.exists():
        raise FileNotFoundError(f"Missing test file: {py_path}")

    spec = importlib.util.spec_from_file_location("score_and_test_mod", str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from: {py_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_backtest_once(mod, params):
    """
    Run mod.main() after overwriting its global knob values.
    Return:
      - status
      - stats dict (computed from trade_log.csv)
      - captured_stdout (so we can print the same report at the end)
    """
    # Apply params onto module globals (must match names in score-and-test.py)
    for k, v in params.items():
        if not hasattr(mod, k):
            return "MISSING_PARAM", None, ""
        setattr(mod, k, v)

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            mod.main()
    except Exception as e:
        return f"ERROR: {e}", None, buf.getvalue()

    rows = load_trade_log(TRADE_LOG)
    stats = compute_stats_from_trade_log(rows)
    return "OK", stats, buf.getvalue()


def in_neighbor(a, b, deltas):
    """
    True if param dict a is within deltas of param dict b.
    """
    for k, d in deltas.items():
        if abs(a[k] - b[k]) > d + 1e-12:
            return False
    return True


def fmt_inf(x):
    if x == float("inf"):
        return "inf"
    if x != x:
        return "nan"
    return f"{x:.6g}"


# -----------------------------
# Main sweep + robust selection
# -----------------------------
def main():
    mod = load_test_module(TEST_FILE)

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))

    results = []
    print(f"[sweep] running {len(combos)} combos...", file=sys.stderr)

    for i, tup in enumerate(combos, start=1):
        params = dict(zip(keys, tup))
        status, stats, _ = run_backtest_once(mod, params)

        if status != "OK" or stats is None:
            row = {
                **params,
                "status": status,
                "trades": 0,
                "total_pnl": 0.0,
                "winrate": 0.0,
                "avg_pnl": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
            }
        else:
            row = {
                **params,
                "status": "OK",
                "trades": stats["trades"],
                "total_pnl": stats["total_pnl"],
                "winrate": stats["winrate"],
                "avg_pnl": stats["avg_pnl"],
                "profit_factor": stats["profit_factor"],
                "max_drawdown": stats["max_drawdown"],
            }

        results.append(row)

        if i % 10 == 0:
            print(f"[sweep] {i}/{len(combos)}", file=sys.stderr)

    # Write sweep CSV
    SWEEP_OUT.parent.mkdir(parents=True, exist_ok=True)
    with SWEEP_OUT.open("w", encoding="utf-8", newline="") as f:
        cols = keys + ["status", "trades", "total_pnl", "winrate", "avg_pnl", "profit_factor", "max_drawdown"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\nWrote sweep results: {SWEEP_OUT}")

    # -----------------------------
    # Robust selection (stable maximum)
    # -----------------------------
    ok = [r for r in results if r["status"] == "OK" and r["trades"] >= MIN_TRADES_FOR_SELECTION]
    if not ok:
        print("\nNo OK rows with enough trades to select from. Lower MIN_TRADES_FOR_SELECTION or expand data.")
        return

    # For each candidate, compute neighborhood worst-case pnl and neighborhood mean pnl
    enriched = []
    for r in ok:
        params_r = {k: r[k] for k in keys}
        neigh = [q for q in ok if in_neighbor({k: q[k] for k in keys}, params_r, NEIGHBOR_DELTAS)]
        if not neigh:
            continue
        min_pnl = min(q["total_pnl"] for q in neigh)
        mean_pnl = sum(q["total_pnl"] for q in neigh) / len(neigh)
        mean_pf = sum(q["profit_factor"] for q in neigh if q["profit_factor"] != float("inf")) / max(
            1, sum(1 for q in neigh if q["profit_factor"] != float("inf"))
        )
        max_dd = max(q["max_drawdown"] for q in neigh)

        # primary objective: maximin pnl (stability)
        # tie-break objective: prefer higher mean pnl and mean profit factor, lower dd
        tie_score = mean_pnl + 0.05 * mean_pf - DRAW_DOWN_PENALTY * max_dd

        enriched.append({
            **r,
            "neigh_n": len(neigh),
            "neigh_min_pnl": min_pnl,
            "neigh_mean_pnl": mean_pnl,
            "neigh_mean_pf": mean_pf,
            "neigh_max_dd": max_dd,
            "tie_score": tie_score,
        })

    if not enriched:
        print("\nCould not form any neighborhoods. Widen NEIGHBOR_DELTAS or reduce MIN_TRADES_FOR_SELECTION.")
        return

    # Sort by robust objective first, then tie-breakers
    enriched.sort(
        key=lambda x: (
            x["neigh_min_pnl"],      # maximize worst-case pnl
            x["tie_score"],          # then maximize tie score
            x["profit_factor"],      # then raw PF
            -x["max_drawdown"],      # then lower dd (note: -dd means higher is worse, so flip)
            x["trades"],             # then more trades
        ),
        reverse=True
    )

    best = enriched[0]
    best_params = {k: best[k] for k in keys}

    print("\n=== Recommended robust params (stable maximum) ===")
    for k in keys:
        print(f"{k:>15} = {best_params[k]}")
    print("\nRobustness neighborhood:")
    print(f"  neigh_n        = {best['neigh_n']}")
    print(f"  neigh_min_pnl  = {best['neigh_min_pnl']:.4f}   (worst case within small param changes)")
    print(f"  neigh_mean_pnl = {best['neigh_mean_pnl']:.4f}")
    print(f"  neigh_mean_pf  = {best['neigh_mean_pf']:.4f}")
    print(f"  neigh_max_dd   = {best['neigh_max_dd']:.4f}")

    # Show the neighborhood table (compact)
    params_r = best_params
    neigh = [q for q in ok if in_neighbor({k: q[k] for k in keys}, params_r, NEIGHBOR_DELTAS)]
    neigh_sorted = sorted(neigh, key=lambda q: q["total_pnl"], reverse=True)

    print("\nTop neighborhood rows (by total_pnl):")
    for q in neigh_sorted[:10]:
        print(
            f"  EV_MIN={q['EV_MIN']:.2f} PMKT=[{q['PMKT_MIN']:.2f},{q['PMKT_MAX']:.2f}] "
            f"MAX_ENTRY_PRICE={q['MAX_ENTRY_PRICE']:.2f}  trades={q['trades']:2d} "
            f"pnl={q['total_pnl']:+.4f} pf={fmt_inf(q['profit_factor'])} dd={q['max_drawdown']:.3f}"
        )

    print("\nWorst neighborhood rows (by total_pnl):")
    for q in neigh_sorted[-5:]:
        print(
            f"  EV_MIN={q['EV_MIN']:.2f} PMKT=[{q['PMKT_MIN']:.2f},{q['PMKT_MAX']:.2f}] "
            f"MAX_ENTRY_PRICE={q['MAX_ENTRY_PRICE']:.2f}  trades={q['trades']:2d} "
            f"pnl={q['total_pnl']:+.4f} pf={fmt_inf(q['profit_factor'])} dd={q['max_drawdown']:.3f}"
        )

    # -----------------------------
    # Final run with best params and print the original report text
    # -----------------------------
    print("\n=== Final run with recommended params ===")
    # Reload module fresh to avoid any weird state from prior runs
    mod2 = load_test_module(TEST_FILE)
    status, stats, captured = run_backtest_once(mod2, best_params)

    if status != "OK" or stats is None:
        print(f"Final run failed: {status}")
        print(captured)
        return

    # Print the "old file" output (captured stdout from score-and-test.py)
    # This should include your win rate, P&L, PF, max DD lines, and next recommendations.
    print(captured.strip())

    print("\n(Also recomputed from trade_log.csv for sanity)")
    print(f"Trades: {stats['trades']}")
    print(f"Win rate: {stats['winrate']:.3f}")
    print(f"Total P&L: {stats['total_pnl']:.4f}")
    print(f"Avg P&L/trade: {stats['avg_pnl']:.4f}")
    print(f"Profit factor: {fmt_inf(stats['profit_factor'])}")
    print(f"Max drawdown: {stats['max_drawdown']:.4f}")


if __name__ == "__main__":
    main()
