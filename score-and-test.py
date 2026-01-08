# score_and_backtest.py
"""
Conservative word scoring + walk-forward backtest + next-trade recommendations.

Assumptions / inputs (you said these exist):
- data/results-v1/predictions.csv
- data/results-v1/metrics_by_target.csv
- Kalshi data exists "in a convenient format"

Because I don't have your exact Kalshi schema here, this script supports TWO simple formats:
A) data/kalshi/historic.csv  (for backtest) with columns:
   date,target,p_mkt
B) data/kalshi/upcoming.csv    (for next recommendation) with columns:
   date,target,p_mkt

Everything is per-target (word) and per-date (meeting).
Profit model: 1 contract, $1 payout, ignore fees/spread.
- If you buy YES at price p and event happens => profit = 1 - p, else profit = -p
- If you buy NO  at price (1-p) and event does NOT happen => profit = 1 - (1-p) = p, else profit = -(1-p)

Outputs:
- data/scoring-v1/trade_log.csv
- data/scoring-v1/equity_curve.csv
- data/scoring-v1/recommendations.csv
and prints a readable summary.

Run:
  python score_and_backtest.py
"""

import csv
import math
from pathlib import Path
from collections import defaultdict
import os

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default

# -----------------------------
# Paths
# -----------------------------
PRED_PATH = Path("data/results-v1/predictions.csv")
METRICS_PATH = Path("data/results-v1/metrics_by_target.csv")

KALSHI_HIST_PATH = Path("data/kalshi/historic.csv")
KALSHI_UPCOMING_PATH = Path("data/kalshi/upcoming.csv")

OUT_DIR = Path("data/scoring-v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRADE_LOG_PATH = OUT_DIR / "trade_log.csv"
EQUITY_PATH = OUT_DIR / "equity_curve.csv"
RECS_PATH = OUT_DIR / "recommendations.csv"

# -----------------------------
# Scoring knobs (tune later)
# -----------------------------

# Use EV-edge instead of abs diff (see compute_opportunity_grade below)
EV_MIN = _env_float("EV_MIN", 0.03)  # minimum expected value per $1 contract (ignore fees)

WORD_GRADE_MIN = _env_float("WORD_GRADE_MIN", 45.0)
OPPORTUNITY_GRADE_MIN = _env_float("OPPORTUNITY_GRADE_MIN", 45.0)

# Hard market-probability filter to avoid "shorting certainty"
PMKT_MIN = _env_float("PMKT_MIN", 0.25)
PMKT_MAX = _env_float("PMKT_MAX", 0.75)

# Optional: require model not too close to 0.5
PMDL_MIN_DIST_FROM_HALF = _env_float("PMDL_MIN_DIST_FROM_HALF", 0.00)

# Missing market data penalty
COVERAGE_PENALTY_WEIGHT = _env_float("COVERAGE_PENALTY_WEIGHT", 30.0)

# Risk kill-switch: avoid paying too much (huge downside if wrong)
MAX_ENTRY_PRICE = _env_float("MAX_ENTRY_PRICE", 0.55)

# Mild selection-bias / "recent performance" cap (you requested)
PERF_NUDGE_CAP = 5.0            # max +/- impact on WordGrade

# Weights inside WordGrade (0..100 scale)
W_SKILL = 45.0
W_STABILITY = 20.0
W_CALIB = 15.0
W_BAYES = 15.0
W_PERF = 5.0   # must remain low weight (your “mild selection bias”)

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

def fint(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def safe_get(row, *keys, default=None):
    for k in keys:
        if k in row and str(row[k]).strip() != "":
            return row[k]
    return default

def sigmoid(z):
    # stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def wilson_lower_bound(successes: int, n: int, z: float = 1.0) -> float:
    """
    Wilson score interval lower bound for Bernoulli proportion.
    z=1.0 ~ 68% (pessimistic but not too harsh); z=1.96 ~ 95% (very conservative).
    """
    if n <= 0:
        return 0.0
    phat = successes / n
    denom = 1.0 + (z*z)/n
    center = phat + (z*z)/(2*n)
    rad = z * math.sqrt((phat*(1-phat) + (z*z)/(4*n)) / n)
    lb = (center - rad) / denom
    return clamp(lb, 0.0, 1.0)

def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    return rows

def index_kalshi(rows):
    """
    Build dict[(date,target)] = pmkt
    Accepts columns:
      - date OR yyyymm (your cleaned file uses yyyymm)
      - target OR word
      - p_mkt
    """
    out = {}
    for row in rows:
        date = str(safe_get(row, "date", "Date", "yyyymm", "YYYYMM", default="")).strip()
        target = str(safe_get(row, "target", "Target", "word", "Word", default="")).strip()
        pmkt = ffloat(safe_get(row, "p_mkt", "pmkt", "p_market", "market_prob", default=""))
        if date and target and pmkt == pmkt:
            out[(date, target)] = clamp(pmkt, 0.0, 1.0)
    return out


# -----------------------------
# Loading predictions + metrics
# -----------------------------
def load_predictions():
    """
    Expect at least: date, target, y, p_raw/p_model/p_platt...
    We'll choose p_mdl in this priority:
      p_platt > p_raw > p_model > p
    """
    rows = load_csv(PRED_PATH)
    preds = []
    for row in rows:
        date = str(safe_get(row, "date", "Date")).strip()
        target = str(safe_get(row, "target", "Target")).strip()
        y = fint(safe_get(row, "y", "Y", "y_any", default=0), 0)

        p = safe_get(row, "p_platt", "p_cal", "p_calibrated", default=None)
        if p is None:
            p = safe_get(row, "p_raw", "p_model", "p", default=None)
        p_mdl = clamp(ffloat(p, default=float("nan")), 0.0, 1.0) if p is not None else float("nan")

        if not date or not target or not (p_mdl == p_mdl):
            continue

        preds.append({"date": date, "target": target, "y": y, "p_mdl": p_mdl})
    preds.sort(key=lambda d: (d["date"], d["target"]))
    return preds

def load_metrics_by_target():
    """
    We don't rely on exact column names; we try to extract:
      - logloss_model, logloss_hist_mean, logloss_ewma_y (optional), brier_model, brier_hist_mean, brier_ewma_y
      - pos_rate, n_total/n_predicted
    If missing, the WordGrade will degrade gracefully.
    """
    rows = load_csv(METRICS_PATH)
    m = {}
    for row in rows:
        target = str(safe_get(row, "target", "Target")).strip()
        if not target:
            continue

        def g(*ks):
            return ffloat(safe_get(row, *ks, default=""), default=float("nan"))

        m[target] = {
            "pos_rate": g("pos_rate", "PosRate"),
            "n_total": fint(safe_get(row, "n_total", "N", "n", default=0), 0),
            "n_predicted": fint(safe_get(row, "n_predicted", "n_pred", default=0), 0),

            "logloss_model": g("logloss_model", "logloss_raw", "logloss_platt", "logloss"),
            "logloss_hist_mean": g("logloss_hist_mean", "logloss_hist"),
            "logloss_ewma_y": g("logloss_ewma_y", "logloss_ewma"),

            "brier_model": g("brier_model", "brier_raw", "brier_platt", "brier"),
            "brier_hist_mean": g("brier_hist_mean", "brier_hist"),
            "brier_ewma_y": g("brier_ewma_y", "brier_ewma"),
        }
    return m

# -----------------------------
# Word grade + opportunity grade
# -----------------------------
def compute_word_grade(target: str, metrics: dict, coverage_rate: float, perf_nudge: float, bayes_lb: float):
    """
    Returns (word_grade_0_100, debug_dict)
    """
    mt = metrics.get(target, {})
    ll_m = mt.get("logloss_model", float("nan"))
    ll_h = mt.get("logloss_hist_mean", float("nan"))
    ll_e = mt.get("logloss_ewma_y", float("nan"))
    br_m = mt.get("brier_model", float("nan"))
    br_h = mt.get("brier_hist_mean", float("nan"))
    br_e = mt.get("brier_ewma_y", float("nan"))

    # Skill vs baselines (logloss + brier)
    # Score in [0,1] where 1 means clearly better than baselines.
    def better_score(model, base):
        if not (model == model) or not (base == base) or base <= 0:
            return 0.5  # unknown -> neutral-ish
        # Positive if model < base
        rel = (base - model) / max(base, 1e-6)
        # squash to [0,1]
        return clamp(0.5 + 0.8 * rel, 0.0, 1.0)

    s_ll_hist = better_score(ll_m, ll_h)
    s_ll_ewma = better_score(ll_m, ll_e) if (ll_e == ll_e) else 0.5
    s_br_hist = better_score(br_m, br_h)
    s_br_ewma = better_score(br_m, br_e) if (br_e == br_e) else 0.5

    skill = (s_ll_hist + s_ll_ewma + s_br_hist + s_br_ewma) / 4.0

    # Stability proxy: avoid extremely rare or extremely common words
    pos = mt.get("pos_rate", float("nan"))
    if pos == pos:
        # max at pos=0.5, decreases toward 0 or 1
        stability = 1.0 - abs(pos - 0.5) / 0.5
        stability = clamp(stability, 0.0, 1.0)
    else:
        stability = 0.5

    # Calibration proxy: brier relative to hist baseline
    calib = better_score(br_m, br_h)

    # Bayesian reliability already computed from realized trades (lower bound)
    bayes = clamp(bayes_lb, 0.0, 1.0)

    # Coverage penalty (smooth, multiplicative effect)
    # coverage_rate in [0,1], penalty in [0,1] where 1 = no penalty
    # If coverage=0 -> huge penalty
    penalty = math.exp(-COVERAGE_PENALTY_WEIGHT * (1.0 - coverage_rate) / 100.0)
    penalty = clamp(penalty, 0.0, 1.0)

    # Combine on 0..100 scale
    word_grade = (
        W_SKILL * skill +
        W_STABILITY * stability +
        W_CALIB * calib +
        W_BAYES * bayes
    )

    # Mild performance nudge, capped
    nudge = clamp(perf_nudge, -PERF_NUDGE_CAP, PERF_NUDGE_CAP)
    word_grade += W_PERF * (nudge / PERF_NUDGE_CAP)  # maps to [-W_PERF, +W_PERF]

    # Apply coverage penalty
    word_grade *= penalty

    dbg = {
        "skill": skill,
        "stability": stability,
        "calib": calib,
        "bayes_lb": bayes_lb,
        "coverage_rate": coverage_rate,
        "coverage_penalty": penalty,
        "perf_nudge": nudge,
        "logloss_model": ll_m,
        "logloss_hist": ll_h,
        "brier_model": br_m,
        "brier_hist": br_h,
    }

    return clamp(word_grade, 0.0, 100.0), dbg

def compute_opportunity_grade(word_grade: float, p_mdl: float, p_mkt: float):
    """
    0..100 grade using EV for the *best* side.

    EV_yes = p_mdl - p_mkt
    EV_no  = p_mkt - p_mdl = -EV_yes

    Choose side that makes EV positive, edge = abs(EV_yes).
    """
    ev_yes = p_mdl - p_mkt
    side_hint = "YES" if ev_yes >= 0 else "NO"
    edge = abs(ev_yes)

    # Convert EV to [0,1] score: 0 at 0, 1 around 2*EV_MIN
    edge_score = clamp(edge / (2 * EV_MIN), 0.0, 1.0)

    # Include word_grade, but weight EV more
    opp = 0.40 * (word_grade / 100.0) + 0.60 * edge_score
    return clamp(opp * 100.0, 0.0, 100.0), edge, side_hint


# -----------------------------
# Trading logic
# -----------------------------
def decide_trade(side: str, p_mkt: float):
    """
    side in {"YES","NO"} decided elsewhere.
    Returns (side, entry_price)
    """
    if side == "YES":
        return "YES", p_mkt
    else:
        return "NO", (1.0 - p_mkt)


def settle_trade(side: str, entry_price: float, y: int):
    """
    y is actual outcome (1 if word said, else 0)
    """
    if side == "YES":
        return (1.0 - entry_price) if y == 1 else (-entry_price)
    else:
        # NO pays out if y==0
        return (1.0 - entry_price) if y == 0 else (-entry_price)

# -----------------------------
# Main
# -----------------------------
def main():
    preds = load_predictions()
    metrics = load_metrics_by_target()

    kal_hist = index_kalshi(load_csv(KALSHI_HIST_PATH)) if KALSHI_HIST_PATH.exists() else {}
    kal_upc = index_kalshi(load_csv(KALSHI_UPCOMING_PATH)) if KALSHI_UPCOMING_PATH.exists() else {}

    # Market coverage per target (historical) - only on dates where Kalshi has *any* market data
    kal_dates = {d for (d, _) in kal_hist.keys()}

    cov_count = defaultdict(int)
    tot_count = defaultdict(int)

    for p in preds:
        if p["date"] not in kal_dates:
            continue  # ignore dates where Kalshi has no data at all
        t = p["target"]
        tot_count[t] += 1
        if (p["date"], t) in kal_hist:
            cov_count[t] += 1

    coverage_rate = {t: (cov_count[t] / tot_count[t]) if tot_count[t] else 0.0 for t in tot_count}


    # Per-target performance state (mild selection bias + Bayesian reliability)
    # We'll track realized trades only.
    traded_wins = defaultdict(int)
    traded_n = defaultdict(int)
    perf_nudge = defaultdict(float)  # in [-PERF_NUDGE_CAP, PERF_NUDGE_CAP], updated slowly

    # Conservative: use Wilson LB of win rate as reliability
    def target_bayes_lb(t):
        if traded_n[t] == 0:
            return 0.5   # neutral prior until you have evidence
        return wilson_lower_bound(traded_wins[t], traded_n[t], z=1.0)


    equity = 0.0
    trade_log = []
    equity_curve = []

    seen = 0
    matched_market = 0
    wg_pass = 0
    edge_pass = 0
    og_pass = 0

    for p in preds:
        seen += 1
        date, target, y, p_mdl = p["date"], p["target"], p["y"], p["p_mdl"]

        if (date, target) not in kal_hist:
            equity_curve.append({"date": date, "equity": f"{equity:.4f}"})
            continue
        matched_market += 1

        p_mkt = kal_hist[(date, target)]
        # --- HARD FILTER: avoid trading extreme market probabilities ---
        if not (PMKT_MIN <= p_mkt <= PMKT_MAX):
            equity_curve.append({"date": date, "equity": f"{equity:.4f}"})
            continue

        # Optional: require model conviction away from 0.5
        if PMDL_MIN_DIST_FROM_HALF > 0.0:
            if abs(p_mdl - 0.5) < PMDL_MIN_DIST_FROM_HALF:
                equity_curve.append({"date": date, "equity": f"{equity:.4f}"})
                continue

        cov = coverage_rate.get(target, 0.0)
        bay_lb = target_bayes_lb(target)

        wg, dbg = compute_word_grade(target, metrics, cov, perf_nudge[target], bay_lb)
        og, edge, side_hint = compute_opportunity_grade(wg, p_mdl, p_mkt)
                    
        did_trade = (wg >= WORD_GRADE_MIN) and (og >= OPPORTUNITY_GRADE_MIN) and (edge >= EV_MIN)

        if did_trade:
            side, entry_price = decide_trade(side_hint, p_mkt)

            # Risk kill-switch: avoid paying too much (huge downside if wrong)
            if entry_price > MAX_ENTRY_PRICE:
                equity_curve.append({"date": date, "equity": f"{equity:.4f}"})
                continue

            pnl = settle_trade(side, entry_price, y)
            equity += pnl

            win = 1 if pnl > 0 else 0
            traded_wins[target] += win
            traded_n[target] += 1

            # Mild selection bias update: small step up/down, capped
            # (wins increase confidence slightly, losses decrease slightly)
            step_up = 0.6
            step_dn = 0.8  # harsher down than up (conservative)
            perf_nudge[target] += (step_up if win else -step_dn)
            perf_nudge[target] = clamp(perf_nudge[target], -PERF_NUDGE_CAP, PERF_NUDGE_CAP)

            trade_log.append({
                "date": date,
                "target": target,
                "y": y,
                "p_mdl": f"{p_mdl:.6f}",
                "p_mkt": f"{p_mkt:.6f}",
                "edge": f"{edge:.6f}",
                "word_grade": f"{wg:.2f}",
                "opp_grade": f"{og:.2f}",
                "side": side,
                "entry_price": f"{entry_price:.6f}",
                "pnl": f"{pnl:.6f}",
                "equity": f"{equity:.6f}",
                "bayes_lb": f"{bay_lb:.6f}",
                "coverage_rate": f"{cov:.3f}",
                "perf_nudge": f"{perf_nudge[target]:.3f}",
            })

        equity_curve.append({"date": date, "equity": f"{equity:.4f}"})

    # Write outputs
    with TRADE_LOG_PATH.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "date","target","y","p_mdl","p_mkt","edge","word_grade","opp_grade",
            "side","entry_price","pnl","equity","bayes_lb","coverage_rate","perf_nudge"
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(trade_log)

    with EQUITY_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date","equity"])
        w.writeheader()
        w.writerows(equity_curve)

    # Summary stats
    n_trades = len(trade_log)
    wins = sum(1 for r in trade_log if ffloat(r["pnl"]) > 0)
    winrate = (wins / n_trades) if n_trades else 0.0
    avg_pnl = (sum(ffloat(r["pnl"]) for r in trade_log) / n_trades) if n_trades else 0.0

    pnls = [ffloat(r["pnl"]) for r in trade_log]
    gross_profit = sum(x for x in pnls if x > 0)
    gross_loss = -sum(x for x in pnls if x < 0)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # Max drawdown on equity curve from trade_log (not per pred)
    eqs = [0.0]
    e = 0.0
    for x in pnls:
        e += x
        eqs.append(e)
    peak = eqs[0]
    max_dd = 0.0
    for v in eqs:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd

    print(f"Gross profit: {gross_profit:.4f}")
    print(f"Gross loss:   {gross_loss:.4f}")
    print(f"Profit factor:{profit_factor:.3f}")
    print(f"Max drawdown: {max_dd:.4f}")

    # Next recommendation (from upcoming.csv if present; otherwise use last date in hist for a “what would I trade now”)
    recs = []
    if kal_upc:
        # find latest upcoming date(s)
        up_dates = sorted({d for (d, _) in kal_upc.keys()})
        next_date = up_dates[0]  # earliest upcoming
        # build latest per-target p_mdl from predictions (last available date)
        last_pred_by_target = {}
        for row in preds:
            last_pred_by_target[row["target"]] = row  # overwrites; preds are date-sorted
        for (d, t), pmkt in kal_upc.items():
            if d != next_date:
                continue
            if t not in last_pred_by_target:
                continue
            p_mdl = last_pred_by_target[t]["p_mdl"]

            # Optional: require model conviction away from 0.5 (same as backtest)
            if PMDL_MIN_DIST_FROM_HALF > 0.0:
                if abs(p_mdl - 0.5) < PMDL_MIN_DIST_FROM_HALF:
                    continue

            cov = coverage_rate.get(t, 0.0)
            bay_lb = target_bayes_lb(t)
            wg, _ = compute_word_grade(t, metrics, cov, perf_nudge[t], bay_lb)
            # Apply the same hard filter in recommendations
            if not (PMKT_MIN <= pmkt <= PMKT_MAX):
                continue

            og, edge, side_hint = compute_opportunity_grade(wg, p_mdl, pmkt)
            side, entry = decide_trade(side_hint, pmkt)

            would_trade = (
                (PMKT_MIN <= pmkt <= PMKT_MAX) and
                (abs(p_mdl - 0.5) >= PMDL_MIN_DIST_FROM_HALF) and
                (wg >= WORD_GRADE_MIN) and
                (og >= OPPORTUNITY_GRADE_MIN) and
                (edge >= EV_MIN) and
                (entry <= MAX_ENTRY_PRICE)
            )

            recs.append({
                "date": d,
                "target": t,
                "p_mdl": f"{p_mdl:.6f}",
                "p_mkt": f"{pmkt:.6f}",
                "edge": f"{edge:.6f}",
                "word_grade": f"{wg:.2f}",
                "opp_grade": f"{og:.2f}",
                "side": side,
                "entry_price": f"{entry:.6f}",
                "coverage_rate": f"{cov:.3f}",
                "bayes_lb": f"{bay_lb:.3f}",
                "perf_nudge": f"{perf_nudge[t]:.3f}",
                "would_trade": "YES" if would_trade else "NO",
            })

            # Sort: would_trade YES first, then by opp_grade desc
            recs.sort(
                key=lambda r: (
                    1 if r["would_trade"] == "YES" else 0,
                    ffloat(r["opp_grade"])
                ),
                reverse=True
)


    with RECS_PATH.open("w", encoding="utf-8", newline="") as f:
        cols = ["date","target","p_mdl","p_mkt","edge","word_grade","opp_grade","side","entry_price","coverage_rate","bayes_lb","perf_nudge","would_trade"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(recs)

    # Print a compact report
    print("\n=== Backtest summary ===")
    print(f"Trades: {n_trades}")
    print(f"Win rate: {winrate:.3f}")
    print(f"Total P&L (1 contract each): {equity:.4f}")
    print(f"Avg P&L per trade: {avg_pnl:.4f}")
    print(f"Wrote: {TRADE_LOG_PATH}")
    print(f"Wrote: {EQUITY_PATH}")

    if recs:
        print("\n=== Next recommendation (YES first, then highest OG) ===")

        yes_recs = [r for r in recs if r["would_trade"] == "YES"]
        no_recs  = [r for r in recs if r["would_trade"] == "NO"]

        top10 = yes_recs[:10]
        if len(top10) < 10:
            top10 += no_recs[:(10 - len(top10))]

        for r in top10:
            print(
                f"{r['date']}  {r['target']:<15} side={r['side']:<3}  "
                f"p_mdl={r['p_mdl']} p_mkt={r['p_mkt']} edge={r['edge']}  "
                f"WG={r['word_grade']} OG={r['opp_grade']} trade={r['would_trade']}"
            )

        print(f"\nWrote: {RECS_PATH}")
    else:
        print("\n(No upcoming Kalshi file found or no matches; create data/kalshi/upcoming.csv to get next-trade picks.)")


if __name__ == "__main__":
    main()
    
