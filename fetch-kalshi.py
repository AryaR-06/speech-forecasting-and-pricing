#!/usr/bin/env python3
"""
Fetch p_mkt for all markets in a series near close time, using ONLY public Kalshi endpoints.

We use Kalshi's built-in feature:
  include_latest_before_start=true

So even if there are no trades/quotes exactly near close, we still get the latest candle
before our target time (if any exists).

Output:
  data/kalshi/historic.csv
"""

import csv
import datetime as dt
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests

BASE_URL = "https://api.elections.kalshi.com"
SERIES_TICKER = "KXFEDMENTION"   # <-- keep your real series ticker here
OUT_PATH = "data/kalshi/raw-kalshi.csv"

PERIOD_INTERVAL = 1                 # 1-minute candles
NEAR_CLOSE_OFFSET_SECONDS = 24 * 60 * 60  # "near close": 1 day before close
END_PAD_SECONDS = 120               # query a tiny window after start


def iso_to_unix_seconds(iso: str) -> Optional[int]:
    if not iso:
        return None
    s = iso.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return int(dt.datetime.fromisoformat(s).timestamp())
    except Exception:
        return None


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def get_json(path: str, params: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    r = requests.get(BASE_URL + path, params=params, timeout=timeout)
    if r.status_code >= 400:
        raise requests.HTTPError(
            f"{r.status_code} {r.reason} for {r.url}\nResponse: {r.text[:500]}",
            response=r,
        )
    return r.json()


def fetch_all_markets_in_series(series_ticker: str) -> List[Dict[str, Any]]:
    markets: List[Dict[str, Any]] = []
    cursor = None
    while True:
        params = {"series_ticker": series_ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        data = get_json("/trade-api/v2/markets", params)
        markets.extend(data.get("markets", []))
        cursor = data.get("cursor")
        if not cursor:
            break
    return markets


def close_val(obj: Optional[Dict[str, Any]]) -> Optional[int]:
    if not obj:
        return None
    v = obj.get("close")
    return v if isinstance(v, int) else None


def compute_p_mkt(yb: Optional[int], ya: Optional[int], pc: Optional[int]) -> Optional[float]:
    if yb is not None and ya is not None:
        return (yb + ya) / 2.0
    if pc is not None:
        return float(pc)
    return None


def fetch_best_candle_near_target(ticker: str, target_ts: int) -> Optional[Dict[str, Any]]:
    """
    Key trick:
      start_ts = target_ts
      include_latest_before_start = true

    That returns:
      - any candles in [target_ts, target_ts+END_PAD_SECONDS]
      - PLUS the latest candle immediately before target_ts (if it exists)

    We then pick the candle with end_period_ts closest to target_ts from below,
    falling back to earliest after if needed.
    """
    params = {
        "market_tickers": ticker,
        "start_ts": target_ts,
        "end_ts": target_ts + END_PAD_SECONDS,
        "period_interval": PERIOD_INTERVAL,
        "include_latest_before_start": "true",
    }
    data = get_json("/trade-api/v2/markets/candlesticks", params)
    markets = data.get("markets", []) or []
    if not markets:
        return None
    candles = markets[0].get("candlesticks", []) or []
    if not candles:
        return None

    # Prefer latest candle with end <= target_ts (closest from below)
    before = [c for c in candles if isinstance(c.get("end_period_ts"), int) and c["end_period_ts"] <= target_ts]
    if before:
        return max(before, key=lambda c: c["end_period_ts"])

    # Otherwise, take earliest candle after target_ts
    after = [c for c in candles if isinstance(c.get("end_period_ts"), int) and c["end_period_ts"] >= target_ts]
    if after:
        return min(after, key=lambda c: c["end_period_ts"])

    return None


def main() -> None:
    print(f"[kalshi] fetching {SERIES_TICKER} markets...", file=sys.stderr)
    markets = fetch_all_markets_in_series(SERIES_TICKER)
    print(f"[kalshi] markets found: {len(markets)}", file=sys.stderr)

    targets = []
    for m in markets:
        ticker = m.get("ticker")
        close_iso = m.get("close_time") or ""
        close_ts = iso_to_unix_seconds(close_iso)
        if not ticker or close_ts is None:
            continue

        target_ts = close_ts - NEAR_CLOSE_OFFSET_SECONDS
        targets.append((m, ticker, close_ts, target_ts))

    print(f"[kalshi] markets with close_time: {len(targets)}", file=sys.stderr)

    ensure_dir(OUT_PATH)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "series_ticker",
            "market_ticker",
            "status",
            "title",
            "subtitle",
            "close_ts",
            "target_ts",
            "selected_candle_end_ts",
            "candle_distance_sec",
            "yes_bid_close",
            "yes_ask_close",
            "price_close",
            "p_mkt",
        ])

        for i, (m, ticker, close_ts, target_ts) in enumerate(targets, start=1):
            try:
                c = fetch_best_candle_near_target(ticker, target_ts)
            except requests.HTTPError as e:
                print(f"[kalshi] WARN {ticker}: {e}", file=sys.stderr)
                c = None

            if c:
                end_period = c.get("end_period_ts")
                dist = (end_period - target_ts) if isinstance(end_period, int) else ""
                yb = close_val(c.get("yes_bid"))
                ya = close_val(c.get("yes_ask"))
                pc = close_val(c.get("price"))
                p_mkt = compute_p_mkt(yb, ya, pc)
            else:
                end_period = dist = yb = ya = pc = p_mkt = ""

            writer.writerow([
                SERIES_TICKER,
                ticker,
                m.get("status", ""),
                m.get("title", ""),
                m.get("subtitle", ""),
                close_ts,
                target_ts,
                end_period,
                dist,
                yb,
                ya,
                pc,
                p_mkt,
            ])

            if i % 25 == 0:
                time.sleep(0.15)

    print(f"[kalshi] wrote {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
