#!/usr/bin/env python3
"""
clean-kalshi

Reads raw-kalshi.csv and writes two new files:
  - historic.csv  (status == finalized)
  - upcoming.csv  (status == active)

Each output row has exactly 3 columns:
  1) yyyymm  (from market_ticker like KXFEDMENTION-26JAN-ADP -> 202601)
  2) word    (from title: only if strictly alphabetical, single token, no slash,
              and subtitle is empty)
  3) p_mkt

All other rows are dropped.
"""

import csv
import re
from pathlib import Path

MONTHS = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}

TICKER_RE = re.compile(r"-(\d{2})([A-Z]{3})-")  # e.g. "-26JAN-"
TITLE_RE = re.compile(r"Will Powell say (.+?) at his ", re.IGNORECASE)

def yyyymm_from_market_ticker(market_ticker: str) -> str | None:
    m = TICKER_RE.search(market_ticker or "")
    if not m:
        return None
    yy = m.group(1)
    mon = m.group(2).upper()
    mm = MONTHS.get(mon)
    if not mm:
        return None
    yyyy = f"20{yy}"
    return f"{yyyy}{mm}"

def extract_word(title: str) -> str | None:
    """
    Extract X from: "Will Powell say X at his ..."
    Keep only if:
      - X contains no slash
      - X is a single strictly alphabetical token (A-Z only)
    """
    m = TITLE_RE.search(title or "")
    if not m:
        return None

    x = m.group(1).strip()

    if "/" in x:
        return None

    # alphabetical words, spaces allowed between words
    if not re.fullmatch(r"[A-Za-z]+(?: [A-Za-z]+)*", x):
        return None


    return x

def run(input_path: str = "data/kalshi/raw-kalshi.csv") -> None:
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_dir = in_path.parent
    historic_path = out_dir / "historic.csv"
    upcoming_path = out_dir / "upcoming.csv"

    with in_path.open("r", newline="", encoding="utf-8") as f_in, \
         historic_path.open("w", newline="", encoding="utf-8") as f_hist, \
         upcoming_path.open("w", newline="", encoding="utf-8") as f_up:

        reader = csv.DictReader(f_in)

        # writers (include header; remove if you truly don't want headers)
        w_hist = csv.writer(f_hist)
        w_up = csv.writer(f_up)
        w_hist.writerow(["yyyymm", "word", "p_mkt"])
        w_up.writerow(["yyyymm", "word", "p_mkt"])

        required = {"status", "market_ticker", "title", "subtitle", "p_mkt"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in raw-kalshi.csv: {sorted(missing)}")

        for row in reader:
            status = (row.get("status") or "").strip().lower()
            subtitle = (row.get("subtitle") or "").strip()
            if subtitle != "":
                continue

            yyyymm = yyyymm_from_market_ticker((row.get("market_ticker") or "").strip())
            if not yyyymm:
                continue

            word = extract_word((row.get("title") or "").strip())
            if not word:
                continue
            word = word.lower()

            p_mkt = (row.get("p_mkt") or "").strip()
            if p_mkt == "":
                continue
            p_mkt = float(p_mkt)/100

            out_row = [yyyymm, word, p_mkt]

            if status == "finalized":
                w_hist.writerow(out_row)
            elif status == "active":
                w_up.writerow(out_row)
            else:
                continue

if __name__ == "__main__":
    run()
    print('[Kalshi] Finished cleaning raw-kalshi.csv')
