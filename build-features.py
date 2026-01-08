# build_features.py
import re
import csv
from pathlib import Path
from datetime import datetime
import math

CLEAN_DIR = Path("data/cleaned")
OUT_DIR = Path("data/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KALSHI_DIR = Path("data/kalshi")
FILES = ["historic.csv", "upcoming.csv"]

words = set()

for fname in FILES:
    path = KALSHI_DIR / fname
    if not path.exists():
        continue

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            word = row[1].strip().lower()
            if word:
                words.add(word)

TARGETS = list(words)

SINCE_LAST_CAP = 6

# EWMA parameters
ALPHA_LEVEL = 0.75      # for ew_intro_rate, ew_qna_rate
ALPHA_SHORT = 0.60      # for trend_any
ALPHA_LONG = 0.90       # for trend_any

DATE_RE = re.compile(r"(\d{8})")  # find YYYYMMDD anywhere in filename


def parse_date_from_name(name: str) -> str:
    m = DATE_RE.search(name)
    if not m:
        raise ValueError(f"No YYYYMMDD date found in filename: {name}")
    yyyymmdd = m.group(1)
    # Validate date
    datetime.strptime(yyyymmdd, "%Y%m%d")
    return yyyymmdd


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def contains_target(text: str, target: str) -> bool:
    """
    For single words: exact token match.
    For phrases: exact phrase match on token boundaries.
    Assumes text is normalized: a-z, space, apostrophe only.
    """
    target = target.strip().lower()
    if not target:
        return False

    if " " in target:
        # phrase match on boundaries
        hay = f" {text} "
        needle = f" {target} "
        return needle in hay
    else:
        # single word token match
        # (split is fine because cleaned text has only spaces/apostrophes/letters)
        return target in set(text.split())


def ewma_update(prev: float, y_prev: int, alpha: float) -> float:
    """EWMA_t = alpha*EWMA_{t-1} + (1-alpha)*y_{t-1}"""
    return alpha * prev + (1 - alpha) * y_prev


def load_meetings():
    """
    Returns a list of meetings: [(date_str, intro_text, qna_text), ...] sorted by date.
    Requires both *_intro.txt and *_qna.txt for each meeting.
    """
    intro_files = sorted(CLEAN_DIR.glob("*_intro.txt"))
    meetings = {}

    for intro_path in intro_files:
        stem = intro_path.name
        date = parse_date_from_name(stem)
        base = intro_path.name.replace("_intro.txt", "")
        qna_path = CLEAN_DIR / f"{base}_qna.txt"
        if not qna_path.exists():
            continue
        meetings[date] = (read_text(intro_path), read_text(qna_path))

    out = [(d, meetings[d][0], meetings[d][1]) for d in sorted(meetings.keys())]
    if not out:
        raise RuntimeError("No meetings found. Check data/cleaned contains *_intro.txt and *_qna.txt pairs.")
    return out


def main():
    meetings = load_meetings()

    out_path = OUT_DIR / "features.csv"
    fieldnames = [
        "date",
        "target",
        "y_any",
        "y_intro",
        "y_qna",
        "ew_intro_rate",
        "ew_qna_rate",
        "since_last_any_log",
        "trend_any",
        "surprise_any"
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for target in TARGETS:
            # Per-target state (time-safe)
            ew_intro = 0.0
            ew_qna = 0.0
            ew_any_short = 0.0
            ew_any_long = 0.0
            last_any_idx = None  # meeting index of last any-mention
            prev_y_any = 0

            for i, (date, intro_text, qna_text) in enumerate(meetings):
                # Labels for THIS meeting (used only for output + future updates)
                y_intro = int(contains_target(intro_text, target))
                y_qna = int(contains_target(qna_text, target))
                y_any = int(y_intro or y_qna)

                # Features for THIS meeting must use only past info (< i)
                # Fix 2: cap + log-transform since_last
                since_last_any = i if last_any_idx is None else (i - last_any_idx)
                since_last_capped = min(since_last_any, SINCE_LAST_CAP)
                since_last_any_log = math.log1p(since_last_capped)

                trend_any = ew_any_short - ew_any_long

                # Fix 1: surprise feature (uses only past info)
                # "expected" is ew_any_long (computed up to t-1), and prev_y_any is y_{t-1}
                surprise_any = abs(prev_y_any - ew_any_long)

                w.writerow({
                    "date": date[0:6],
                    "target": target,
                    "y_any": y_any,
                    "y_intro": y_intro,
                    "y_qna": y_qna,
                    "ew_intro_rate": f"{ew_intro:.6f}",
                    "ew_qna_rate": f"{ew_qna:.6f}",
                    "since_last_any_log": f"{since_last_any_log:.6f}",
                    "trend_any": f"{trend_any:.6f}",
                    "surprise_any": f"{surprise_any:.6f}",
                })

                # Update state AFTER writing row (so next row is out-of-sample w.r.t. this meeting)
                ew_intro = ewma_update(ew_intro, y_intro, ALPHA_LEVEL)
                ew_qna = ewma_update(ew_qna, y_qna, ALPHA_LEVEL)
                ew_any_short = ewma_update(ew_any_short, y_any, ALPHA_SHORT)
                ew_any_long = ewma_update(ew_any_long, y_any, ALPHA_LONG)

                if y_any == 1:
                    last_any_idx = i
                
                prev_y_any = y_any

    print(f"[OK] Wrote features to: {out_path}")


if __name__ == "__main__":
    main()
