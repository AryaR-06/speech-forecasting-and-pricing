# clean-text.py
import re
from pathlib import Path

IN_DIR = Path("data/extracted")
OUT_DIR = Path("data/cleaned")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NON_ALLOWED_RE = re.compile(r"[^a-z'\s]")      # after normalization, remove everything except a-z, apostrophe, whitespace
MULTISPACE_RE = re.compile(r"\s+")

def normalize(text: str) -> str:
    # normalize quotes
    text = (text.replace("’", "'")
                .replace("‘", "'")
            )

    # lowercase
    text = text.lower()

    # turn hyphens/dashes into spaces (compound words should count when spaced) :contentReference[oaicite:1]{index=1}
    text = re.sub(r"[-–—]", " ", text)

    # remove everything except letters, apostrophes, whitespace
    text = NON_ALLOWED_RE.sub(" ", text)

    # collapse whitespace
    text = MULTISPACE_RE.sub(" ", text).strip()

    return text + "\n"

def main():
    for path in sorted(IN_DIR.glob("*.txt")):
        try:
            cleaned = normalize(path.read_text(encoding="utf-8", errors="ignore"))
            out_path = OUT_DIR / path.name
            out_path.write_text(cleaned, encoding="utf-8")
            print(f"[OK] {path.name} -> {out_path}")
        except Exception as e:
            print(f"[ERR] {path.name}: {e}")

if __name__ == "__main__":
    main()
