import re
from pathlib import Path
import pdfplumber

RAW_DIR = Path("data/raw_pdfs")
OUT_DIR = Path("data/extracted")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Speaker detection
# -----------------------------
SPEAKER_LINE_RE = re.compile(
    r"^([A-Z][A-Z'\-]*(?:\s+[A-Z][A-Z'\-]*)+)[\.:]\s+(.*)$"
)

POWELL_SPEAKERS = {"CHAIR POWELL", "CHAIRMAN POWELL"}

# -----------------------------
# Noise / cleanup patterns
# -----------------------------
PAGE_RE = re.compile(r"^Page\s+\d+\s+of\s+\d+$", re.IGNORECASE)

DATE_RE = re.compile(
    r"^(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+\d{4}$",
    re.IGNORECASE
)

TITLE_RE = re.compile(
    r"Chair(?:man)?\s+Powell['’]s\s+Press\s+Conference",
    re.IGNORECASE
)

FINAL_RE = re.compile(r"^FINAL$", re.IGNORECASE)
FOOTNOTE_RE = re.compile(r"^\d+\s+[A-Z].+")
BRACKETED_RE = re.compile(r"\[[^\]]*\]")

def is_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if PAGE_RE.match(s):
        return True
    if DATE_RE.match(s):
        return True
    if TITLE_RE.search(s):
        return True
    if FINAL_RE.match(s):
        return True
    if FOOTNOTE_RE.match(s):
        return True
    return False

# -----------------------------
# Extraction logic (MODIFIED)
# -----------------------------
def extract_powell_intro_qna(pdf_path: Path):
    intro = []
    qna = []

    current_speaker = None
    powell_block_count = 0  # NEW

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw_line in text.splitlines():
                line = raw_line.rstrip()

                if is_noise(line):
                    continue

                m = SPEAKER_LINE_RE.match(line)
                if m:
                    current_speaker = m.group(1)
                    spoken = BRACKETED_RE.sub("", m.group(2)).strip()

                    if current_speaker in POWELL_SPEAKERS:
                        powell_block_count += 1  # NEW
                        if spoken:
                            if powell_block_count == 1:
                                intro.append(spoken)
                            else:
                                qna.append(spoken)
                    continue

                if current_speaker in POWELL_SPEAKERS:
                    cleaned = BRACKETED_RE.sub("", line).strip()
                    if cleaned:
                        if powell_block_count == 1:
                            intro.append(cleaned)
                        else:
                            qna.append(cleaned)

    return (
        "\n".join(intro).strip() + "\n",
        "\n".join(qna).strip() + "\n",
    )

# -----------------------------
# Main loop (MODIFIED)
# -----------------------------
def main():
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    for pdf in pdfs:
        try:
            intro_text, qna_text = extract_powell_intro_qna(pdf)

            (OUT_DIR / f"{pdf.stem}_intro.txt").write_text(
                intro_text, encoding="utf-8"
            )
            (OUT_DIR / f"{pdf.stem}_qna.txt").write_text(
                qna_text, encoding="utf-8"
            )

            print(f"[OK] {pdf.name} -> intro + qna")

        except Exception as e:
            print(f"[ERR] {pdf.name}: {e}")

if __name__ == "__main__":
    main()
