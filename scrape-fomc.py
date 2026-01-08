import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FOMC-Transcript-Scraper/1.0)"
}

PRESS_CONF_TEXT_RE = re.compile(r"^\s*Press\s+Conference\s*$", re.IGNORECASE)
TRANSCRIPT_PDF_TEXT_RE = re.compile(r"Press\s+Conference\s+Transcript.*PDF", re.IGNORECASE)

def fetch(url: str, session: requests.Session, timeout: int = 30) -> str:
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text

def safe_filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name
    return name.lower() or "download.pdf"

def find_press_conference_links_yearpage(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        if PRESS_CONF_TEXT_RE.match(a.get_text(" ", strip=True)):
            links.append(urljoin(base_url, a["href"]))
    return list(set(links)) 

def find_press_conference_links_calendars(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for meeting in soup.select("div.row.fomc-meeting"):
        date_div = meeting.select_one(".fomc-meeting__date")
        date_text = date_div.get_text(" ", strip=True).lower() if date_div else ""

        # Skip unscheduled meetings
        if "unscheduled" in date_text:
            continue

        # Only proceed if this meeting block has a Press Conference link
        a = meeting.find("a", href=True, string=PRESS_CONF_TEXT_RE)
        if a:
            links.append(urljoin(base_url, a["href"]))

    return list(set(links))

def find_transcript_pdf_link(press_conf_html: str, base_url: str) -> str | None:
    soup = BeautifulSoup(press_conf_html, "html.parser")
    for a in soup.find_all("a", href=True):
        if TRANSCRIPT_PDF_TEXT_RE.search(a.get_text(" ", strip=True)):
            return urljoin(base_url, a["href"])
    return None

def download_pdf(pdf_url: str, session: requests.Session, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = session.get(pdf_url, timeout=30)
    r.raise_for_status()

    filename = safe_filename_from_url(pdf_url)
    target = out_dir / filename

    if not target.exists():
        target.write_bytes(r.content)

    return target

def scrape_press_conference_transcripts(
    page_url: str,
    mode: str,  # "yearpage" or "calendars"
    out_dir: str = "data/raw_pdfs",
    sleep_s: float = 0.5,
) -> None:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    out_path = Path(out_dir)

    page_html = fetch(page_url, session)

    if mode == "yearpage":
        pc_links = find_press_conference_links_yearpage(page_html, page_url)
    elif mode == "calendars":
        pc_links = find_press_conference_links_calendars(page_html, page_url)
    else:
        raise ValueError("mode must be 'yearpage' or 'calendars'")

    for pc_url in pc_links:
        try:
            pc_html = fetch(pc_url, session)
            pdf_url = find_transcript_pdf_link(pc_html, pc_url)
            if not pdf_url:
                print(f"[WARN] No transcript PDF found: {pc_url}")
                continue

            local = download_pdf(pdf_url, session, out_path)
            print(f"[OK] {pdf_url} -> {local.name}")
            time.sleep(sleep_s)

        except Exception as e:
            print(f"[ERROR] {pc_url}: {e}")

if __name__ == "__main__":
    # Historic pages (2018/2019 format)
    year_pages = [
        "https://www.federalreserve.gov/monetarypolicy/fomchistorical2018.htm",
        "https://www.federalreserve.gov/monetarypolicy/fomchistorical2019.htm",
    ]
    for yp in year_pages:
        scrape_press_conference_transcripts(yp, mode="yearpage")

    # Calendars page (skip unscheduled)
    scrape_press_conference_transcripts(
        "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        mode="calendars",
    )
