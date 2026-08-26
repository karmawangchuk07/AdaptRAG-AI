"""
Filter the full StatPearls NXML corpus (already downloaded/extracted from NCBI)
down to cardiovascular chapters, and save each as a simple PDF into data/,
ready for the existing docling ingestion pipeline.

Usage:
    pip install beautifulsoup4 lxml fpdf2
    python filter_statpearls_cardio.py
"""

import os
import re
import glob
from bs4 import BeautifulSoup
from fpdf import FPDF

SOURCE_DIR = "statpearls_raw/statpearls_NBK430685"
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

CARDIO_KEYWORDS = [
    "heart failure", "arrhythmia", "atrial fibrillation", "coronary artery",
    "myocardial infarction", "hypertension", "cardiomyopathy", "angina",
    "statin", "ace inhibitor", "beta blocker", "anticoagulant",
    "heart valve", "valvular", "cardiac", "ischemic heart", "heart block",
    "pericarditis", "endocarditis", "aortic aneurysm", "tachycardia",
    "bradycardia", "echocardiog", "pacemaker", "coronary bypass",
    "cardiovascular", "hyperlipidemia", "cholesterol", "stent", "cardiomyo",
]

EXCLUDE_KEYWORDS = ["biliary", "laryngeal", "tracheal", "esophageal", "ureteral", "urethral", "prostatic"]

_CARDIO_PATTERNS = [re.compile(r"\b" + re.escape(kw)) for kw in CARDIO_KEYWORDS]
_EXCLUDE_PATTERNS = [re.compile(r"\b" + re.escape(kw)) for kw in EXCLUDE_KEYWORDS]


def matches_cardio(title: str) -> bool:
    t = title.lower()
    if any(p.search(t) for p in _EXCLUDE_PATTERNS):
        return False
    return any(p.search(t) for p in _CARDIO_PATTERNS)


def parse_chapter(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml-xml")

    title_tag = soup.find("title-group")
    title = title_tag.find("title").get_text(strip=True) if title_tag and title_tag.find("title") else None
    if not title:
        return None, None

    body = soup.find("body")
    if not body:
        return title, ""

    parts = []
    for el in body.find_all(["title", "p"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name == "title":
            parts.append(f"\n{text.upper()}\n")
        else:
            parts.append(text)

    return title, "\n\n".join(parts)


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\s-]", "", name).strip()
    name = re.sub(r"[\s]+", "_", name)
    return name[:120] if name else "untitled"


def save_as_pdf(title: str, text: str, article_id: str):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 10, title)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)

    safe_text = text.encode("latin-1", "ignore").decode("latin-1")
    pdf.multi_cell(0, 6, safe_text)

    fname = f"{sanitize_filename(title)}_{article_id}.pdf"
    path = os.path.join(OUT_DIR, fname)
    pdf.output(path)
    return path


def main():
    files = glob.glob(os.path.join(SOURCE_DIR, "article-*.nxml"))
    print(f"Found {len(files)} total StatPearls chapters in archive")

    saved, skipped, matched = 0, 0, 0

    for i, path in enumerate(files, 1):
        article_id = os.path.splitext(os.path.basename(path))[0].replace("article-", "")

        try:
            title, text = parse_chapter(path)
        except Exception as e:
            print(f"[{i}/{len(files)}] FAILED to parse {path}: {e}")
            skipped += 1
            continue

        if not title:
            skipped += 1
            continue

        if not matches_cardio(title):
            continue

        matched += 1

        if len(text.strip()) < 200:
            print(f"[{i}/{len(files)}] SKIP (too short): {title}")
            skipped += 1
            continue

        try:
            out_path = save_as_pdf(title, text, article_id)
            saved += 1
            print(f"[{i}/{len(files)}] Saved: {out_path}")
        except Exception as e:
            print(f"[{i}/{len(files)}] FAILED to save PDF for '{title}': {e}")
            skipped += 1

    print(f"\nDone. {matched} cardio-matching chapters found, {saved} PDFs saved, {skipped} skipped/failed.")
    print(f"Output in '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
