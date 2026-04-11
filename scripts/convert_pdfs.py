#!/usr/bin/env python3
"""
Convert PDF files to Markdown using marker-pdf.

Supports two input modes:
  1. PDF files manually placed in resources/pdfs/
  2. URLs listed in resources/pdfs/urls.txt (one per line)

Incremental: only processes PDFs that don't already have a corresponding
markdown file in the output directory.

Usage (from repo root):
    python scripts/convert_pdfs.py

Or with custom paths:
    python scripts/convert_pdfs.py --pdf-dir /path/to/pdfs --output-dir /path/to/output
"""

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

# ---------------------------------------------------------------------------
# STEP 1: Download PDFs from URLs
# ---------------------------------------------------------------------------
# Why a separate step? Some PDFs live on the web (arxiv, blogs, etc.) and
# we want to fetch them once, save them locally, and then convert. This
# keeps the conversion step uniform — it always works on local files.
# ---------------------------------------------------------------------------


def sanitize_filename(name: str) -> str:
    """Turn a URL or title into a safe filename."""
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:100]  # cap length


def download_pdfs_from_urls(url_file: Path, pdf_dir: Path) -> list[Path]:
    """
    Read urls.txt, download any PDFs not already present in pdf_dir.
    Returns list of newly downloaded PDF paths.

    Each line in urls.txt is either:
      - A direct PDF URL (https://arxiv.org/pdf/2301.00001.pdf)
      - A URL that redirects to a PDF
    Blank lines and lines starting with # are skipped.
    """
    if not url_file.exists():
        return []

    new_downloads = []
    lines = url_file.read_text().strip().splitlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Derive a filename from the URL
        url_path = urlparse(line).path
        filename = Path(url_path).stem or sanitize_filename(line)
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        dest = pdf_dir / filename

        if dest.exists():
            continue  # Already downloaded

        print(f"  Downloading: {line}")
        try:
            resp = requests.get(line, timeout=60, allow_redirects=True)
            resp.raise_for_status()

            # Verify we actually got a PDF (not an HTML error page)
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and not resp.content[:5] == b"%PDF-":
                print(f"    SKIP: URL did not return a PDF ({content_type})")
                continue

            dest.write_bytes(resp.content)
            new_downloads.append(dest)
            print(f"    → Saved: {dest.name}")
        except Exception as e:
            print(f"    ERROR downloading: {e}", file=sys.stderr)

    return new_downloads


# ---------------------------------------------------------------------------
# STEP 2: Find PDFs that need conversion (incremental check)
# ---------------------------------------------------------------------------
# Why incremental? We don't want to re-convert PDFs that were already
# processed in a previous run. We check if a .md file with the same stem
# already exists in the output directory.
# ---------------------------------------------------------------------------


def find_unconverted_pdfs(pdf_dir: Path, output_dir: Path) -> list[Path]:
    """Return PDFs that don't yet have a matching .md in output_dir."""
    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    existing_md = {p.stem for p in output_dir.glob("*.md")}
    return [p for p in all_pdfs if p.stem not in existing_md]


# ---------------------------------------------------------------------------
# STEP 3: Convert PDFs to Markdown using Marker
# ---------------------------------------------------------------------------
# Marker loads several ML models (OCR, layout detection, table recognition)
# into memory. We initialize them ONCE and reuse across all files. This is
# important — loading models is slow (~10-30s), but conversion per file is
# fast once models are loaded.
# ---------------------------------------------------------------------------


def convert_pdfs(pdfs: list[Path], output_dir: Path) -> int:
    """Convert a list of PDFs to markdown. Returns count of successes."""
    if not pdfs:
        print("No new PDFs to convert.")
        return 0

    # Lazy import so the script starts fast (model loading is heavy)
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    print("Loading Marker models (this may take a moment on first run)...")
    converter = PdfConverter(artifact_dict=create_model_dict())

    successes = 0
    for i, pdf in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] Converting: {pdf.name}")
        try:
            rendered = converter(str(pdf))
            text, _, images = text_from_rendered(rendered)

            # Write markdown
            md_path = output_dir / f"{pdf.stem}.md"
            md_path.write_text(text, encoding="utf-8")

            # Save extracted images (diagrams, figures, etc.)
            if images:
                img_dir = output_dir / f"{pdf.stem}_images"
                img_dir.mkdir(exist_ok=True)
                for img_name, img_data in images.items():
                    img_data.save(str(img_dir / img_name))
                print(f"  → {md_path.name} + {len(images)} images")
            else:
                print(f"  → {md_path.name}")

            successes += 1
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    return successes


# ---------------------------------------------------------------------------
# MAIN: tie it all together
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDFs to Markdown (incremental, with URL support)"
    )
    parser.add_argument(
        "--pdf-dir",
        default="resources/pdfs",
        help="Directory containing PDFs and urls.txt (default: resources/pdfs)",
    )
    parser.add_argument(
        "--output-dir",
        default="references/papers",
        help="Directory for markdown output (default: references/papers)",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    url_file = pdf_dir / "urls.txt"

    pdf_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Download any new PDFs from URLs
    print("=== Phase 1: Checking for new URLs to download ===")
    downloaded = download_pdfs_from_urls(url_file, pdf_dir)
    print(f"Downloaded {len(downloaded)} new PDF(s)\n")

    # Phase 2: Find PDFs not yet converted
    print("=== Phase 2: Finding unconverted PDFs ===")
    to_convert = find_unconverted_pdfs(pdf_dir, output_dir)
    print(f"Found {len(to_convert)} PDF(s) to convert\n")

    # Phase 3: Convert
    if to_convert:
        print("=== Phase 3: Converting to Markdown ===")
        successes = convert_pdfs(to_convert, output_dir)
        print(f"\nDone: {successes}/{len(to_convert)} converted successfully")
    else:
        print("Nothing new to convert. All PDFs already have markdown output.")


if __name__ == "__main__":
    main()
