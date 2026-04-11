#!/usr/bin/env python3
"""
Convert PDF files to Markdown using pymupdf4llm.

This uses PyMuPDF under the hood — lightweight (no PyTorch), fast, and
produces clean Markdown from digital PDFs. For scanned/image-only PDFs
that need OCR, consider marker-pdf (requires more disk space + GPU).

PDFs are organized by subject area. Each subject is a subdirectory:

    resources/pdfs/
      transformers/
        urls.txt          # URLs for this subject
        manual_paper.pdf  # or drop PDFs directly
      reinforcement-learning/
        urls.txt

    references/papers/
      transformers/
        1706.03762.md     # output mirrors the subject structure
      reinforcement-learning/
        some_paper.md

To add a new subject, just create a folder under resources/pdfs/ and
add a urls.txt or drop PDF files in it.

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
# STEP 3: Convert PDFs to Markdown using pymupdf4llm
# ---------------------------------------------------------------------------
# pymupdf4llm is lightweight (~50MB vs ~10GB for marker-pdf). It uses
# PyMuPDF to extract text with layout awareness and outputs clean
# Markdown. No ML models needed — conversion is near-instant.
# ---------------------------------------------------------------------------


def convert_pdfs(pdfs: list[Path], output_dir: Path) -> int:
    """Convert a list of PDFs to markdown. Returns count of successes."""
    if not pdfs:
        print("No new PDFs to convert.")
        return 0

    import pymupdf4llm

    successes = 0
    for i, pdf in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] Converting: {pdf.name}")
        try:
            text = pymupdf4llm.to_markdown(str(pdf))

            # Write markdown
            md_path = output_dir / f"{pdf.stem}.md"
            md_path.write_text(text, encoding="utf-8")
            print(f"  → {md_path.name}")

            successes += 1
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    return successes


# ---------------------------------------------------------------------------
# STEP 4: Discover subject directories
# ---------------------------------------------------------------------------
# Each subdirectory under pdf_dir is a "subject" (e.g., transformers,
# reinforcement-learning). The script processes each subject independently,
# mirroring the folder structure in the output directory.
# ---------------------------------------------------------------------------


def discover_subjects(pdf_dir: Path) -> list[Path]:
    """Return sorted list of subject subdirectories under pdf_dir."""
    return sorted(
        [d for d in pdf_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )


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
        help="Root directory containing subject subdirectories (default: resources/pdfs)",
    )
    parser.add_argument(
        "--output-dir",
        default="references/papers",
        help="Root directory for markdown output (default: references/papers)",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    pdf_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all subject subdirectories
    subjects = discover_subjects(pdf_dir)
    if not subjects:
        print(f"No subject directories found under {pdf_dir}/")
        print("Create a subject folder (e.g., resources/pdfs/transformers/) with a urls.txt to get started.")
        return

    print(f"Found {len(subjects)} subject(s): {', '.join(s.name for s in subjects)}\n")

    total_converted = 0

    for subject_dir in subjects:
        subject = subject_dir.name
        subject_output = output_dir / subject
        subject_output.mkdir(parents=True, exist_ok=True)
        url_file = subject_dir / "urls.txt"

        print(f"\n{'='*60}")
        print(f"Subject: {subject}")
        print(f"{'='*60}")

        # Phase 1: Download any new PDFs from URLs
        downloaded = download_pdfs_from_urls(url_file, subject_dir)
        if downloaded:
            print(f"  Downloaded {len(downloaded)} new PDF(s)")

        # Phase 2: Find PDFs not yet converted
        to_convert = find_unconverted_pdfs(subject_dir, subject_output)

        # Phase 3: Convert
        if to_convert:
            print(f"  Converting {len(to_convert)} PDF(s)...")
            successes = convert_pdfs(to_convert, subject_output)
            total_converted += successes
        else:
            print("  All PDFs already converted.")

    print(f"\n{'='*60}")
    print(f"Done: {total_converted} new file(s) converted across {len(subjects)} subject(s)")


if __name__ == "__main__":
    main()
