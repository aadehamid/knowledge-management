#!/usr/bin/env python3
"""
Convert PDFs and web pages to Markdown.

Uses two converters:
  - pymupdf4llm for PDFs (lightweight, extracts images in-place)
  - markitdown for web pages / HTML URLs

The script auto-detects the source type from each URL.

Sources are organized by subject area. Each subject is a subdirectory:

    resources/sources/
      transformers/
        urls.txt              # URLs (PDF or HTML) for this subject
        manual_paper.pdf      # or drop PDFs directly
      cuda/
        urls.txt

    references/papers/
      transformers/
        1706.03762.md         # output mirrors the subject structure
      cuda/
        cuda-programming-guide.md

To add a new subject, just create a folder under resources/sources/ and
add a urls.txt or drop PDF files in it.

Incremental: only processes sources that don't already have a corresponding
markdown file in the output directory.

Usage (from repo root):
    python scripts/convert_pdfs.py

Or with custom paths:
    python scripts/convert_pdfs.py --source-dir /path/to/sources --output-dir /path/to/output
"""

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

# ---------------------------------------------------------------------------
# STEP 1: Classify and process URLs
# ---------------------------------------------------------------------------
# URLs can point to PDFs or web pages. We detect the type by checking
# the URL extension and HTTP content-type header. PDFs are downloaded
# locally; web pages are converted directly to markdown via markitdown.
# ---------------------------------------------------------------------------


def sanitize_filename(name: str) -> str:
    """Turn a URL or title into a safe filename."""
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:100]  # cap length


def is_pdf_url(url: str) -> bool:
    """Guess if a URL points to a PDF from its path."""
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")


def download_pdfs_from_urls(url_file: Path, pdf_dir: Path) -> list[Path]:
    """
    Read urls.txt, download any PDFs not already present in pdf_dir.
    Returns list of newly downloaded PDF paths.
    Non-PDF URLs are skipped here (handled by convert_web_urls).
    """
    if not url_file.exists():
        return []

    new_downloads = []
    lines = url_file.read_text().strip().splitlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if not is_pdf_url(line):
            continue  # Web URLs handled separately

        # Derive a filename from the URL
        url_path = urlparse(line).path
        filename = Path(url_path).stem or sanitize_filename(line)
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        dest = pdf_dir / filename

        if dest.exists():
            continue  # Already downloaded

        print(f"  Downloading PDF: {line}")
        try:
            resp = requests.get(line, timeout=60, allow_redirects=True)
            resp.raise_for_status()

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


def convert_web_urls(url_file: Path, output_dir: Path) -> int:
    """
    Convert non-PDF URLs (HTML pages) directly to markdown via markitdown.
    Returns count of successfully converted URLs.
    """
    if not url_file.exists():
        return 0

    from markitdown import MarkItDown

    md_converter = MarkItDown()
    existing_md = {p.stem for p in output_dir.glob("*.md")}
    lines = url_file.read_text().strip().splitlines()
    successes = 0

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if is_pdf_url(line):
            continue  # PDFs handled separately

        # Derive a filename from the URL path
        url_path = urlparse(line).path.rstrip("/")
        stem = Path(url_path).stem or sanitize_filename(line)
        if stem == "index":
            # Use parent directory name for index.html pages
            parts = [p for p in url_path.split("/") if p and p != "index.html"]
            stem = parts[-1] if parts else sanitize_filename(line)

        if stem in existing_md:
            continue  # Already converted

        print(f"  Converting web page: {line}")
        try:
            result = md_converter.convert_url(line)
            md_path = output_dir / f"{stem}.md"
            md_path.write_text(result.text_content, encoding="utf-8")
            print(f"    → {md_path.name}")
            successes += 1
        except Exception as e:
            print(f"    ERROR: {e}", file=sys.stderr)

    return successes


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
            # Create a per-paper image directory
            img_dir = output_dir / f"{pdf.stem}_images"
            img_dir.mkdir(exist_ok=True)

            # write_images=True extracts charts/figures as PNGs and
            # inserts markdown image links at the exact position they
            # appear in the text, preserving the document flow.
            # image_path sets where the PNGs are saved.
            text = pymupdf4llm.to_markdown(
                str(pdf),
                write_images=True,
                image_path=str(img_dir),
                image_format="png",
                dpi=150,
            )

            # Write markdown
            md_path = output_dir / f"{pdf.stem}.md"
            md_path.write_text(text, encoding="utf-8")

            # Count extracted images
            img_count = len(list(img_dir.glob("*.png")))
            if img_count:
                print(f"  → {md_path.name} + {img_count} chart(s)/figure(s)")
            else:
                # Remove empty image dir
                img_dir.rmdir()
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
        description="Convert PDFs and web pages to Markdown (incremental, by subject)"
    )
    parser.add_argument(
        "--source-dir",
        default="resources/sources",
        help="Root directory containing subject subdirectories (default: resources/sources)",
    )
    parser.add_argument(
        "--output-dir",
        default="references/papers",
        help="Root directory for markdown output (default: references/papers)",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    source_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all subject subdirectories
    subjects = discover_subjects(source_dir)
    if not subjects:
        print(f"No subject directories found under {source_dir}/")
        print("Create a subject folder (e.g., resources/sources/transformers/) with a urls.txt to get started.")
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

        # Phase 2: Convert web page URLs directly to markdown
        web_converted = convert_web_urls(url_file, subject_output)
        if web_converted:
            total_converted += web_converted

        # Phase 3: Find and convert unconverted PDFs
        to_convert = find_unconverted_pdfs(subject_dir, subject_output)
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
