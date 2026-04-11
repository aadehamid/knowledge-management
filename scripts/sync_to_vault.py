#!/usr/bin/env python3
"""
Sync converted markdown + images from the repo to your Obsidian vault.

This is a LOCAL-ONLY script (not for cloud agents). It reads
sync_config.json to map each subject to a vault destination path,
then copies new/updated files into the vault's Raw/ folder with
proper frontmatter and naming for the wiki schema.

What it does:
  1. Reads sync_config.json for subject → vault path mapping
  2. For each subject, finds .md files in references/papers/<subject>/
  3. Reads the .meta.json sidecar (if present) for title, url, source_type
  4. Renames the file to <source_type>-<slug>.md (wiki schema convention)
  5. Prepends YAML frontmatter to the FULL markdown body (no summarization)
  6. Copies the renamed .md + any _images/ folder to the vault path

Only subjects with a non-empty path in sync_config.json are synced.

Usage (from repo root):
    python scripts/sync_to_vault.py
"""

import json
import re
import shutil
import sys
from pathlib import Path


def slugify(title: str) -> str:
    """Convert a title to kebab-case slug for wiki filenames."""
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:80]


def build_frontmatter(meta: dict) -> str:
    """Build YAML frontmatter string from metadata."""
    lines = ["---"]
    lines.append(f"url: {meta.get('url', '')}")
    lines.append(f"title: {meta.get('title', 'Untitled')}")
    if meta.get("author"):
        lines.append(f"author: {meta['author']}")
    lines.append(f"source_type: {meta.get('source_type', 'doc')}")
    lines.append("status: ingested")
    lines.append(f"fetched_at: {meta.get('fetched_at', '')}")
    lines.append("wiki_refs: []")
    lines.append("---")
    return "\n".join(lines)


def wiki_filename(meta: dict, original_stem: str) -> str:
    """
    Generate wiki-schema filename: <source_type>-<slug>.md
    Uses title for slug if available, otherwise falls back to original stem.
    """
    source_type = meta.get("source_type", "doc")
    title = meta.get("title", "")
    slug = slugify(title) if title else slugify(original_stem)
    return f"{source_type}-{slug}.md"


def sync_subject(subject: str, source_dir: Path, vault_dir: Path) -> int:
    """
    Copy new/updated markdown + images from source_dir to vault_dir.
    Adds frontmatter and renames files per wiki schema.
    Returns count of files synced.
    """
    if not source_dir.exists():
        return 0

    vault_dir.mkdir(parents=True, exist_ok=True)
    synced = 0

    for md_file in sorted(source_dir.glob("*.md")):
        # Load metadata sidecar if it exists
        meta_file = source_dir / f"{md_file.stem}.meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
        else:
            # Fallback: minimal metadata from filename
            meta = {
                "url": "",
                "title": md_file.stem,
                "source_type": "doc",
                "fetched_at": "",
            }

        # Determine the wiki-schema filename
        dest_name = wiki_filename(meta, md_file.stem)
        dest = vault_dir / dest_name

        # Skip if already in vault and source hasn't changed
        if dest.exists() and dest.stat().st_mtime >= md_file.stat().st_mtime:
            continue

        # Read the full original markdown body
        original_body = md_file.read_text(encoding="utf-8")

        # Fix image paths: update references from the original image
        # directory name to the wiki-schema-based name so Obsidian
        # can find them in the same folder.
        original_img_dir_name = f"{md_file.stem}_images"
        wiki_stem = dest_name.removesuffix(".md")
        new_img_dir_name = f"{wiki_stem}_images"
        if original_img_dir_name != new_img_dir_name:
            original_body = original_body.replace(
                original_img_dir_name, new_img_dir_name
            )

        # Build frontmatter + full body
        frontmatter = build_frontmatter(meta)
        full_content = f"{frontmatter}\n\n{original_body}"

        # Write to vault
        dest.write_text(full_content, encoding="utf-8")
        print(f"  → {dest_name}")
        synced += 1

        # Copy associated images folder if it exists
        img_dir = source_dir / original_img_dir_name
        if img_dir.exists():
            dest_img_dir = vault_dir / new_img_dir_name
            if dest_img_dir.exists():
                shutil.rmtree(dest_img_dir)
            shutil.copytree(img_dir, dest_img_dir)
            img_count = len(list(dest_img_dir.glob("*")))
            print(f"    + {img_count} image(s)")

    return synced


def main():
    # Resolve paths relative to the repo root (parent of scripts/)
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "sync_config.json"
    papers_dir = repo_root / "references" / "papers"

    if not config_path.exists():
        print(f"Error: {config_path} not found", file=sys.stderr)
        sys.exit(1)

    config = json.loads(config_path.read_text())
    subjects = config.get("subjects", {})

    if not subjects:
        print("No subjects configured in sync_config.json")
        return

    total_synced = 0

    for subject, vault_path_str in subjects.items():
        if not vault_path_str:
            print(f"\n[{subject}] No vault path configured — skipping")
            continue

        vault_path = Path(vault_path_str)
        source_path = papers_dir / subject

        print(f"\n[{subject}]")
        print(f"  From: {source_path}")
        print(f"  To:   {vault_path}")

        synced = sync_subject(subject, source_path, vault_path)
        if synced:
            total_synced += synced
        else:
            print("  Already up to date.")

    print(f"\nDone: {total_synced} file(s) synced to Obsidian vault")


if __name__ == "__main__":
    main()
