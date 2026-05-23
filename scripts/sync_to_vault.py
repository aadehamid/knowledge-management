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
import subprocess
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


def subject_to_display_name(subject: str) -> str:
    """Convert a kebab-case subject key to a title-case display name.
    e.g. 'llm-inference-optimization' -> 'LLM Inference Optimization'
    """
    acronyms = {"llm", "cuda", "gpu", "nlp", "ml", "ai", "gpt", "rag", "rlhf"}
    words = subject.replace("-", " ").replace("_", " ").split()
    return " ".join(w.upper() if w.lower() in acronyms else w.capitalize() for w in words)


def create_notebooklm_notebook(display_name: str) -> str | None:
    """Create a new NotebookLM notebook and return its ID, or None on failure."""
    try:
        result = subprocess.run(
            ["nlm", "notebook", "create", display_name],
            capture_output=True,
            text=True,
            timeout=60,
        )
        combined = result.stdout + result.stderr
        # Parse notebook ID from output like: "Created notebook: <id>"
        match = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", combined)
        if match:
            notebook_id = match.group(0)
            print(f"  NLM:  Created new notebook '{display_name}' → {notebook_id}")
            return notebook_id
        print(f"  NLM:  Failed to create notebook '{display_name}': {combined.strip()}")
        return None
    except FileNotFoundError:
        print("  NLM:  nlm CLI not found — skipping notebook creation")
        return None
    except subprocess.TimeoutExpired:
        print("  NLM:  nlm CLI timed out during notebook creation")
        return None


def add_to_notebooklm(notebook_id: str, file_path: Path) -> bool:
    """Add a markdown file as a source to a NotebookLM notebook via the nlm CLI.

    Correct syntax (nlm 0.6.x+):
        nlm source add <notebook_id> --file <file_path>
    """
    try:
        result = subprocess.run(
            ["nlm", "source", "add", notebook_id, "--file", str(file_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            print(f"    📓 Added to NotebookLM")
            return True
        print(f"    ⚠️  NotebookLM add failed: {result.stderr.strip() or result.stdout.strip()}")
        return False
    except FileNotFoundError:
        print("    ⚠️  nlm CLI not found — skipping NotebookLM sync")
        return False
    except subprocess.TimeoutExpired:
        print("    ⚠️  nlm CLI timed out — skipping")
        return False


def sync_subject(
    subject: str,
    source_dir: Path,
    vault_dir: Path,
    notebooklm_id: str | None = None,
) -> int:
    """
    Copy new/updated markdown + images from source_dir to vault_dir.
    Adds frontmatter and renames files per wiki schema.
    Optionally adds new files to a NotebookLM notebook.
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

        # Slug-collision guard: if the title-derived dest doesn't exist yet,
        # check whether a vault file with the same URL-stem slug already exists
        # under a different name (e.g., different source_type prefix or old slug).
        # If so, reuse that file to avoid creating a duplicate.
        url_stem_slug = slugify(md_file.stem)
        if not dest.exists() and url_stem_slug not in dest_name:
            for candidate in sorted(vault_dir.glob(f"*-{url_stem_slug}.md")):
                dest = candidate
                dest_name = candidate.name
                break

        # Skip if already in vault and source hasn't changed
        if dest.exists() and dest.stat().st_mtime >= md_file.stat().st_mtime:
            continue

        # Read the full original markdown body
        original_body = md_file.read_text(encoding="utf-8")

        # Fix image paths: pymupdf4llm may generate paths like
        # "references/papers/transformers/1706.03762_images/img.png"
        # but in the vault, images sit next to the .md file. We use
        # regex to strip any path prefix before the _images/ folder
        # and rename to the wiki-schema-based directory name.
        original_img_dir_name = f"{md_file.stem}_images"
        wiki_stem = dest_name.removesuffix(".md")
        new_img_dir_name = f"{wiki_stem}_images"

        # Match any path prefix ending with the original _images dir
        # e.g. "references/papers/transformers/1706.03762_images/"
        # or just "1706.03762_images/" — replace with new name only
        original_body = re.sub(
            r"[^\s()!\[\]]*" + re.escape(original_img_dir_name),
            new_img_dir_name,
            original_body,
        )

        # Build frontmatter + full body
        frontmatter = build_frontmatter(meta)
        full_content = f"{frontmatter}\n\n{original_body}"

        # Write to vault
        dest.write_text(full_content, encoding="utf-8")
        print(f"  → {dest_name}")
        synced += 1

        # Add to NotebookLM if configured
        if notebooklm_id:
            add_to_notebooklm(notebooklm_id, md_file)

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
    notebooklm = config.get("notebooklm", {})

    if not subjects:
        print("No subjects configured in sync_config.json")
        return

    # Auto-provision NotebookLM notebooks for any new subjects not yet configured.
    config_updated = False
    for subject in subjects:
        if subject not in notebooklm:
            display_name = subject_to_display_name(subject)
            print(f"\n[{subject}] No NotebookLM notebook configured — creating '{display_name}'...")
            notebook_id = create_notebooklm_notebook(display_name)
            if notebook_id:
                notebooklm[subject] = notebook_id
                config["notebooklm"] = notebooklm
                config_updated = True

    # Persist any newly created notebook IDs back to sync_config.json.
    if config_updated:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        print("  Updated sync_config.json with new notebook IDs.")

    total_synced = 0

    for subject, vault_path_str in subjects.items():
        if not vault_path_str:
            print(f"\n[{subject}] No vault path configured — skipping")
            continue

        vault_path = Path(vault_path_str)
        source_path = papers_dir / subject
        nlm_id = notebooklm.get(subject)

        print(f"\n[{subject}]")
        print(f"  From: {source_path}")
        print(f"  To:   {vault_path}")
        if nlm_id:
            print(f"  NLM:  {nlm_id}")

        synced = sync_subject(subject, source_path, vault_path, notebooklm_id=nlm_id)
        if synced:
            total_synced += synced
        else:
            print("  Already up to date.")

    print(f"\nDone: {total_synced} file(s) synced to Obsidian vault")


if __name__ == "__main__":
    main()
