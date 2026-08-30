#!/usr/bin/env python3
"""Post-ingest self-check for a vault bundle.

Turns the parts of the ingest QA loop that are mechanical into a command, so an
agent does not have to remember them. Every check here exists because a real
review caught the failure at least once.

    python3 scripts/check_ingest.py "<vault path>"
    python3 scripts/check_ingest.py "<vault path>" --since 2026-08-30

Exit 0 = clean. Exit 1 = problems listed. This does NOT check factual accuracy
against the sources; only an independent reviewer does that.
"""
import argparse
import datetime
import os
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

ERRORS: list[str] = []
WARNINGS: list[str] = []


def err(msg):
    ERRORS.append(msg)


def warn(msg):
    WARNINGS.append(msg)


def frontmatter(path: Path):
    """Return (frontmatter_text, body) or (None, text) when there is no frontmatter."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, text
    try:
        end = lines.index("---", 1)
    except ValueError:
        return None, text
    return "\n".join(lines[1:end]), "\n".join(lines[end + 1:])


def check_okf(vault: Path, repo: Path):
    """Run the OKF conformance battery and surface its output."""
    script = repo / "scripts" / "test_okf_bundle.py"
    if not script.exists():
        warn(f"OKF battery not found at {script}")
        return
    r = subprocess.run([sys.executable, str(script), str(vault)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        err("OKF conformance FAILED:\n    " + r.stdout.strip().replace("\n", "\n    "))
    else:
        print("  ok  " + r.stdout.strip())


def check_links(vault: Path):
    """Every relative markdown link outside code spans must resolve."""
    checked = broken = 0
    for p in list((vault / "Wiki").rglob("*.md")) + list((vault / "Learning Path").glob("*.md")):
        text = re.sub(r"`[^`]*`", "", p.read_text(encoding="utf-8", errors="ignore"))
        for m in re.finditer(r"\[[^\]]+\]\(([^)#]+?\.md)\)", text):
            href = m.group(1)
            if href.startswith(("http://", "https://")):
                continue
            checked += 1
            target = os.path.normpath(os.path.join(p.parent, urllib.parse.unquote(href)))
            if not os.path.exists(target):
                broken += 1
                err(f"broken link: {p.relative_to(vault)} -> {href}")
    print(f"  ok  {checked} internal links checked, {broken} broken")


def check_wiki_refs(vault: Path):
    """A Raw file's wiki_refs must name pages that really cite it, and vice versa."""
    raw_dir = vault / "Raw"
    if not raw_dir.is_dir():
        return
    wiki_text = {}
    for p in (vault / "Wiki").rglob("*.md"):
        wiki_text[str(p.relative_to(vault))] = p.read_text(encoding="utf-8", errors="ignore")

    listed = consistent = 0
    for raw in sorted(raw_dir.glob("*.md")):
        fm, _ = frontmatter(raw)
        if fm is None:
            continue
        # page names contain spaces ("Reward Model.md"), so \S+ is wrong here
        refs = re.findall(r"^\s*-\s+(Wiki/.+?\.md)\s*$", fm, re.M)
        for ref in refs:
            listed += 1
            if ref not in wiki_text:
                err(f"wiki_refs points at a missing page: {raw.name} -> {ref}")
            elif raw.name not in wiki_text[ref]:
                err(f"wiki_refs not reciprocated: {raw.name} lists {ref}, "
                    f"but that page does not cite the source")
            else:
                consistent += 1
        # the reverse: a page citing this source should be listed
        citers = [rel for rel, t in wiki_text.items() if raw.name in t]
        missing = [c for c in citers if c not in refs]
        if missing and refs:
            warn(f"{raw.name}: cited by {missing} but they are not in its wiki_refs")
    print(f"  ok  wiki_refs: {consistent}/{listed} reciprocated")


def check_stamps(vault: Path, since: str | None):
    """Ingest-touched wiki pages need a truthful `generated` stamp."""
    now = datetime.datetime.now(datetime.timezone.utc)
    pages = list((vault / "Wiki").rglob("*.md")) + list((vault / "Learning Path").glob("*.md"))
    stamped = future = 0
    for p in pages:
        if p.name in ("index.md", "log.md"):
            continue
        fm, _ = frontmatter(p)
        if fm is None:
            continue
        updated = re.search(r"^updated:\s*(\S+)", fm, re.M)
        gen = re.search(r'generated:\s*\{[^}]*at:\s*"?([0-9T:\-Z]+)"?', fm)
        if since and updated and updated.group(1) >= since and not gen:
            err(f"missing `generated` stamp on a page updated in this run: {p.relative_to(vault)}")
        if gen:
            stamped += 1
            try:
                at = datetime.datetime.strptime(gen.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=datetime.timezone.utc)
            except ValueError:
                err(f"unparseable generated timestamp: {p.relative_to(vault)}")
                continue
            mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime, datetime.timezone.utc)
            if at > now + datetime.timedelta(minutes=1):
                future += 1
                err(f"generated stamp is in the FUTURE: {p.relative_to(vault)} ({gen.group(1)})")
            elif at > mtime + datetime.timedelta(hours=1):
                warn(f"generated stamp is well after the file mtime: {p.relative_to(vault)}")
    print(f"  ok  {stamped} pages carry a generated stamp, {future} in the future")


def check_learning_path(vault: Path):
    """A stage with reading-list entries must not still say SKELETON."""
    lp = vault / "Learning Path"
    if not lp.is_dir():
        return
    for p in sorted(lp.glob("*.md")):
        if p.name == "README.md":
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        populated = "_None yet._" not in text
        skeleton = "> SKELETON" in text
        if populated and skeleton:
            err(f"stage is populated but still marked SKELETON: {p.name}")
        if populated and "Populated with" not in text:
            err(f"stage is populated but has no `> Populated with N sources:` line: {p.name}")
        m = re.search(r"> Populated with (\d+) sources?:", text)
        if m:
            claimed = int(m.group(1))
            links = len(set(re.findall(r"\]\(\.\./Wiki/(?:summaries|papers)/[^)]+\)", text)))
            if links < claimed:
                warn(f"{p.name}: claims {claimed} sources but links {links} summary/paper "
                     f"pages. Legitimate when several sources were folded into one summary "
                     f"— confirm that is the reason, do not pad the list.")
    print("  ok  Learning Path markers checked")


def check_index(vault: Path):
    """Every wiki page should be reachable from the index."""
    idx = vault / "Wiki" / "index.md"
    if not idx.exists():
        err("Wiki/index.md is missing")
        return
    text = idx.read_text(encoding="utf-8", errors="ignore")
    orphans = 0
    for p in (vault / "Wiki").rglob("*.md"):
        if p.name in ("index.md", "log.md", "overview.md"):
            continue
        stem = urllib.parse.quote(p.stem)
        if p.stem not in text and stem not in text:
            orphans += 1
            err(f"page not listed in Wiki/index.md: {p.relative_to(vault)}")
    print(f"  ok  index coverage checked, {orphans} orphans")


def check_log(vault: Path):
    """log.md must be frontmatter-free with unique, newest-first date headings."""
    log = vault / "Wiki" / "log.md"
    if not log.exists():
        err("Wiki/log.md is missing")
        return
    fm, body = frontmatter(log)
    if fm is not None:
        err("Wiki/log.md must be frontmatter-free (OKF reserved file)")
    dates = re.findall(r"^## (\d{4}-\d{2}-\d{2})$", body or log.read_text(encoding="utf-8"), re.M)
    if dates != sorted(dates, reverse=True):
        err("Wiki/log.md date headings are not newest-first")
    if len(dates) != len(set(dates)):
        err("Wiki/log.md has duplicate date headings — merge them")
    print(f"  ok  log.md: {len(dates)} date headings, ordered and unique")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vault")
    ap.add_argument("--since", help="YYYY-MM-DD: require generated stamps on pages "
                                    "with an `updated` on or after this date")
    args = ap.parse_args()

    vault = Path(args.vault)
    if not vault.is_dir():
        sys.exit(f"not a directory: {vault}")
    repo = Path(__file__).resolve().parent.parent

    print(f"Checking {vault.name}")
    check_okf(vault, repo)
    check_links(vault)
    check_wiki_refs(vault)
    check_stamps(vault, args.since)
    check_learning_path(vault)
    check_index(vault)
    check_log(vault)

    print()
    for w in WARNINGS:
        print(f"WARN  {w}")
    for e in ERRORS:
        print(f"FAIL  {e}")
    if ERRORS:
        print(f"\n{len(ERRORS)} problem(s). Fix them before requesting a review.")
        sys.exit(1)
    print(f"\nClean{f' ({len(WARNINGS)} warning(s))' if WARNINGS else ''}. "
          f"Mechanical checks only — factual accuracy still needs the independent review.")


if __name__ == "__main__":
    main()
