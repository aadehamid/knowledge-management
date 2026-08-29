#!/usr/bin/env python3
"""OKF v0.2 conformance battery for an LLM-wiki-style bundle.

Checks (per the recurring-review postmortem of 2026-08-29):
  - every non-reserved .md has parseable YAML frontmatter with non-empty type
  - reserved index.md/log.md shapes (frontmatter-free; log date-grouped newest-first, unique dates)
  - no [[wikilinks]] outside Raw bodies and code spans
  - markdown links resolve FROM THE CONTAINING FILE (URL-decoded)
  - sources[].resource provenance paths resolve FROM THE CONTAINING FILE
  - Raw files use fetch_status, not status
  - wiki pages carry description

Usage: python3 test_okf_bundle.py "<bundle dir>"
Exit 0 = conformant.
"""
import os, re, sys, urllib.parse

try:
    import yaml
except ImportError:
    yaml = None

BUNDLE = sys.argv[1] if len(sys.argv) > 1 else None
if not BUNDLE or not os.path.isdir(BUNDLE):
    print("usage: test_okf_bundle.py <bundle-dir>"); sys.exit(2)

failures = []

def fm_of(path):
    text = open(path, encoding="utf-8").read()
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, text, lines
    try:
        end = lines.index("---", 1)
    except ValueError:
        return None, text, lines
    return "\n".join(lines[1:end]), "\n".join(lines[end+1:]), lines


def load_fm(raw_fm):
    """Parse frontmatter; PyYAML when available, minimal fallback otherwise.

    The fallback is intentionally conservative: it validates the structural
    things this test needs (non-empty type:, no lifecycle 'status:' in Raw,
    presence of description:) without attempting full YAML semantics.
    """
    if yaml is not None:
        try:
            return yaml.safe_load(raw_fm), None
        except yaml.YAMLError as e:
            return None, str(e)
    out = {}
    for ln in raw_fm.splitlines():
        m = re.match(r"^(\w[\w-]*):\s*(.*)$", ln)
        if m and not ln.startswith((" ", "\t")):
            out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out, None

def resolve_from(src_path, href):
    return os.path.normpath(os.path.join(os.path.dirname(src_path), urllib.parse.unquote(href)))

concepts = 0
for dp, dns, fns in os.walk(BUNDLE):
    for f in sorted(fns):
        if not f.endswith(".md"): continue
        p = os.path.join(dp, f)
        rel = os.path.relpath(p, BUNDLE)
        raw_fm, body, lines = fm_of(p)
        reserved = os.path.basename(f) in ("index.md", "log.md")

        if reserved:
            if raw_fm is not None:
                fm = yaml.safe_load(raw_fm) or {}
                allowed = {"okf_version"} if rel == "index.md" else set()
                extra = set(fm.keys()) - allowed
                if extra: failures.append(f"{rel}: reserved file has frontmatter keys {extra}")
            if os.path.basename(f) == "log.md":
                dates = re.findall(r"^## (\d{4}-\d{2}-\d{2})$", body, re.M)
                if dates != sorted(dates, reverse=True):
                    failures.append(f"{rel}: log not newest-first")
                if len(dates) != len(set(dates)):
                    failures.append(f"{rel}: duplicate date headings")
            continue

        concepts += 1
        if raw_fm is None:
            failures.append(f"{rel}: no frontmatter"); continue
        fm, err = load_fm(raw_fm)
        if err:
            failures.append(f"{rel}: YAML parse error: {err}"); continue
        if not fm or not fm.get("type"):
            failures.append(f"{rel}: missing type")

        # wikilinks outside code spans; Raw BODIES excluded — wikilink-shaped
        # text in fetched sources (cross-reference notes, [[.]] artifacts) is
        # immutable fetched content, not bundle links
        check_txt = open(p, encoding="utf-8").read()
        if rel.startswith("Raw/"):
            _, _, rl = fm_of(p)
            check_txt = "\n".join(rl[:rl.index("---", 1)])
        clean = re.sub(r"`[^`]*`", "", check_txt)
        wl = re.findall(r"\[\[[^\]]+\]\]", clean)
        if wl: failures.append(f"{rel}: wikilinks present: {wl[:2]}")

        # markdown links resolve from containing file
        # (Raw BODIES excluded: their md links are fetched-webpage artifacts)
        check_text = clean
        if rel.startswith("Raw/"):
            rf, _, _ = fm_of(p)          # frontmatter only
            check_text = rf or ""        # body md links are fetch artifacts, out of scope
        for m in re.finditer(r"\[[^\]]+\]\(([^)#]+?\.md)\)", check_text):
            href = m.group(1)
            if href.startswith(("http://", "https://")): continue
            if not os.path.exists(resolve_from(p, href)):
                failures.append(f"{rel}: broken md link {href}")

        # sources[].resource resolves from containing file
        if isinstance(fm.get("sources"), list):
            for s in fm["sources"]:
                if isinstance(s, dict) and s.get("resource"):
                    r = s["resource"]
                    if r.startswith(("http://", "https://")): continue
                    if not os.path.exists(resolve_from(p, r)):
                        failures.append(f"{rel}: broken provenance resource {r}")

        # description present on wiki-type pages
        if rel.startswith("Wiki/") and not fm.get("description"):
            failures.append(f"{rel}: missing description")

# Raw files: fetch_status not status; frontmatter links fine
raw_dir = os.path.join(BUNDLE, "Raw")
if os.path.isdir(raw_dir):
    for f in sorted(os.listdir(raw_dir)):
        if not f.endswith(".md"): continue
        p = os.path.join(raw_dir, f)
        raw_fm, body, _ = fm_of(p)
        if raw_fm is None: continue
        # Raw BODY markdown links are fetched-webpage artifacts (site-relative
        # hrefs like /zilliztech/...), out of scope per the conversion contract.
        # Only frontmatter-provided paths are checked here.
        if re.search(r"^status:", raw_fm, re.M):
            failures.append(f"Raw/{f}: uses lifecycle 'status' (must be fetch_status)")
        _, err = load_fm(raw_fm)
        if err:
            failures.append(f"Raw/{f}: YAML parse error: {err}")

if failures:
    print(f"FAIL: {len(failures)} issue(s)")
    for x in failures[:40]: print("  -", x)
    sys.exit(1)
print(f"OK: bundle conformant ({concepts} concepts checked)")
