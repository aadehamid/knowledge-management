#!/usr/bin/env python3
"""Diff a Recall knowledge base against the DEVONthink corpus inventory.

Answers: what is saved in Recall that never entered the DEVONthink corpus?

    # from a file of URLs (one per line, or CSV/JSON containing URLs)
    python3 scripts/compare_recall.py --recall recall_export.json

    # or paste URLs on stdin
    pbpaste | python3 scripts/compare_recall.py --recall -

URL matching is deliberate, not naive. Stripping query strings collapses every
YouTube link to "youtube.com/watch", which in this project once produced a
5-overlap answer where the truth was 35. Video ids are preserved; tracking
parameters are dropped.
"""
import argparse
import csv
import json
import re
import sys
import urllib.parse
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_INVENTORY = REPO / "projects/llm-corpus-expansion/data/devonthink-inventory.tsv"
DEFAULT_ROUTING = REPO / "projects/llm-corpus-expansion/data/routing.tsv"

# Parameters that identify the resource and must be kept.
MEANINGFUL_QS = {"v", "id", "p", "list", "paper_id", "arxiv"}
# Common tracking parameters, always dropped.
TRACKING = re.compile(r"^(utm_|ref|si|s|feature|fbclid|gclid|mc_|source|usp)")


# Known aliases: the same resource reachable under a different host, path or
# owner. Verified case by case against both corpora — not guessed. Without
# these the diff reports dozens of false "only in Recall" hits.
REPO_RENAMES = {
    "openaccess-ai-collective/axolotl": "axolotl-ai-cloud/axolotl",
    "facebookresearch/llama-recipes": "meta-llama/llama-cookbook",
    "mozilla-ocho/llamafile": "mozilla-ai/llamafile",
    "neuralmagic/guidellm": "vllm-project/guidellm",
}


# Unsloth doc pages that were renamed, not just moved. Verified by comparing
# the slug lists on both sides.
UNSLOTH_SLUGS = {
    "reinforcement-learning-guide": "reinforcement-learning-rl-guide",
    "how-to-finetune-llama-3-and-export-to-ollama":
        "tutorial-how-to-finetune-llama-3-and-use-in-ollama",
    "qwen3-vl-run-and-fine-tune": "qwen3-vl-how-to-run-and-fine-tune",
    "tutorials-how-to-fine-tune-and-run-llms": "tutorials",
}


def apply_aliases(host: str, path: str) -> tuple[str, str]:
    # Unsloth moved docs.unsloth.ai/X to unsloth.ai/docs/X AND restructured the
    # tree underneath. Page slugs stayed unique, so key on the final slug.
    if host in ("docs.unsloth.ai", "unsloth.ai") and (host == "docs.unsloth.ai"
                                                      or path.startswith("/docs")):
        slug = path.rstrip("/").split("/")[-1]
        slug = UNSLOTH_SLUGS.get(slug, slug)
        return "unsloth.ai", "/docs/" + slug

    # Lightning renamed /studios/ to /templates/ and /pages/courses/ to /courses/.
    if host == "lightning.ai":
        path = path.replace("/studios/", "/templates/").replace("/pages/courses/", "/courses/")
    # DeepLearning.AI renamed /short-courses/ to /courses/, and learn.* is the
    # course player for the same course as the www landing page.
    if host in ("deeplearning.ai", "learn.deeplearning.ai"):
        path = path.replace("/short-courses/", "/courses/")
        path = re.sub(r"(/courses/[^/]+)/lesson/.*$", r"\1", path)
        return "deeplearning.ai", path
    # GitHub repo renames.
    if host == "github.com":
        m = re.match(r"^/([^/]+/[^/]+)(/.*)?$", path)
        if m and m.group(1).lower() in REPO_RENAMES:
            path = "/" + REPO_RENAMES[m.group(1).lower()] + (m.group(2) or "")
    return host, path


def norm(url: str) -> str:
    """Canonical key for one URL. Two URLs sharing a key are the same resource."""
    url = (url or "").strip()
    if not url:
        return ""
    if "://" not in url:
        url = "https://" + url
    p = urllib.parse.urlparse(url)
    host = p.netloc.lower().split("@")[-1].split(":")[0]
    for prefix in ("www.", "m.", "mobile."):
        host = host.removeprefix(prefix)
    path = p.path.rstrip("/")

    # YouTube: the video id IS the identity; the path never is.
    if host in ("youtube.com", "youtu.be", "youtube-nocookie.com"):
        q = urllib.parse.parse_qs(p.query)
        vid = q.get("v", [None])[0]
        if not vid and host == "youtu.be":
            vid = path.lstrip("/")
        if not vid and "/embed/" in path:
            vid = path.split("/embed/")[-1]
        if not vid and (lst := q.get("list", [None])[0]):
            return f"youtube:playlist:{lst}"
        return f"youtube:{vid}" if vid else "youtube:" + path

    # arXiv: /abs/ID and /pdf/ID are the same paper.
    if host == "arxiv.org":
        m = re.search(r"/(?:abs|pdf)/([0-9v.]+)", path)
        if m:
            return f"arxiv:{m.group(1).rstrip('.')}"

    # GitHub: strip the trailing view fragments that do not change the resource.
    if host == "github.com":
        path = re.sub(r"/(tree|blob)/(main|master)/?$", "", path)

    host, path = apply_aliases(host, path)

    keep = {k: v for k, v in urllib.parse.parse_qs(p.query).items()
            if k in MEANINGFUL_QS and not TRACKING.match(k)}
    qs = "?" + urllib.parse.urlencode(sorted(keep.items()), doseq=True) if keep else ""
    return f"{host}{path}{qs}".lower()


def urls_from(path: str) -> list[str]:
    """Pull URLs out of whatever Recall exported: txt, csv, json, or markdown."""
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8",
                                                                    errors="ignore")
    found: list[str] = []
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            def walk(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if isinstance(v, str) and v.startswith("http") and \
                           k.lower() in ("url", "link", "source", "source_url", "href", "uri"):
                            found.append(v)
                        else:
                            walk(v)
                elif isinstance(o, list):
                    for i in o:
                        walk(i)
            walk(json.loads(text))
        except json.JSONDecodeError:
            pass
    if not found:
        found = re.findall(r"https?://[^\s\"'<>)\]},]+", text)
    seen, out = set(), []
    for u in found:
        u = u.rstrip(".,;)")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def load_inventory(path: Path) -> dict[str, dict]:
    """key -> record, from the DEVONthink inventory TSV."""
    recs: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            k = norm(row.get("url", ""))
            if k:
                recs.setdefault(k, row)
    return recs


def load_subjects(path: Path) -> dict[str, str]:
    """key -> subject it was routed to, when the routing table is available."""
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            k = norm(row.get("url", ""))
            if k:
                out[k] = row.get("subject", "")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recall", required=True,
                    help="file of Recall URLs (txt/csv/json/md), or - for stdin")
    ap.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    ap.add_argument("--routing", default=str(DEFAULT_ROUTING))
    ap.add_argument("--out", help="write the Recall-only URLs here, one per line")
    args = ap.parse_args()

    dt = load_inventory(Path(args.inventory))
    subjects = load_subjects(Path(args.routing))
    recall_urls = urls_from(args.recall)
    recall = {}
    for u in recall_urls:
        k = norm(u)
        if k:
            recall.setdefault(k, u)

    only_recall = sorted(set(recall) - set(dt))
    only_dt = sorted(set(dt) - set(recall))
    both = sorted(set(dt) & set(recall))

    print(f"Recall URLs read      : {len(recall_urls)} ({len(recall)} unique resources)")
    print(f"DEVONthink records    : {len(dt)} unique resources")
    print()
    print(f"  in BOTH             : {len(both)}")
    print(f"  in RECALL only      : {len(only_recall)}   <- not captured in DEVONthink")
    print(f"  in DEVONthink only  : {len(only_dt)}")
    print()

    if only_recall:
        by_host = defaultdict(list)
        for k in only_recall:
            by_host[urllib.parse.urlparse(recall[k]).netloc.lower().removeprefix("www.")
                    or "?"].append(recall[k])
        print("=" * 72)
        print("IN RECALL ONLY — candidates to add to the pipeline")
        print("=" * 72)
        for host, urls in sorted(by_host.items(), key=lambda kv: -len(kv[1])):
            print(f"\n  {host}  ({len(urls)})")
            for u in sorted(urls):
                print(f"    {u}")

    if both and subjects:
        print()
        print("=" * 72)
        print("OVERLAP by the subject DEVONthink routed it to")
        print("=" * 72)
        counts = defaultdict(int)
        for k in both:
            counts[subjects.get(k, "(unrouted)")] += 1
        for s, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {s:<32}{n:>5}")

    if args.out:
        Path(args.out).write_text("\n".join(recall[k] for k in only_recall) + "\n",
                                  encoding="utf-8")
        print(f"\nRecall-only URLs written to {args.out}")
        print("These are ready to append to a resources/sources/<subject>/urls.txt "
              "after routing.")


if __name__ == "__main__":
    main()
