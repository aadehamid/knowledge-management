#!/usr/bin/env python3
"""Diff an external saved-links source against a corpus inventory.

Answers: what is saved in an external tool that never entered this corpus?

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

# No project defaults live here. The reusable parts of this file are norm() and
# the alias table; where a project keeps its inventory is the project's business,
# so those paths are required arguments rather than baked-in guesses.

# Parameters that identify the resource and must be kept.
MEANINGFUL_QS = {"v", "id", "p", "list", "paper_id", "arxiv"}
# Common tracking parameters, always dropped.
TRACKING = re.compile(r"^(utm_|ref|si|s|feature|fbclid|gclid|mc_|source|usp)")


# Known aliases: the same resource reachable under a different host, path or
# owner. Verified case by case against both corpora — not guessed. Without
# these the diff reports dozens of false "only external" hits.
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


# Sites that moved domain. Each pair was confirmed present on both sides with
# the same article slug before being added here.
HOST_MOVES = {
    "mlops.systems": "alexstrick.com",          # same author, same post slugs
    "e2eml.school": "brandonrohrer.com",
    "buttondown.email": "buttondown.com",
    "camdavidsonpilon.github.io": "dataorigami.net",
    "eigenfoo.xyz": "georgeho.org",
    "v0.dev": "v0.app",
}


def apply_aliases(host: str, path: str) -> tuple[str, str]:
    host = HOST_MOVES.get(host, host)
    # apeatling moved dated permalinks to /articles/.
    if host == "apeatling.com":
        path = re.sub(r"^/\d{4}/\d{2}/\d{2}/", "/articles/", path).removesuffix(".html")
    # The OpenAI cookbook moved to developers.openai.com/cookbook/.
    if host == "cookbook.openai.com":
        host, path = "developers.openai.com", "/cookbook" + path
    # x.ai renamed /blog/ to /news/; jordivillar /data/ to /blog/.
    if host == "x.ai":
        path = path.replace("/blog/", "/news/")
    if host == "jordivillar.com":
        path = path.replace("/data/", "/blog/")
    # MIT OCW dropped the department segment from course paths.
    if host == "ocw.mit.edu":
        path = re.sub(r"^/courses/[a-z-]+/(\d)", r"/courses/\1", path)
    # Renamed repos / model orgs.
    path = re.sub(r"/videlalvaro/leet-llm", "/videlalvaro/inference-school", path, flags=re.I)
    path = re.sub(r"/ds4sd/smoldocling", "/docling-project/SmolDocling", path, flags=re.I)
    # The "Inside vLLM" article is published on both the author's site and vllm.ai.
    if (host, path) == ("vllm.ai", "/blog/2025-09-05-anatomy-of-vllm"):
        host, path = "aleksagordic.com", "/blog/vllm"
    # Unsloth moved docs.unsloth.ai/X to unsloth.ai/docs/X AND restructured the
    # tree underneath. Page slugs stayed unique, so key on the final slug.
    if host in ("docs.unsloth.ai", "unsloth.ai") and (host == "docs.unsloth.ai"
                                                      or path.startswith("/docs")):
        slug = path.rstrip("/").split("/")[-1]
        slug = UNSLOTH_SLUGS.get(slug, slug)
        return "unsloth.ai", "/docs/" + slug

    # Blog subdomain vs /blog/ path — same article, two URL shapes.
    for sub, base in (("blog.lancedb.com", "lancedb.com"),
                      ("blog.llamaindex.ai", "llamaindex.ai"),
                      ("blog.vllm.ai", "vllm.ai")):
        if host == sub:
            host, path = base, "/blog" + path
    if host == "vllm.ai" and path.startswith("/blog/"):
        # blog.vllm.ai/2025/11/19/slug.html vs vllm.ai/blog/2025-11-19-slug
        path = re.sub(r"^/blog/(\d{4})/(\d{2})/(\d{2})/([^/]+?)(\.html)?$",
                      r"/blog/\1-\2-\3-\4", path)
    # Stability renamed /news/ to /news-updates/.
    if host == "stability.ai":
        path = path.replace("/news-updates/", "/news/")
    # Slack engineering dropped the Medium hash suffix from its slugs.
    if host == "slack.engineering":
        path = re.sub(r"-[0-9a-f]{12}$", "", path)
    # O'Reilly moved the reader to learning.oreilly.com.
    if host == "learning.oreilly.com":
        host = "oreilly.com"

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
        for seg in ("/embed/", "/live/", "/shorts/", "/v/"):
            if not vid and seg in path:
                vid = path.split(seg)[-1].split("/")[0]
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
    """Pull URLs out of whatever the external tool exported: txt, csv, json, markdown."""
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
    """key -> record, from the corpus inventory TSV."""
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
    ap.add_argument("--external", "--recall", dest="external", required=True,
                    help="file of URLs from the external source (txt/csv/json/md), or - for stdin")
    ap.add_argument("--inventory", required=True,
                    help="TSV of what the corpus already holds; needs a 'url' column")
    ap.add_argument("--routing", default=None,
                    help="optional TSV mapping url -> subject, for the overlap breakdown")
    ap.add_argument("--sources", default=None,
                    help="directory of <subject>/urls.txt files already queued in the pipeline")
    ap.add_argument("--out", help="write the external-only URLs here, one per line")
    args = ap.parse_args()

    dt = load_inventory(Path(args.inventory))
    # The corpus is whatever the pipeline already knows about: the
    # inventory PLUS every <sources>/*/urls.txt when --sources is given. Comparing
    # against the inventory alone reports sources as missing when already queued.
    for uf in sorted(Path(args.sources).glob("*/urls.txt")) if args.sources else []:
        subject = uf.parent.name
        for line in uf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k = norm(line.split("|")[0])
            if k:
                dt.setdefault(k, {"url": line.split("|")[0].strip(),
                                  "title": "", "_from": f"urls.txt:{subject}"})
    subjects = load_subjects(Path(args.routing)) if args.routing else {}
    external_urls = urls_from(args.external)
    external = {}
    for u in external_urls:
        k = norm(u)
        if k:
            external.setdefault(k, u)

    only_external = sorted(set(external) - set(dt))
    only_dt = sorted(set(dt) - set(external))
    both = sorted(set(dt) & set(external))

    print(f"external URLs read : {len(external_urls)} ({len(external)} unique resources)")
    print(f"corpus resources   : {len(dt)} unique resources")
    print()
    print(f"  in BOTH           : {len(both)}")
    print(f"  external only     : {len(only_external)}   <- not in the corpus")
    print(f"  corpus only       : {len(only_dt)}")
    print()

    if only_external:
        by_host = defaultdict(list)
        for k in only_external:
            by_host[urllib.parse.urlparse(external[k]).netloc.lower().removeprefix("www.")
                    or "?"].append(external[k])
        print("=" * 72)
        print("EXTERNAL ONLY — candidates to add to the pipeline")
        print("=" * 72)
        for host, urls in sorted(by_host.items(), key=lambda kv: -len(kv[1])):
            print(f"\n  {host}  ({len(urls)})")
            for u in sorted(urls):
                print(f"    {u}")

    if both and subjects:
        print()
        print("=" * 72)
        print("OVERLAP by the subject the corpus routed it to")
        print("=" * 72)
        counts = defaultdict(int)
        for k in both:
            counts[subjects.get(k, "(unrouted)")] += 1
        for s, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {s:<32}{n:>5}")

    if args.out:
        Path(args.out).write_text("\n".join(external[k] for k in only_external) + "\n",
                                  encoding="utf-8")
        print(f"\nExternal-only URLs written to {args.out}")
        print("These are ready to append to a resources/sources/<subject>/urls.txt "
              "after routing.")


if __name__ == "__main__":
    main()
