#!/usr/bin/env python3
"""Regression tests for sync_to_vault.py frontmatter generation.

Run: python3 scripts/test_sync_frontmatter.py
Exists because raw f-string YAML interpolation produced invalid frontmatter
for titles containing ': ' (found by the 2026-08-29 sol-5.6 recurring review
— 18 Raw files had unparseable frontmatter).
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sync_to_vault as stv

try:
    import yaml
except ImportError:
    print("PyYAML not installed — install with: pip install pyyaml")
    sys.exit(2)

CASES = [
    "A Deep Dive into NVIDIA Rubin CPX: Architecture, Splitwise/DistServe, Inference Economics",
    "H2O: Heavy-Hitter Oracle for Efficient Generative Inference",
    "GPTCache: Semantic cache for LLMs",
    "Let's build GPT: from scratch, in code, spelled out",
    'He said "hello" loudly',
    "Plain Title No Punctuation",
    "",
    "Trailing space ",
    "true",
    "#starts-with-hash",
    "colon: inside: title: multiple",
]

failures = 0
for title in CASES:
    fm = stv.build_frontmatter(
        {"url": "https://example.com/x", "title": title, "source_type": "blog",
         "fetched_at": "2026-08-29", "author": "Author: with colon"}
    )
    inner = fm.splitlines()[1:-1]
    try:
        parsed = yaml.safe_load("\n".join(inner))
        if parsed.get("title") != title:
            print(f"FAIL round-trip mismatch: {title!r} -> {parsed.get('title')!r}")
            failures += 1
        else:
            print(f"OK   {title[:60]!r}")
    except yaml.YAMLError as e:
        print(f"FAIL parse: {title!r} -> {e}")
        failures += 1

if failures:
    print(f"\n{failures} failure(s)")
    sys.exit(1)
print("\nAll frontmatter tests pass")
