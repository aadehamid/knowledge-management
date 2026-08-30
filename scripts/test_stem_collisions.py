#!/usr/bin/env python3
"""Regression tests for filename-collision handling on both pipeline surfaces.

Two distinct sources can produce the same filename, and before these guards the
second silently overwrote the first — taking its frontmatter url and wiki_refs
with it. Two independent surfaces need protecting:

  repo   references/papers/<subject>/<stem>.md   -- convert_pdfs.assign_stem
  vault  Raw/<source_type>-<title-slug>.md       -- sync_to_vault.disambiguate

Run: python3 scripts/test_stem_collisions.py
"""
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

sys.modules.setdefault("requests", types.ModuleType("requests"))
ROOT = Path(__file__).resolve().parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cp = _load("cp", "convert_pdfs.py")
sv = _load("sv", "sync_to_vault.py")

failures = []


def check(label, got, want):
    if got == want:
        print(f"OK   {label}")
    else:
        failures.append(label)
        print(f"FAIL {label}\n       got:  {got!r}\n       want: {want!r}")


# --- repo surface -----------------------------------------------------------

# Same path stem, different sources: first keeps it, second is disambiguated.
claimed = {}
a = cp.assign_stem("https://docs.ell.so/introduction", "ell docs", "", claimed)
claimed[a] = "https://docs.ell.so/introduction"
b = cp.assign_stem("https://huggingface.co/learn/llm-course/chapter3/1", "HF course", "", claimed)
check("distinct sources sharing '/introduction' get distinct stems", a != b, True)
check("first claimant keeps the bare stem", a, "introduction")

# A source re-run keeps the stem it already owns (no churn on disk).
claimed2 = {"lora": "https://thinkingmachines.ai/blog/lora/"}
same = cp.assign_stem("https://thinkingmachines.ai/blog/lora/", "LoRA Without Regret", "", claimed2)
check("a source keeps the stem it already owns", same, "lora")

# ...but a different source wanting that stem is moved aside.
other = cp.assign_stem("https://github.com/ml-explore/mlx-examples/tree/main/lora", "mlx lora", "", claimed2)
check("different source does not steal an owned stem", other != "lora", True)

# Path.stem truncates dotted names: llama3.3 -> 'llama3', colliding with llama3.
claimed3 = {}
l3 = cp.assign_stem("https://ollama.com/library/llama3", "llama3", "", claimed3)
claimed3[l3] = "https://ollama.com/library/llama3"
l33 = cp.assign_stem("https://ollama.com/library/llama3.3", "llama3.3", "", claimed3)
check("llama3 and llama3.3 do not collide", l3 != l33, True)

# arXiv ids are dotted too: /abs/2401.04088 -> '2401' for every 2401 paper.
claimed4 = {}
p1 = cp.assign_stem("https://arxiv.org/abs/2401.04088", "Mixtral of Experts", "", claimed4)
claimed4[p1] = "https://arxiv.org/abs/2401.04088"
p2 = cp.assign_stem("https://arxiv.org/abs/2401.08406", "RAG vs Fine-tuning", "", claimed4)
check("two arXiv 2401.* papers do not collide", p1 != p2, True)

# Determinism: same inputs, same answer.
check(
    "assignment is deterministic",
    cp.assign_stem("https://ollama.com/library/llama3.3", "llama3.3", "", dict(claimed3)),
    l33,
)

# --- vault surface ----------------------------------------------------------

with tempfile.TemporaryDirectory() as td:
    vault = Path(td)
    incumbent = vault / "doc-finetuning-large-language-models.md"
    incumbent.write_text(
        '---\nurl: "https://www.deeplearning.ai/short-courses/finetuning-large-language-models/"\n'
        'title: "Finetuning Large Language Models"\n---\n\nbody\n',
        encoding="utf-8",
    )
    check(
        "frontmatter_url reads the incumbent's url",
        sv.frontmatter_url(incumbent),
        "https://www.deeplearning.ai/short-courses/finetuning-large-language-models/",
    )
    newcomer_url = "https://magazine.sebastianraschka.com/p/finetuning-large-language-models"
    dest = sv.disambiguate(vault, incumbent, newcomer_url)
    check("colliding vault name is redirected", dest != incumbent, True)
    check("incumbent file is untouched", incumbent.exists(), True)
    check("redirected name carries the host", "magazine" in dest.name, True)

    # A second collision on the same name still resolves.
    dest.write_text('---\nurl: "%s"\n---\n' % newcomer_url, encoding="utf-8")
    third = sv.disambiguate(vault, incumbent, "https://magazine.sebastianraschka.com/p/other")
    check("a third collision resolves to yet another name", third not in (incumbent, dest), True)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    sys.exit(1)
print("All collision tests pass")
