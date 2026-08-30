#!/usr/bin/env python3
"""Bootstrap a new subject-area vault to the OKF v0.2 bundle shape.

The vault CLAUDE.md calls bootstrap a mandatory one-time step: without the
scaffolding, the first ingest produces orphan summaries with nothing to link
into. This script creates that scaffolding.

Two deliberate departures from a straight clone of an existing vault:

  * Stub pages are NOT copied from a template term list. Each vault declares
    the entities/concepts its own Raw corpus actually discusses, chosen from
    term frequencies over that corpus. A generic list produces stubs nobody
    fills - the LLM Inference vault still carries an unfilled "HBM vs SRAM"
    bootstrap stub from May.
  * A vault may be CATALOG-shaped (no concept layer, no curriculum). Some
    subjects are archives of dated material rather than concept domains, and
    a wiki over them goes stale faster than it gets written.

Process sections of CLAUDE.md (workflows, QA, independent review, OKF
conventions) are inherited verbatim from the reference vault; only the
domain-specific sections are rewritten.

Usage: python3 scripts/bootstrap_vault.py [--dry-run]
"""
import re
import shutil
import sys
from datetime import date
from pathlib import Path
from urllib.parse import quote

KM = Path("/Users/hamidadesokan/Documents/Knowledge Management")
REFERENCE = KM / "LLM Inference Optimization"
TODAY = date.today().isoformat()

# Stage TITLES are per-vault, not inherited. The reference vault's names
# ("First Measurements", "Hardware Track") describe inference work and mean
# nothing in a fine-tuning or foundations curriculum; the repo README is
# explicit that a schema clone must strip stage scopes belonging to a sibling
# subject. Each vault below declares its own seven stages as (title, scope).

VAULTS = {
    "LLM Fine-tuning": {
        "subject": "llm-finetuning",
        "domain": "LLM fine-tuning and post-training",
        "blurb": "adapting pretrained language models to specific tasks and preferences — "
                 "parameter-efficient methods (LoRA, QLoRA, adapters), supervised fine-tuning, "
                 "preference optimization (RLHF, DPO, GRPO), dataset construction, and the "
                 "distributed-training machinery that makes it affordable",
        "entities": {
            "LoRA": "low-rank adaptation: train two small matrices instead of full weights",
            "QLoRA": "LoRA on a quantized base model, cutting memory further",
            "PEFT": "Hugging Face library implementing parameter-efficient fine-tuning methods",
            "Unsloth": "kernel-optimized fine-tuning stack, notable for speed and memory claims",
            "Axolotl": "config-driven fine-tuning framework wrapping the common recipes",
            "FSDP": "PyTorch Fully Sharded Data Parallel: shards params, grads and optimizer state",
        },
        "concepts": {
            "Supervised Fine-Tuning": "training on labelled instruction/response pairs (SFT)",
            "RLHF": "reinforcement learning from human feedback: reward model plus policy optimization",
            "DPO": "direct preference optimization: preference learning without a separate reward model",
            "GRPO": "group relative policy optimization, the RL method behind recent reasoning models",
            "Distillation": "training a smaller student model on a larger teacher's behaviour",
            "Synthetic Data": "model-generated training data, and the curation it demands",
        },
        "stages": [
            ("1 - Why Fine-tune", "when to fine-tune at all — prompting vs RAG vs tuning, and what each can and cannot fix"),
            ("2 - Data and Datasets", "dataset construction, formatting, curation, and how much data you actually need"),
            ("3 - Parameter-Efficient Methods", "LoRA, QLoRA, adapters, and what rank actually buys you"),
            ("4 - A Full SFT Run", "supervised fine-tuning end to end — hyperparameters, and the common failure modes"),
            ("5a - Preference and RL Track", "reward models, RLHF, DPO, GRPO — learning from preferences"),
            ("5b - Scaling and Systems Track", "FSDP, quantized training, multi-GPU, and the memory arithmetic"),
            ("6 - Frontier and Open Problems", "reasoning-model post-training, on-policy distillation, open problems"),
        ],
    },
    "ML Foundations": {
        "subject": "ml-foundations",
        "domain": "machine learning foundations",
        "blurb": "the mathematics and mechanics under every model — backpropagation and the chain "
                 "rule, gradient descent, automatic differentiation, training dynamics, and the "
                 "probability and linear algebra they rest on",
        "entities": {
            "PyTorch": "the array/autograd framework most of this corpus builds on",
            "Micrograd": "Karpathy's minimal scalar autograd engine, used to teach backprop",
        },
        "concepts": {
            "Backpropagation": "reverse-mode differentiation applied to a network's loss",
            "Gradient Descent": "iterative parameter update along the negative gradient",
            "Chain Rule": "the composition rule that makes backpropagation possible",
            "Automatic Differentiation": "computing exact derivatives by tracking a computational graph",
            "Jacobian": "the matrix of partial derivatives of a vector-valued function",
            "Learning Rate": "the step-size hyperparameter and its schedules",
            "Regularization": "constraints that trade training fit for generalization",
        },
        "stages": [
            ("1 - Intuition Without Calculus", "what learning from data means, before any derivatives"),
            ("2 - The Calculus of Learning", "derivatives, the chain rule, computational graphs"),
            ("3 - Backpropagation from Scratch", "deriving and implementing it by hand, then in code"),
            ("4 - Training Dynamics", "learning rates, initialization, normalization, why runs diverge"),
            ("5a - Generalization Track", "overfitting, regularization, and evaluating honestly"),
            ("5b - Probability and Statistics Track", "the inferential half of the foundations"),
            ("6 - Frontier and Open Problems", "where the simple story about optimization stops being true"),
        ],
    },
    "RAG and Retrieval": {
        "subject": "rag-retrieval",
        "domain": "document AI and retrieval",
        "blurb": "getting documents into a form models can use and finding the right passage — "
                 "OCR and document parsing, embeddings, rerankers, late-interaction retrieval, "
                 "and the vector infrastructure underneath",
        "entities": {
            "ColBERT": "late-interaction retrieval model scoring token-level similarity",
            "ColPali": "vision-language late-interaction retrieval directly over page images",
            "Qdrant": "vector database used for embedding storage and search",
            "olmOCR": "open OCR model for converting scanned documents to text",
        },
        "concepts": {
            "OCR": "optical character recognition — the dominant concern in this corpus",
            "Document Parsing": "turning PDFs and office files into structured, chunkable text",
            "Embeddings": "dense vector representations that make semantic search possible",
            "Reranking": "second-stage scoring that reorders a cheap first-stage candidate set",
            "Late Interaction": "deferring token-level matching to query time, as in ColBERT",
            "Chunking": "splitting documents so retrieval units carry coherent meaning",
        },
        "stages": [
            ("1 - Why Retrieval", "what grounding buys you, and where it fails"),
            ("2 - Documents In", "OCR, parsing, and the messy reality of real PDFs"),
            ("3 - Embeddings", "what they encode, how they are trained, how to choose one"),
            ("4 - Ranking and Reranking", "first-stage retrieval, rerankers, late interaction"),
            ("5a - Systems Track", "vector stores, hybrid search, indexing and scale"),
            ("5b - Evaluation Track", "measuring retrieval quality rather than trusting vibes"),
            ("6 - Frontier and Open Problems", "visual document retrieval, unified document models"),
        ],
    },
    "AI Engineering": {
        "subject": "ai-engineering",
        "domain": "AI engineering",
        "blurb": "building and operating systems on top of language models — agents and tool use, "
                 "the Model Context Protocol, prompt and context engineering, evaluation, and the "
                 "production concerns that decide whether any of it survives contact with users",
        "entities": {
            "MCP": "Model Context Protocol: a standard interface between models and tools",
            "DSPy": "framework for programming, rather than prompting, language models",
        },
        "concepts": {
            "Agent": "a model given tools and a loop in which to use them",
            "Tool Calling": "structured invocation of external functions by a model",
            "Evaluation": "measuring model and system quality — the discipline, not the vibes",
            "Prompt Engineering": "shaping model behaviour through instruction design",
            "Context Engineering": "deciding what enters the window, and what earns its tokens",
        },
        "stages": [
            ("1 - Building Blocks", "what an LLM application is actually made of"),
            ("2 - Prompting and Context", "instruction design, context windows, what earns its tokens"),
            ("3 - Tools and Agents", "function calling, MCP, agent loops and their failure modes"),
            ("4 - Evaluation", "building evals that catch regressions before users do"),
            ("5a - Production Track", "observability, cost, latency, guardrails"),
            ("5b - Orchestration Track", "multi-step workflows, retries, and where determinism belongs"),
            ("6 - Frontier and Open Problems", "self-improving harnesses, long-horizon agents"),
        ],
    },
    "LLM Landscape": {
        "subject": "llm-landscape",
        "domain": "the LLM landscape",
        "blurb": "model releases, providers, pricing and industry commentary — a dated record of "
                 "what shipped and when",
        "catalog_only": True,
        "entities": {}, "concepts": {}, "stages": [],
    },
}


def fm(**kv):
    lines = ["---"]
    for k, v in kv.items():
        if isinstance(v, list):
            lines.append(f"{k}: [{', '.join(v)}]")
        elif v == "":
            lines.append(f"{k}: []")
        else:
            lines.append(f"{k}: {v}" if not re.search(r"[:#]", str(v)) else f'{k}: "{v}"')
    lines.append("---")
    return "\n".join(lines)


def write(path: Path, text: str, dry: bool):
    if dry:
        print(f"    would write {path.relative_to(KM)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip("\n") + "\n", encoding="utf-8")


def build_claude_md(name: str, spec: dict) -> str:
    """Adapt the reference schema: rewrite the domain sections, inherit the process ones."""
    src = (REFERENCE / "CLAUDE.md").read_text(encoding="utf-8")
    out = src

    # Domain-specific header
    out = out.replace(
        "title: LLM Inference Optimization — Wiki Schema & Agent Workflows",
        f"title: {name} — Wiki Schema & Agent Workflows")
    out = out.replace("# LLM Inference Optimization — Wiki Schema", f"# {name} — Wiki Schema")
    out = re.sub(
        r"A personal knowledge base for developing intuitive \+ rigorous understanding of LLM inference.*?\n",
        f"A personal knowledge base for developing intuitive + rigorous understanding of "
        f"{spec['domain']}: {spec['blurb']}.\n", out, count=1, flags=re.S)

    if spec.get("catalog_only"):
        # Replace the curriculum layout with an explicit statement of the deviation.
        out = re.sub(r"## Learning Path directory layout.*?(?=\n## File conventions)",
                     CATALOG_NOTE, out, count=1, flags=re.S)
        out = re.sub(r"### Bootstrap workflow.*?(?=\n## QA)", CATALOG_BOOTSTRAP, out, count=1, flags=re.S)
    else:
        stage_block = "## Learning Path directory layout\n\n```\nLearning Path/\n├── README.md             # rationale, map, how to use\n"
        stages = spec["stages"]
        for i, (st, scope) in enumerate(stages):
            joint = "└──" if i == len(stages) - 1 else "├──"
            stage_block += f"{joint} {st}.md".ljust(46) + f"# {scope}\n"
        stage_block += ("```\n\nThe Learning Path is a **curriculum layer** over the wiki. It curates entity, "
                        "concept, and summary pages into a beginner-to-advanced progression with explicit "
                        "prerequisites, goals, and concrete activities at each stage. Each stage references "
                        "wiki pages by relative markdown link rather than restating their content — the wiki "
                        "is the textbook, the path is the table of contents in study order.\n\nThe path "
                        "branches at Stage 5 into two parallel tracks reflecting the two directions a "
                        "practitioner typically specializes in.\n")
        out = re.sub(r"## Learning Path directory layout.*?(?=\n## File conventions)",
                     stage_block, out, count=1, flags=re.S)
        ents = ", ".join(spec["entities"]) or "(none)"
        cons = ", ".join(spec["concepts"]) or "(none)"
        boot = (f"### Bootstrap workflow (one-time, **already performed for this vault on {TODAY}**)\n\n"
                f"Recorded for reference. The scaffolding below exists; do not recreate it.\n\n"
                f"1. `Wiki/index.md`, `Wiki/log.md`, `Wiki/overview.md`.\n"
                f"2. Stub entity pages: {ents}.\n"
                f"3. Stub concept pages: {cons}.\n"
                f"4. The seven `Learning Path/` stage files plus `README.md`, as empty skeletons.\n"
                f"5. A `bootstrap` entry in `Wiki/log.md`.\n\n"
                f"The stub list was derived from term frequencies over this vault's own `Raw/` corpus, "
                f"not cloned from a sibling vault. Keep it that way when adding stubs: a stub nobody "
                f"fills is worse than a missing page, because it looks like coverage.\n\n")
        out = re.sub(r"### Bootstrap workflow.*?(?=\n## QA)", boot, out, count=1, flags=re.S)

    out = out.replace("LLM Inference Optimization", name)
    out = out.replace("LLM inference optimization", spec["domain"])
    note = ("\n<!-- Process sections (Workflows, QA, Independent agent review, OKF format\n"
            "     conventions) are inherited verbatim from the reference vault, which is the\n"
            f"     canonical implementation of this schema. Domain sections were written for {name}.\n"
            "     Reserved OKF files (index.md, log.md) must stay frontmatter-free; this file\n"
            "     carries type: Process, so its frontmatter must remain the first line. -->\n")
    # insert after the closing fence of the frontmatter, never before it
    end = out.index("---", out.index("---") + 3) + 3
    return out[:end] + note + out[end:]


CATALOG_NOTE = """## No curriculum layer (deliberate)

This bundle is **catalog-shaped**. Its sources are model releases, provider
announcements, pricing pages and industry commentary — dated records of what
shipped and when, not a body of durable concepts. A concept wiki over them
would be stale faster than it could be written, so this vault has:

- no `Learning Path/` curriculum
- no bootstrap stub entity/concept pages

What it does have is `Wiki/overview.md` as a running synthesis of where the
landscape stands, and summaries for sources worth more than their metadata.
If a genuinely durable concept emerges here, create the page — but prefer
filing it in the vault that owns the concept.

"""

CATALOG_BOOTSTRAP = """### Bootstrap workflow (catalog bundle)

Performed once: `Wiki/index.md`, `Wiki/log.md`, `Wiki/overview.md` and a log
entry. No stub entity/concept pages and no `Learning Path/` — see "No
curriculum layer" above for why.

"""


def bootstrap(name: str, spec: dict, dry: bool):
    v = KM / name
    raw_n = len(list((v / "Raw").glob("*.md"))) if (v / "Raw").exists() else 0
    catalog = spec.get("catalog_only", False)
    print(f"\n  {name}  ({raw_n} Raw files){'  [catalog]' if catalog else ''}")

    write(v / "CLAUDE.md", build_claude_md(name, spec), dry)

    # root index.md
    start = [f"- [Wiki index](Wiki/index.md) - catalog of every wiki page by type",
             f"- [Overview](Wiki/overview.md) - top-level synthesis of {spec['domain']}"]
    layout = ["- [Wiki/](Wiki/index.md) - entities, concepts, papers, summaries (the knowledge)"]
    if not catalog:
        start.append("- [Learning Path](Learning%20Path/README.md) - beginner-to-advanced curriculum over the wiki")
        layout.append("- [Learning Path/](Learning%20Path/README.md) - the curriculum layer (the map)")
    layout += ["- [Raw/](Raw/) - immutable sources (the evidence)",
               "- CLAUDE.md - schema and agent workflows (process layer)"]
    write(v / "index.md",
          fm(okf_version='"0.2"') + f"\n\n# {name} — Knowledge Bundle\n\n"
          f"An Open Knowledge Format bundle: {spec['blurb']}, compiled from {raw_n} immutable "
          f"sources into an agent-maintained wiki"
          f"{'' if catalog else ' with a curriculum layer'}.\n\n"
          "# Start Here\n\n" + "\n".join(start) + "\n\n# Layout\n\n" + "\n".join(layout) + "\n\n"
          "# Conventions\n\n"
          "- Wiki pages carry frontmatter: `type`, `title`, `description`, `tags`,\n"
          "  `sources` (OKF provenance objects pointing into Raw/), `updated`.\n"
          "- Raw sources carry `type: Source` plus `fetch_status` (ingested | stub |\n"
          "  failed) and `wiki_refs` (a producer extension listing citing pages).\n"
          "- index.md and log.md are frontmatter-free OKF reserved files.\n", dry)

    # Wiki/overview.md
    write(v / "Wiki" / "overview.md",
          fm(type="overview", title=name, tags=[spec["subject"]], sources="", updated=TODAY,
             description=f"top-level synthesis of {spec['domain']}") +
          f"\n\n# {name} — Overview\n\n> STUB. Written on first ingest.\n\n"
          f"This bundle covers {spec['blurb']}.\n\n"
          f"It currently holds {raw_n} sources in `Raw/` and no synthesis yet. The first ingest "
          f"should replace this stub with a real orientation: what the domain is for, the two or "
          f"three ideas that organize it, and where a reader should start.\n", dry)

    ents, cons = spec["entities"], spec["concepts"]
    for kind, items in (("entities", ents), ("concepts", cons)):
        for term, desc in items.items():
            write(v / "Wiki" / kind / f"{term}.md",
                  fm(type={"entities": "entity", "concepts": "concept"}[kind], title=term, tags=[spec["subject"]], sources="", updated=TODAY,
                     description=desc) +
                  f"\n\n# {term}\n\n> STUB created at bootstrap. Fill on the first ingest that "
                  f"cites a source about it.\n\n{desc[0].upper() + desc[1:]}.\n", dry)

    # Wiki/index.md (reserved: frontmatter-free)
    idx = [f"# Wiki Index — {name}", "", "## Overview", "",
           f"- [overview](overview.md) - top-level synthesis of {spec['domain']}", ""]
    if ents:
        idx += ["## Entities", ""] + [f"- [{t}](entities/{quote(t)}.md) - {d}" for t, d in ents.items()] + [""]
    if cons:
        idx += ["## Concepts", ""] + [f"- [{t}](concepts/{quote(t)}.md) - {d}" for t, d in cons.items()] + [""]
    idx += ["## Papers", "", "_None yet._", "", "## Summaries", "", "_None yet._", ""]
    write(v / "Wiki" / "index.md", "\n".join(idx), dry)

    # Wiki/log.md (reserved: frontmatter-free, newest-first date groups)
    n_stub = len(ents) + len(cons)
    write(v / "Wiki" / "log.md",
          f"# Wiki Log — {name}\n\n## {TODAY}\n"
          f"* **Creation**: Bootstrapped this bundle. Created `index.md`, `Wiki/index.md`, "
          f"`Wiki/log.md`, `Wiki/overview.md`"
          + (f", {len(ents)} stub entity and {len(cons)} stub concept pages, and the seven "
             f"`Learning Path/` stage skeletons plus README" if not catalog else
             ", and no stub or curriculum pages (catalog bundle — see CLAUDE.md)")
          + f". `Raw/` already held {raw_n} sources from the corpus-expansion sync. "
          f"{'Stub terms were chosen from term frequencies over this vault own Raw corpus rather than cloned from a sibling vault, to avoid scaffolding nobody fills. ' if n_stub else ''}"
          f"No ingest has run yet: every wiki page here is a stub.\n"
          f"  - touched: [index](index.md), [overview](overview.md), [log](log.md)\n", dry)

    if catalog:
        return

    # Learning Path
    lp = v / "Learning Path"
    rows = []
    for i, (st, scope) in enumerate(spec["stages"], 1):
        rows.append(f"{i}. **[{st}]({quote(st)}.md)** — {scope}")
    write(lp / "README.md",
          fm(type="learning-path-index", title=f"Learning Path — {name}", updated=TODAY) +
          f"\n\n# Learning Path — {name}\n\n> SKELETON. Fills in as ingests classify sources into stages.\n\n"
          f"A beginner-to-advanced curriculum over the [{name} wiki](../Wiki/index.md). Each stage "
          f"references wiki pages by relative markdown link rather than restating their content — "
          f"the wiki is the textbook, this path is the table of contents in study order.\n\n"
          "## Map\n\n" + "\n".join(rows) + "\n", dry)
    for st, scope in spec["stages"]:
        write(lp / f"{st}.md",
              fm(type="learning-path-stage", title=st, updated=TODAY) +
              f"\n\n# {st}\n\n> SKELETON. Populated by ingests.\n\n"
              f"**Scope.** {scope[0].upper() + scope[1:]}.\n\n"
              "## Prerequisites\n\n_To be filled._\n\n## Goals\n\n_To be filled._\n\n"
              "## Wiki pages\n\n_None yet._\n\n## Activities\n\n_To be filled._\n", dry)


def main():
    dry = "--dry-run" in sys.argv
    if not REFERENCE.exists():
        sys.exit(f"reference vault not found: {REFERENCE}")
    print(f"Bootstrapping {len(VAULTS)} vault(s){' (dry run)' if dry else ''}")
    for name, spec in VAULTS.items():
        bootstrap(name, spec, dry)
    print("\nDone.")


if __name__ == "__main__":
    main()
