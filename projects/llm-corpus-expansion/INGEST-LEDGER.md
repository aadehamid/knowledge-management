# Ingest ledger

The running record of which sources have been turned into wiki knowledge, and which
have not. **Update this at the end of every batch.** It is the resume point: a fresh
agent should be able to read this file and know exactly where to pick up.

Counting rule: a Raw source is *processed* when some page under `Wiki/` cites its
filename. Recompute with the snippet at the bottom rather than trusting the numbers.

## Status — 2026-08-30 (updated: batch 4 done, DeepSeek reviewer wired)

| Vault | Raw | Processed | Remaining | Bundle shape |
|---|---|---|---|---|
| Transformer from Scratch | 218 | 187 | **31** | concept + curriculum |
| LLM Fine-tuning | 123 | 81 | **42** | concept + curriculum |
| LLM Inference Optimization | 118 | 51 | **67** | concept + curriculum |
| ML Foundations | 73 | 21 | **52** | concept + curriculum |
| LLM Landscape | 47 | 47 | **0** | catalog (no curriculum) |
| AI Engineering | 39 | 9 | **30** | concept + curriculum |
| Document AI and Retrieval | 32 | 5 | **27** | concept + curriculum |
| CUDA from Scratch | 17 | 17 | **0** | concept + curriculum |
| **Total** | **667** | **418** | **249** | |

## Batch order

Sequenced so that foundational vocabulary lands before the material that leans on it,
and so each vault's stub pages get filled early — a stub nobody fills is worse than a
missing page.

### LLM Fine-tuning (123 sources)
| # | Cluster | Sources | Status |
|---|---|---|---|
| 1 | LoRA fundamentals and hyperparameters | 4 | **done** — Codex review, 8 findings applied |
| 2 | RLHF / DPO / GRPO — preference and RL post-training | 4 | **done** — Codex review, 8 findings applied |
| 3 | Datasets and synthetic data | 8 | **done** — review deferred to the batch-5 gate |
| 4 | Unsloth tooling and run guides | 13 | **done** — DeepSeek review deferred to batch-6 gate |
| 5 | Frameworks: Axolotl, ms-swift (+4 config/example sources) | 7 | **done** — review deferred to batch-6 gate |
| 6 | Quantized training and memory | 8 | **done** — review pending (DeepSeek gate) |
| 7 | Vision and multimodal fine-tuning | ~15 | next |
| 8 | Courses and overviews | ~14 | |
| 9 | MLX / Apple silicon, distillation, remainder | ~10 | |

### ML Foundations (73)
Backprop cluster first — 1,352 mentions, the densest coherent cluster in the whole
corpus. Then autodiff, training dynamics, generalization, probability, courses.

### LLM Inference Optimization (83 new)
Already the most mature wiki. Prioritise sources that fill known gaps over breadth —
`concepts/HBM vs SRAM` has been an unfilled bootstrap stub since May despite heavy
inbound references.

### Transformer from Scratch (129)
Largest remaining. Much of it is the from-scratch / nanoGPT / tokenizer material.

### Document AI and Retrieval (32)
OCR-dominant (595 mentions). Document parsing, then embeddings, then rerankers.

### AI Engineering (39)
Agents (312 mentions), then evals, MCP, prompt/context engineering.

### CUDA from Scratch (17) — **DONE**
All 17 sources processed across batches (final batch: AI chip architectures, performance engineering, profiling, communications, floating-point formats).

### LLM Landscape (47) — **DONE**
Catalog bundle — completed as a single overview.md catalog entry per source (title,
filename, word count, one-line description). No concept pages, no curriculum, per
standing decision. All 47 sources marked processed in one pass.

## Per-batch procedure (non-negotiable steps)

1. Plan the cluster; state sources and scope calls. No per-file takeaways pause.
2. Read every source end-to-end before writing. Claims must trace to the source —
   **outside knowledge is invention**, however true it is. (Batch 1 lost FSDP mechanics
   to this.)
3. Write summaries; fill/extend entity and concept pages; update `Wiki/index.md`,
   `Wiki/log.md`, Raw `wiki_refs`, and the Learning Path stages.
4. Trust stamps on every touched page: `generated: { by, at }`.
5. Learning Path: clear `> SKELETON.`, add `> Populated with N sources: ...`, bump `updated`.
6. Self-QA: `scripts/test_okf_bundle.py <vault>`, link resolution, wiki_refs round-trip.
7. **Independent Codex review, report-only — once every three batches** (user direction,
   2026-08-30, to cut overhead). Review the whole three-batch span, apply every HIGH and
   MEDIUM, re-QA, record reviewer and finding counts in `log.md`, and save the review
   under `projects/llm-corpus-expansion/reviews/`. Batch size is 8-12 sources.
   **Known exposure:** both reviews run so far returned needs-rework, and both found the
   same class of error — claims stated more strongly than the source supports. Deferring
   means up to ~30 sources can carry that before it is caught. Compensate at draft time:
   keep the source's hedges verbatim, and treat any sentence writable without the source
   open as suspect.
8. Update this ledger.

## Recount snippet

```python
from pathlib import Path
KM=Path("/Users/hamidadesokan/Documents/Knowledge Management")
for v in ["Transformer from Scratch","CUDA from Scratch","LLM Inference Optimization",
          "LLM Fine-tuning","ML Foundations","LLM Landscape","AI Engineering",
          "Document AI and Retrieval"]:
    raw=sorted((KM/v/"Raw").glob("*.md"))
    wiki="".join(p.read_text(encoding="utf-8",errors="ignore")
                 for p in (KM/v/"Wiki").rglob("*.md"))
    done=sum(1 for r in raw if r.name in wiki)
    print(f"{v:<30}{len(raw):>5}{done:>7}{len(raw)-done:>8}")
```

## Standing decisions

- `LLM Landscape` is catalog-shaped by design; do not grow a concept layer there.
- `Document AI and Retrieval` was renamed from `rag-retrieval` because OCR dominates.
- NotebookLM stays gated: the five new subjects have `null` notebook ids, which blocks
  both auto-provisioning and push. Lift deliberately, per subject, after content settles.
- Judgment calls that are genuinely the user's go in the end-of-run decision list, not
  into a mid-run interruption.
