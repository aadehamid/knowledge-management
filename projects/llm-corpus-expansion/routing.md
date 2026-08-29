# Routing pass — all 546 records assigned

Completed 2026-08-29. Every record in the DEVONthink `Inference Optimization` group was
read and assigned a subject by judgement on title and captured content. Output is
`data/routing.tsv`; this document explains the method and surfaces the calls that need
a human decision.

**Nothing has been written to `resources/sources/` or any vault yet.** This is a proposal.

## Method

Records were read in six batches and assigned one of eight subjects, or `_inbox` for
sources with no usable content and no clear home. Each carries a confidence marker:
`hi` where the subject is unambiguous from the title, `lo` where the source straddles
two subjects or the title is uninformative (bare "Google Colab", "Medium", "YouTube").

A keyword classifier was built first and **rejected** — it misrouted Unsloth to
inference, nanoGPT to fine-tuning, and backprop to transformers, because first-match
regex cannot separate "quantization for serving" from "quantization during fine-tuning".
See the handoff's "Rejected approaches".

## Result

| Subject | Records | Unique URLs | Net-new to add | Low-confidence | Bookmark-only |
|---|---|---|---|---|---|
| `llm-finetuning` | 139 | 135 | **133** | 20 | 29 |
| `llm-inference-optimization` | 116 | 105 | **79** | 8 | 51 |
| `ml-foundations` | 89 | 75 | **80** | 14 | 13 |
| `llm-landscape` | 48 | 48 | **48** | 13 | 5 |
| `transformers` | 43 | 41 | **39** | 9 | 3 |
| `ai-engineering` | 42 | 40 | **38** | 8 | 3 |
| `rag-retrieval` | 34 | 33 | **33** | 1 | 3 |
| `_inbox` | 22 | 21 | **20** | 1 | 12 |
| `cuda` | 13 | 13 | **10** | 4 | 2 |
| **Total** | **546** | **515** | **480** | **78** | **121** |

546 records resolve to 515 unique URLs (31 duplicate records) of which **35 are already
in a repo `urls.txt`** — these are the pre-existing DEVONthink records from the earlier
LRIO pipeline work, not new material. **480 lines are genuinely net-new.**

Note the corpus is far more fine-tuning than inference, despite the source file being
named `llm-inference-optimization-urls.txt`.

## Decisions needed

### 1. Five sources already filed under `llm-inference-optimization` that I would move

These are in the existing `resources/sources/llm-inference-optimization/urls.txt` but
route elsewhere on content:

| # | Title | Currently | Proposed |
|---|---|---|---|
| 10 | Let's build GPT: from scratch (Karpathy) | LRIO | `transformers` |
| 12 | `nanochat/gpt.py` (Karpathy) | LRIO | `transformers` |
| 18 | A Curated List of ML System Design Case Studies | LRIO | `ai-engineering` |
| 19 | `aadehamid/system-design` | LRIO | `ai-engineering` |
| 32 | Computer Architecture (algorithmica HPC) | LRIO | `cuda` |

Worth noting: the 2026-06-15 vault run independently judged `lets-build-gpt`,
`system-design-interview-prep`, and `computer-architecture` **out of scope** for LRIO
and recorded that in the end-of-run decision list. This routing reproduces that
judgement from content alone. Moving them makes the earlier call durable instead of a
note in a log — but it means editing an existing `urls.txt`, which has downstream
effects on already-converted files.

### 2. Training-systems sources — filed under `llm-finetuning`, but they are neither

ZeRO (#51), PipeDream (#57), "How to Parallelize a Transformer for Training" (#60),
"Democratizing AI: Open-source Scalable LLM Training on GPU Supercomputers" (#174).

These are distributed-*training* systems papers. They are not fine-tuning, not
inference, and not GPU-kernel material. I put them in `llm-finetuning` as the closest
fit, but a `distributed-training` subject would hold them properly. Four sources is
thin for its own vault; the alternative is accepting them in `llm-finetuning`.

### 3. Inference hardware / architecture — split across two subjects

"Domain specific architectures for AI inference" (#516) and "AI Chip Architectures"
(#93) went to inference; "The Case for Co-Designing Model Architectures with Hardware"
(#517) and "Three Other Models of Computer System Performance" (#58) went to `cuda`.
The boundary is genuinely arbitrary. Pick one home for inference-hardware material and
I will make it consistent.

### 4. `RAG vs Fine-tuning` (#311, #324) — coin flip

Same paper twice, one arXiv and one HF paper page. Routed to `llm-finetuning`; equally
defensible in `rag-retrieval`.

### 5. Model pages vs. run guides

Model releases (Gemma 3, Mistral Small, Llama 3.2, `ollama/llama3`) → `llm-landscape`.
Unsloth "How to Run" guides for the *same models* → `llm-inference-optimization` or
`llm-finetuning`. Defensible, but it means a Gemma 4 release note and the Gemma 4 run
guide land in different vaults.

### 6. `Recent Developments in LLM Architectures: KV Sharing, mHC, Compressed Attention` (#476)

Raschka. Architecture survey by framing, inference-efficiency by content. Routed to
inference.

### 7. `modal.com/blog/truly-serverless-gpus` (#471)

Already the CUDA vault's one known net-new source per the 2026-06-15 state. Routed to
`cuda`, consistent — flagged so it is not double-added.

### 8. The `_inbox` 20

Colab drive links with no title, Panopto players, dead x.com posts, an "Access Denied"
O'Reilly TOC, a GitHub Pages 404. Options: drop them, keep them as bookmark-only Raw
files for the record, or hold them out of the pipeline entirely. My recommendation is
to drop them — they carry no content and their URLs are mostly dead or unresolvable.

## Notes for the seeding step

- `llm-finetuning` is the largest subject at 133 net-new sources — larger than the
  entire existing LRIO corpus. Consider seeding it in tranches.
- 121 records are bookmark-only (no captured text). 51 of those are in
  `llm-inference-optimization`. For YouTube among them the repo pipeline will do better
  than DEVONthink did — `markitdown` pulls transcripts — so those should go through the
  normal converter rather than being pre-seeded.
- Low-confidence assignments cluster where the title is uninformative. Nine bare
  "Google Colab" / "Medium" / "YouTube" titles were routed from their URL path alone
  and are the likeliest individual errors.
