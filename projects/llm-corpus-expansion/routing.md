# Routing pass — all 546 records assigned

Completed 2026-08-29. Every record in the DEVONthink `Inference Optimization` group was
read and assigned a subject by judgement on title and captured content. Output is
`data/routing.tsv`; this document explains the method and surfaces the calls that need
a human decision.

**Nothing has been written to `resources/sources/` or any vault yet.** All routing
decisions are resolved (see below); seeding is the next step.

## Method

Records were read in six batches and assigned one of eight subjects, or `_inbox` for
sources with no usable content and no clear home. Each carries a confidence marker:
`hi` where the subject is unambiguous from the title, `lo` where the source straddles
two subjects or the title is uninformative (bare "Google Colab", "Medium", "YouTube").

A keyword classifier was built first and **rejected** — it misrouted Unsloth to
inference, nanoGPT to fine-tuning, and backprop to transformers, because first-match
regex cannot separate "quantization for serving" from "quantization during fine-tuning".
See the handoff's "Rejected approaches".

## Result — as first routed, before the decisions below

| Subject | Records | Unique URLs | Net-new to add | Low-confidence | Bookmark-only |
|---|---|---|---|---|---|
| `llm-finetuning` | 139 | 135 | **133** | 20 | 29 |
| `llm-inference-optimization` | 116 | 105 | **79** | 8 | 51 |
| `ml-foundations` | 89 | 75 | **80** | 14 | 13 |
| `llm-landscape` | 48 | 48 | **48** | 13 | 5 |
| `transformers` | 43 | 41 | **39** | 9 | 3 |
| `ai-engineering` | 42 | 40 | **38** | 8 | 3 |
| `document-ai-retrieval` | 34 | 33 | **33** | 1 | 3 |
| `_inbox` | 22 | 21 | **20** | 1 | 12 |
| `cuda` | 13 | 13 | **10** | 4 | 2 |
| **Total** | **546** | **515** | **480** | **78** | **121** |

546 records resolve to 515 unique URLs (31 duplicate records) of which **35 are already
in a repo `urls.txt`** — these are the pre-existing DEVONthink records from the earlier
LRIO pipeline work, not new material. **480 lines are genuinely net-new.**

Note the corpus is far more fine-tuning than inference, despite the source file being
named `llm-inference-optimization-urls.txt`.

## Decisions — resolved 2026-08-29

All eight open questions are closed. `data/routing.tsv` reflects the resolutions below.

### 1. Sources already in LRIO that route elsewhere — move 4, keep 1

**The user's criterion was to make each OKF bundle as topically coherent as possible.**
Checking the vault before acting changed the answer on one of the five:

| # | Source | Resolution |
|---|---|---|
| 10 | Let's build GPT (Karpathy) | → `transformers` |
| 18 | ML System Design Case Studies | → `ai-engineering` |
| 19 | `aadehamid/system-design` | → `ai-engineering` |
| 32 | Computer Architecture (algorithmica) | → `cuda` |
| 12 | `nanochat/gpt.py` (Karpathy) | **stays in LRIO** |

Moving the first four costs nothing: `Wiki/summaries/out-of-scope-records.md` already
records #19, #32 and #10 as assessed-and-out-of-scope, explicitly stating "no wiki pages,
enrichment, or claims were derived from them". #18's own summary says the bulk is outside
the vault's focus and that no concept page links to it. Relocating them makes a judgement
that currently lives in a log durable in the corpus itself.

**#12 was my routing error.** I assigned it to `transformers` on the title. In fact
`Wiki/entities/KV Cache.md` cites it as the production-style reference that builds the
cache on the FlashAttention `flash_attn_with_kvcache` kernel alongside GQA and sliding
window, and it has a full summary page. It is inference material despite the
"build a ChatGPT" framing, and moving it would break a live citation in a core entity.

Vault cleanup this implies (**not yet performed** — the vault is not git-tracked, so it
needs a backup first): delete 4 `Raw/` files, delete `Wiki/summaries/out-of-scope-records.md`
and `Wiki/summaries/ml-system-design-case-studies-catalog.md`, drop their `Wiki/index.md`
entries, and add a `Wiki/log.md` entry.

### 2. Distributed-training papers → `llm-inference-optimization`

User's decision. ZeRO (#51), PipeDream (#57), "How to Parallelize a Transformer for
Training" (#60) and "Democratizing AI: Open-source Scalable LLM Training" (#174) are now
routed to LRIO rather than parked in `llm-finetuning`. No separate `distributed-training`
subject is created.

### 3. Drop anything with no retrievable content — 27 records removed

User's decision ("remove any doc that adds no value"). The criterion applied is **no
retrievable content**, not "bad title":

| Reason | n |
|---|---|
| Dead or blocked page (404, 403, Cloudflare, edX auth, Panopto) | 12 |
| Private Colab drive link, nothing behind it | 4 |
| Truncated Colab URL, 2-4 candidate notebooks, unresolvable | 5 |
| x.com post with no retrievable content | 2 |
| Bare YouTube embed fragment | 1 |
| Job posting / blog index / site landing page | 3 |

**`_inbox` is now empty** and dissolves as a bucket.

Two things were deliberately *not* dropped:

- **Six records with real content but a useless `<title>`** — Medium and similar serving
  a bare "Medium" tag. This includes Karpathy's "Yes you should understand backprop"
  (1,598 words). Titles were repaired from the URL slug rather than the records discarded.
- **Three Colab links that were recoverable** — rewritten from
  `colab.research.google.com/github/...` to `raw.githubusercontent.com/...`, including
  one truncated Unsloth notebook whose name resolved unambiguously against the repo's
  file list. The other five truncated ones had 2-4 candidates each and were dropped.

### 4-8. Remaining calls

Resolved as routed, no change: `RAG vs Fine-tuning` stays in `llm-finetuning`;
inference-hardware material stays split, with chip/architecture surveys in
`llm-inference-optimization` and CPU/GPU performance-modelling in `cuda`; model release
pages stay in `llm-landscape` while run/fine-tune guides follow their task;
`truly-serverless-gpus` (#471) is flagged as already present in `cuda`.

## Final shape

| Subject | Net-new `urls.txt` lines |
|---|---|
| `llm-finetuning` | 126 |
| `llm-inference-optimization` | 82 |
| `ml-foundations` | 79 |
| `llm-landscape` | 48 |
| `transformers` | 39 |
| `ai-engineering` | 38 |
| `document-ai-retrieval` | 33 |
| `cuda` | 10 |
| **Total** | **455** |

455 net-new sources, from 546 records: 31 duplicates, 35 already in a repo `urls.txt`,
27 dropped as valueless.

## Notes for the seeding step

- `llm-finetuning` is the largest subject at 126 net-new sources — more than three times
  the entire existing LRIO corpus (39 files). Consider seeding it in tranches.
- 121 records are bookmark-only (no captured text). 51 of those are in
  `llm-inference-optimization`. For YouTube among them the repo pipeline will do better
  than DEVONthink did — `markitdown` pulls transcripts — so those should go through the
  normal converter rather than being pre-seeded.
- Low-confidence assignments cluster where the title is uninformative. After the drop
  pass and the six title repairs, the remaining low-confidence rows are genuine
  subject-boundary calls rather than missing metadata.
- The `#12 nanochat` correction is a reminder that titles alone misroute: check whether
  a source is already cited in a vault before relocating it.
