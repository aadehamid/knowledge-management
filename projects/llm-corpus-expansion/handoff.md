---
artifact_contract: "ce-handoff/v1"
created_at: "2026-08-29T21:40:00Z"
title: "LLM corpus expansion: 506-URL DEVONthink ingest and subject-split plan"
summary: "506 URLs captured into DEVONthink and audited; pipeline analysis done; subject split, NotebookLM gating and tiered Wiki ingest decided; no repo/vault writes yet."
keywords: ["devonthink", "knowledge-pipeline", "subject-split", "okf", "notebooklm", "llm-inference-optimization"]
cwd: "/Users/hamidadesokan/Documents/Knowledge Management"
resume_focus: "Run the per-URL routing pass assigning all 546 records to the eight subject areas, then seed references/papers/<subject>/ from the DEVONthink captures."
repository: "github.com/aadehamid/knowledge-management"
repo_root_sha: "a7cf56367bb03b80595f68a745a62575a5b82c7c"
branch: "main"
head: "8064b8dbc7e76e96c24f60036abe88778b7b8ae9"
---

# LLM corpus expansion — session handoff

## Objective

Expand the knowledge base with a 506-URL reading list, routed into subject areas and
landed in the existing repo → Obsidian vault → NotebookLM pipeline in OKF v0.2 format.

The corpus is already captured in DEVONthink. **Nothing has been written to the repo
pipeline or any vault yet** beyond this project folder. The next unit of work is the
per-URL routing pass.

## What exists now

### DEVONthink (machine-local state)

All 506 URLs from `data/source-urls-506.txt` are filed in the **LLMs** database,
group `/Inference Optimization/`. Flat filing was the user's choice.

| | |
|---|---|
| Records in group | 546 |
| Markdown / PDF / bookmark | 405 / 20 / 121 |
| Total words | ~1,107,000 |
| Ingest outcome | 427 captured, 52 bookmarked, 27 already present, 0 errors |
| Still thin (<120 words) | 20 |

Machine-local identifiers, needed to re-query this state:

- LLMs database UUID `B5B4C46F-78E5-4A0A-A33A-4B4E54770CF3`
  (at `~/Databases/LLMs.dtBase2`; it is normally **closed** — open it via
  DEVONthink's AppleScript API, never by touching the package)
- `Inference Optimization` group UUID `9E8EF3C1-C843-45B5-B1DB-567146668171`

Per-URL outcomes are in `data/ingest-outcomes.tsv`; the current group inventory
(type, word count, URL, title) is in `data/devonthink-inventory.tsv`. The inventory
is the input the routing pass should read.

Capture format was chosen by the user: readability Markdown, with bookmark fallback
for media/gated hosts (YouTube, Colab, x.com, learn.deeplearning.ai).

### Repairs already applied to the captures

Do not redo these — they are done and verified:

- 6 GitHub pages that returned rate-limit stubs were retried; 5 recovered, 3 renamed
- 14 GitHub file URLs (notebooks, `.py`, `.yml`, READMEs) captured only the code-viewer
  shell under readability; 12 were refetched from `raw.githubusercontent.com`, with
  notebooks converted to fenced Markdown
- 2 files whose repos were renamed were recovered from git history and repointed at
  commit-pinned permalinks: `tiny-llama.yml` (axolotl `4d2e842e`) and
  `Prompt_Engineering_with_Llama_2.ipynb` (llama-cookbook `85ea8691`)
- 26 JS-rendered pages were re-captured as PDF; 11 kept where the PDF beat the stub
  (distill.pub 32→6,094 words; DigitalOcean 63→4,508; nanoVLM PR 109→3,827)

The remaining 20 thin records are legitimately thin: JS app shells (Lightning studios),
Panopto video players, a paywalled O'Reilly TOC, and a dead `ai.facebook.com` post
with no Wayback snapshot.

### Repo and vault — unchanged

`resources/sources/llm-inference-optimization/urls.txt` still holds its original 42
URLs; `references/papers/llm-inference-optimization/` still holds 39 `.md` + 39
`.meta.json`; the vault still has 39 Raw files, 78 Wiki pages, 8 Learning Path stages.

Overlap between the new corpus and the existing 42 URLs is only **5 URLs** — this is a
~12x corpus expansion, not a top-up.

## Decisions

Marked by whose call each was.

1. **Eight subject areas — three existing, five new.** *(User's decision.)* The user
   reviewed the breakdown, approved creating five new subjects, and explicitly declined
   the offered merge of `rag-retrieval` + `llm-landscape` into `ai-engineering`
   ("dont merge those 3").

   | Subject | Status | approx n |
   |---|---|---|
   | `llm-inference-optimization` | exists | ~135 |
   | `transformers` | exists | ~65 |
   | `cuda` | exists | ~15 |
   | `llm-finetuning` | **new** | ~127 |
   | `ml-foundations` | **new** | ~86 |
   | `llm-landscape` | **new** | ~49 |
   | `ai-engineering` | **new** | ~26 |
   | `rag-retrieval` | **new** | ~10 + OCR/document-AI strays |
   | (inbox / unroutable) | — | ~33 |

   Counts are heuristic estimates only — see "Rejected approaches" below. They describe
   the shape of the corpus; they are not the routing.

2. **DEVONthink becomes the fetch cache for HTML sources; the repo pipeline stays the
   fetcher for video and PDF.** *(My recommendation, accepted implicitly when the user
   approved the plan; not separately restated by the user.)* Rationale, measured rather
   than assumed:

   - `markitdown` has no readability pass — its Substack conversion opens with logo
     images, "SubscribeSign in", and avatar links. DEVONthink's readability capture is clean.
   - `markitdown` cannot execute JS; DEVONthink's WebKit rendering already recovered
     several thousand words on JS-only pages.
   - **But** `markitdown` pulls YouTube transcripts (6,792 words for one lecture, 13,067
     for CS336) where DEVONthink gives only a bookmark, and `pymupdf4llm` extracts PDF
     images inline. So video and PDF must stay with the repo pipeline.

   `urls.txt` remains the single source of truth for provenance in both cases.

3. **NotebookLM pushes must be gated before the first sync of any new subject.**
   *(User agreed.)* See the landmine note under "Traps" — this is not optional cleanup.

4. **Tiered Wiki ingest — PROPOSED BY ME, NOT YET CONFIRMED BY THE USER.** The user
   answered the subject-split and NotebookLM questions explicitly but did not
   separately confirm this one. Treat it as a proposal awaiting a decision:

   - Tier 1 (Wiki ingest now): `llm-inference-optimization` only, prioritising sources
     that fill known gaps — the `Wiki/concepts/HBM vs SRAM` bootstrap stub is still
     open from the 2026-06-15 run despite heavy inbound references.
   - Tier 2 (Raw + NotebookLM, no Wiki): finetuning, transformers, ml-foundations.
   - Tier 3 (Raw only): landscape, inbox, dead pages.

   Rationale: the 39 existing Raw files took 9 reviewed thematic batches. 546 files at
   that rate is months of work, and most of the corpus would not earn a wiki page.

## Load-bearing references

Repository-relative, at the `head` recorded above.

- `scripts/convert_pdfs.py:236,252` — `existing_md = {p.stem for ...}` then
  `if stem in existing_md: ... continue`. **This is the pre-seeding hook that makes the
  whole plan work**: a `.md` already present at the expected stem makes the cloud agent
  skip the fetch and write only the sidecar. No code change is needed to inject
  DEVONthink content.
- `scripts/convert_pdfs.py:114` — `get_url_stem()`. Pre-seeded files **must** be named
  with the stem this function computes for the URL, or the converter will fetch anyway
  and create a duplicate.
- `scripts/sync_to_vault.py:116` — `build_frontmatter()`, the OKF v0.2 Raw frontmatter
  writer (`type: Source`, `fetch_status`, preserved `wiki_refs`).
- `scripts/sync_to_vault.py:282` — `vault_dir.mkdir(parents=True, exist_ok=True)`.
  Sync creates `Raw/` itself, so **landing Raw files does not require the vault
  bootstrap workflow**; bootstrap is only needed before Wiki ingest in that vault.
- `scripts/sync_to_vault.py:425` — the NotebookLM auto-provisioning branch. See traps.
- `scripts/test_okf_bundle.py` — the OKF v0.2 conformance gate. Run it against a vault
  before declaring any batch done.
- `LLM Inference Optimization/CLAUDE.md:361-399` (machine-local, in the Obsidian vault,
  not this repo) — the OKF format conventions adopted 2026-08-29: markdown links not
  wikilinks, `fetch_status` not `status`, frontmatter-free `index.md`/`log.md`,
  newest-first date-grouped log entries.
- `README.md` — pipeline overview, `urls.txt` line format
  (`url | title | source_type [| author]`), and the cloud-agent invocation.

## Traps and fragile state

- **NotebookLM auto-provisioning.** Merely adding the five new subjects to
  `sync_config.json` causes `sync_to_vault.py:425` to create a notebook per subject and
  then push every synced file into it, one `nlm source add` per file. NotebookLM caps
  sources per notebook (~50 free / ~300 Pro). Gate this *before* the first sync.
- **DEVONthink group tagging.** The LLMs database has group-based tagging on, so all 546
  records carry an automatic `Inference Optimization` tag inherited from the group name.
  It is not a topical judgement and must not be read as one during routing.
- **Databases are packages.** Never modify `.dtBase2` contents on the filesystem. All
  record operations go through the DEVONthink MCP server or AppleScript.
- **The vault is not git-tracked.** Deletions under
  `/Users/hamidadesokan/Documents/Knowledge Management/` are irreversible. Back up first.
- **`sync_to_vault.py` slug-collision guard** still fails when only the `source_type`
  prefix changes (doc→pdf), creating orphaned ghost duplicates on rename. Known, unfixed.
- **AppleScript gotcha**, if reusing the ingest scripts: inside a
  `tell application id "DNtp"` block, `tab` resolves to DEVONthink's browser-tab class,
  not the AppleScript tab character, so `& tab &` silently emits the literal string
  "tab". The ingest log in `data/ingest-outcomes.tsv` was repaired for this.

## Rejected approaches — do not retry

- **Keyword/regex classification for routing.** I built an ordered first-match-wins
  classifier and it is not good enough. Verified failures: Unsloth → inference (should
  be finetuning), nanoGPT → finetuning (should be transformers), backprop → transformers
  (should be ml-foundations), DeepSeek-OCR → unclassified. Regex cannot separate
  "quantization for serving" from "quantization during fine-tuning". The classifier
  output was deliberately **not** committed so it cannot be mistaken for routing. The
  bucket counts in the decisions table are all that survived it.
- **Guessing new paths for renamed GitHub repos.** Five candidate URL shapes for the
  moved axolotl and llama-cookbook files all 404'd. What worked: `gh api` for the commit
  that removed the path, then fetching the blob at that commit's parent.
- **Hammering archive.org.** Both the availability and CDX APIs return HTTP 429 under
  rapid sequential access; the failure is silent, surfacing as 4-word capture stubs
  rather than an error. A retry job was queued and then killed at the user's request, so
  two dead-link stubs (a Towards Data Science article and a ModernBERT post) remain
  unrestored. Space requests out if this is retried.

## Verification performed

- Ingest reconciled 506/506 with zero errors; every bookmark fallback was a
  pre-classified media/gated host, with no unexpected fallbacks.
- Post-ingest audit found 46 thin records; after repairs, 20 remain, each inspected and
  confirmed legitimately thin.
- Pre-seeding claim verified by reading the converter's skip logic, not assumed.
- "Raw sync needs no bootstrap" verified by reading the `mkdir` call, not assumed.
- Fetcher comparison verified against real converted output in
  `references/papers/llm-inference-optimization/`, not assumed.
- Repo `git status` clean and level with `origin/main` before this commit.

## Plausible next steps

The main path is sequential — each step depends on the previous:

1. **Routing pass.** Read all 546 rows of `data/devonthink-inventory.tsv` in batches and
   assign each to one of the eight subjects by judgement on title and captured content,
   with a confidence marker. Produce the assignment table for the user to review
   *before* any file is written, along with the genuine judgement calls.
2. **Seed and register.** For each subject: generate `urls.txt` lines with correct
   `source_type` and `author`, export DEVONthink Markdown to
   `references/papers/<subject>/<get_url_stem>.md` with matching `.meta.json`, and add
   the `sync_config.json` mapping — with the NotebookLM gate in place first.
3. **Dry-run and sync.** Run `scripts/test_okf_bundle.py` against a vault before
   declaring the batch done.

One genuine fork, still open: whether the tiered Wiki-ingest policy in decision 4 is
adopted as written, narrowed, or replaced.

## Relevant installed skills

`compound-engineering:ce-plan` for sequencing the routing and seeding work;
`compound-engineering:ce-code-review` before merging any change to `scripts/`.
