# Ingest runbook

Operating instructions for the agent doing the vault ingest. Follow these literally.
When this file and your own judgement disagree, follow this file.

**What you are doing:** turning fetched source documents (`Raw/`) into an interlinked
wiki (`Wiki/`) plus a curriculum (`Learning Path/`), one thematic batch at a time, across
eight Obsidian vaults. ~518 sources remain. `INGEST-LEDGER.md` says where to start.

**What "done" means for one batch:** every source read, pages written, `check_ingest.py`
clean, ledger updated, work committed.

---

## THE ONE RULE THAT MATTERS

**Every factual sentence you write must be traceable to a line in one of the source files
for that batch.**

Not "true". Not "well known". **In the source.** If you would write a sentence without
opening the source, delete it.

This is the rule that has been broken most. Two independent reviews of earlier batches
both returned `needs-rework`, and both found the same failure:

| What was written | What the source said |
|---|---|
| "FSDP shards parameters, gradients and optimizer state" | Nothing. `optimizer state` appears **0 times**. The writer knew this from elsewhere. |
| "reward models are often much smaller than the policy" | Sizes **vary**; only one of three examples was smaller. |
| "the gap occurs **because** of BA optimization dynamics" | "The **likely reason** is ... **on this dataset**" |
| "needs to absorb about 320,000 bits" | Same number, but the source **assumes one bit per completion** and calls it an upper bound |
| "the paper's **answer** is the generation-verification gap" | The paper "found the **most support** for" it, among hypotheses |

Three habits prevent all five:

1. **Copy the hedge.** If the source says "likely", "suggests", "we found the most support
   for", "on this dataset", "assuming X" — those words are part of the claim. Carry them.
2. **Attribute contested claims.** "The authors report…", "the guide recommends…" is
   almost always safer than the bare assertion.
3. **Check quantified claims twice.** Numbers and comparative words ("smaller", "faster",
   "most") are where errors concentrate. Re-read the sentence in the source before writing
   it down.

If a source is genuinely thin, write a short page. A three-line honest page beats a
twenty-line padded one. If a topic needs something no source covers, **say so on the
page** — "the sources here do not explain X" is a legitimate and useful sentence.

---

## Paths

| What | Where |
|---|---|
| Repo | `/Users/hamidadesokan/Dropbox/1_PROJECTS/knowledge-management` |
| Vaults | `/Users/hamidadesokan/Documents/Knowledge Management/<Vault Name>/` |
| Python | `<repo>/.venv/bin/python` (has the deps) |

Eight vaults: `Transformer from Scratch`, `LLM Fine-tuning`, `LLM Inference Optimization`,
`ML Foundations`, `LLM Landscape`, `AI Engineering`, `Document AI and Retrieval`,
`CUDA from Scratch`.

Each vault has: `Raw/` (sources, **read-only except frontmatter `wiki_refs`**), `Wiki/`
(what you write), `Learning Path/` (curriculum), `CLAUDE.md` (that vault's schema),
`index.md`.

**The vaults are NOT in git.** Deleting a vault file is unrecoverable. You never need to
delete one — if you think you do, stop and ask.

---

## The batch loop

### Step 0 — Pick the batch

Read `INGEST-LEDGER.md`. Take the next cluster marked `next`.

Find sources not yet processed (a source is processed when some `Wiki/` page names its
filename):

```bash
VAULT="/Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning"
python3 - <<'PY'
from pathlib import Path
V=Path("/Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning")
wiki="".join(p.read_text(encoding="utf-8",errors="ignore") for p in (V/"Wiki").rglob("*.md"))
for r in sorted((V/"Raw").glob("*.md")):
    if r.name not in wiki:
        print(len(r.read_text(encoding='utf-8',errors='ignore').split()), r.name)
PY
```

Choose **8–12 sources on one topic**. Same topic matters more than exact count: a batch
should produce pages that link to each other. Prefer sources with real word counts; a
200-word page shell has little to give.

### Step 1 — Read every source, completely

```bash
sed -n '1,400p' "$VAULT/Raw/<file>.md"
```

Do not skim. Do not write from the title. You will be reviewed against these files.

While reading, note: the specific numbers, the hedged claims, and anything that
contradicts another source in the batch. **Contradictions are valuable — record both
sides and say they disagree. Never silently pick a winner.**

### Step 2 — Write the pages

Write a Python script that emits the files (easier to get frontmatter right than editing
by hand). Copy the template below exactly.

**Page types and where they go:**

| Type | Path | When |
|---|---|---|
| `summary` | `Wiki/summaries/<slug>.md` | one per source, or one per group of closely-related sources |
| `concept` | `Wiki/concepts/<Name>.md` | a mechanism or idea (Backpropagation, Reward Model) |
| `entity` | `Wiki/entities/<Name>.md` | a named thing (LoRA, PyTorch, ColBERT) |
| `paper` | `Wiki/papers/<Title>.md` | **required** if the source is an academic paper |

Fill existing stub pages before creating new ones. A stub says
`> STUB created at bootstrap.` — replace that whole line with real content.

**Frontmatter template — copy exactly:**

```yaml
---
type: summary
title: Some Title
tags: [llm-finetuning, lora]
sources:
  - id: doc-lora-without-regret-thinking-mac
    resource: ../../Raw/doc-lora-without-regret-thinking-machines-lab.md
    title: "LoRA Without Regret - Thinking Machines Lab"
updated: 2026-08-30
description: one sentence, lower case, no trailing period
generated: { by: "agent:<your-model-id>", at: "2026-08-30T14:22:05Z" }
---
```

Rules that break things if ignored:

- **Quote any value containing `:`** — `description: "Foo: bar"`. An unquoted colon makes
  YAML read it as a mapping and the file fails validation. This has broken twice.
- `resource:` is a **relative path from the page's own directory**. From
  `Wiki/summaries/` or `Wiki/concepts/` that is `../../Raw/...`; from `Wiki/overview.md`
  it is `../Raw/...`. Wrong depth = broken provenance.
- `at:` must be **the real current UTC time**, not a rounded guess. A future timestamp is
  worse than none. Get it with `date -u +%Y-%m-%dT%H:%M:%SZ`.
- `type:` must be exactly one of: `entity`, `concept`, `paper`, `summary`, `overview`,
  `index`, `Source`, `Process`, `learning-path-stage`, `learning-path-index`.

**Links between pages:** relative markdown links, spaces percent-encoded.
`[Reward Model](../concepts/Reward%20Model.md)`. **Never `[[wikilinks]]`** — the validator
rejects them outside `Raw/`.

### Step 3 — Wire it up

Four things, all required:

**a) `Wiki/index.md`** — add every new page under its section (Entities / Concepts /
Papers / Summaries), as `- [Name](path.md) - one-line description`. A page missing from
the index counts as an orphan and fails the check.

**b) Raw `wiki_refs`** — in each source's frontmatter, list every page that cites it:

```yaml
wiki_refs:
  - Wiki/summaries/my-summary.md
  - Wiki/concepts/Some Concept.md
```

This must match reality in **both** directions. The checker verifies it.

**c) `Learning Path/<stage>.md`** — put each source in the stage it belongs to. When a
stage goes from empty to populated:
- delete the `> SKELETON. Populated by ingests.` line
- add `> Populated with N sources: <names>.` under the `# Heading`
- fill `## Wiki pages`, `## Goals`, `## Prerequisites`
- bump `updated:`, add the `generated:` stamp

**d) `Wiki/log.md`** — add one entry. This file is **frontmatter-free**, date headings
are `## YYYY-MM-DD`, **newest first**, and **one heading per date** (add to today's
heading if it exists — never create a second one).

```markdown
## 2026-08-30
* **Update**: Batch N (<cluster name>) — ingested N sources: <list>. Created N summaries
  and N concepts; filled the <X> stub. Learning Path stage N populated. <Any scope call or
  contradiction you recorded.> Self-QA passed.
  - touched: [index](index.md), [log](log.md), N summaries, N concepts, 1 Learning Path stage, N Raw wiki_refs
```

### Step 4 — Check

```bash
cd /Users/hamidadesokan/Dropbox/1_PROJECTS/knowledge-management
python3 scripts/check_ingest.py "$VAULT" --since $(date +%Y-%m-%d)
```

**Must exit clean.** It checks OKF conformance, link resolution, `wiki_refs` in both
directions, trust stamps (including future timestamps), SKELETON markers, index coverage
and log ordering. Warnings are usually fine; read them and confirm each is expected.

Fix everything it reports, then run it again. Do not proceed with failures.

### Step 5 — Commit

```bash
cd /Users/hamidadesokan/Dropbox/1_PROJECTS/knowledge-management
git add -A && git commit -F - <<'MSG'
Batch N: <cluster> (<n> sources)

<What was ingested and what was produced.>
<Any judgment call you made and why.>

Self-QA: check_ingest.py clean.

Co-Authored-By: <your model> <noreply@anthropic.com>
MSG
git push origin main
```

The vault is not in git, so this commits scripts, the ledger and review records — not the
wiki pages themselves. That is expected.

### Step 6 — Update `INGEST-LEDGER.md`

Mark the batch done, mark the next one `next`, update the counts table using the recount
snippet at the bottom of the ledger. **Do not hand-adjust the numbers — recompute them.**

---

## Every third batch: independent review

Batches are reviewed in groups of three. After batches 3/6/9…, run this **before starting
the next batch**.

```bash
C=~/.claude/plugins/cache/openai-codex/codex/1.0.6/scripts/codex-companion.mjs
node "$C" task --background --effort high "$(cat /tmp/review_brief.md)"
# returns a job id; poll it:
node "$C" status <job-id> --json
node "$C" result <job-id>
```

The brief must contain, with absolute paths:

1. The vault's `CLAUDE.md` (the schema being checked against)
2. **Every Raw source** from the batches under review — labelled *the only permitted
   evidence*
3. The explicit list of files created and edited
4. The six check priorities: factual accuracy/no invention · scope · schema conformance ·
   graph connectivity · consistency · voice
5. `REPORT-ONLY: report findings, do NOT edit any file`
6. A list of the specific claims you are least sure about — **name your own weak points.**
   This is where the review earns its cost. Both previous reviews caught things the author
   explicitly asked them to attack.
7. Requested output: findings tagged HIGH / MEDIUM / LOW with file and line, then one
   verdict of `ship` / `ship-with-fixes` / `needs-rework`

Then:

- **Verify each finding against the source yourself** before applying it. Reviewers are
  sometimes wrong. If a finding is a false positive, document why rather than deleting
  correct content.
- **Apply every HIGH and MEDIUM.** LOW is optional.
- Re-run `check_ingest.py`.
- Record in `Wiki/log.md`: which reviewer, the verdict, the finding counts, and what the
  findings were.
- Save the brief and the findings to
  `projects/llm-corpus-expansion/reviews/<date>-batch<N>-<topic>-codex.md`.

Expect `needs-rework`. Both reviews so far returned it. That is the process working.

---

## Stop and ask the user

Only for these. Everything else is your call — record it and keep going.

- You would **delete or overwrite** anything in a vault (not git-tracked, unrecoverable).
- A source is **outside its vault's subject** and you think it should move vaults.
- Two sources **contradict on something load-bearing** and you cannot represent both.
- `check_ingest.py` reports a failure you cannot fix.
- A vault's `CLAUDE.md` tells you to do something this runbook forbids.

Do **not** stop to ask whether a page is good enough, whether to use a concept or an
entity, or how to word something. Decide, note it in the log, move on.

---

## Anti-patterns

| Don't | Do |
|---|---|
| Write what you know about the topic | Write what the sources say about the topic |
| Drop "likely", "suggests", "on this dataset" | Keep the hedge — it is part of the claim |
| Resolve a disagreement between sources | Record both and say they disagree |
| Cite a source from a previous batch as evidence | Only sources in *this* batch's provenance count |
| Give every tool its own thin page | Fold related thin sources into one summary |
| Pad a page to look substantial | Short and accurate beats long and padded |
| Round the `generated` timestamp | Use the real UTC time |
| Leave `> SKELETON.` on a populated stage | Delete it and add the `Populated with` line |
| Create a second `## 2026-08-30` heading in the log | Add to the existing one |
| Skip `check_ingest.py` because it looks fine | Run it; it has caught things every time |

---

## Worked example

`projects/llm-corpus-expansion/ingest_batch3.py` is a complete, working batch script:
8 sources, 5 summaries, 2 paper pages, one stub filled. Read it before your first batch
and copy its shape. `ingest_batch1.py` and `ingest_batch2.py` show the same pattern at
4 sources.

For a finished example of the writing standard, read
`Wiki/concepts/Synthetic Data.md` in the LLM Fine-tuning vault — note how the ReST-EM
result carries its enabling condition ("on tasks where scalar feedback is available")
inline rather than being generalized, and how the Phi dispute is recorded as unresolved.

## Background, if you need it

- `INGEST-LEDGER.md` — what remains, batch order, standing decisions
- `handoff.md` — how the corpus got here
- `reviews/` — every past review, with the findings that shaped these rules
- Each vault's `CLAUDE.md` — the authoritative schema for that vault
