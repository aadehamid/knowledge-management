# Syncing Recall into the pipeline

How to find what is saved in [Recall](https://www.getrecall.ai/) but not in this
knowledge base, and route it in. Repeatable — run it whenever Recall has drifted ahead.

Last run 2026-08-30: 561 Recall cards → **50 new sources appended**, 22 of them CUDA.

---

## Why this is not a one-liner

The two systems store the same article under different URLs constantly. A naive
`comm -13` on raw URLs reported **123 sources missing**; the true figure was **64**, and
50 after dropping dead and out-of-scope links. Nearly half the first answer was noise.

Every false positive is a real cost: it means re-fetching and re-ingesting something the
corpus already holds, under a second URL, producing a duplicate wiki page.

So the work is mostly **URL identity**, and it is iterative: you find aliases, re-run, and
find more. Four rounds were needed the first time. `scripts/compare_recall.py` now carries
them all, so a re-run starts from that baseline.

---

## Procedure

### 1. Connect Recall

Registered once, user scope:

```bash
claude mcp add --transport http --scope user recall https://backend.getrecall.ai/mcp/
```

Then `/mcp` → **recall** → authorize in the browser, and **restart the session** (MCP
servers initialize at startup). Read-only; no API key.

### 2. Pull every card

```
mcp__recall__explore_kb   action=get_stats        # gives total_cards — your target
mcp__recall__filter_by_metadata                    # no args
```

**`filter_by_metadata` caps at 500 results and has no offset parameter.** With more than
500 cards you must page by date:

```
filter_by_metadata  date_to=<oldest date in the first page>
filter_by_metadata  date_from=... date_to=...      # fill any gap
```

Union the pages and check the count against `total_cards`. Cards with **no `source_url`**
(Recall's own notes and concept cards — 67 of 561 last time) cannot take part in a URL
diff. Exclude them explicitly; do not let them silently count as matches.

Write one URL per line to a file.

### 3. Diff

```bash
python3 scripts/compare_recall.py --recall recall_urls.txt --out recall-only.txt
```

It compares against **the DEVONthink inventory plus every `resources/sources/*/urls.txt`**
— the pipeline's real memory. Comparing against the inventory alone reports sources as
missing when they are already queued.

Output: the three-way split, Recall-only grouped by host, and the overlap broken down by
routed subject.

### 4. Hunt aliases — the part that matters

Take the Recall-only list and, **for each host with several entries**, check whether the
corpus holds the same pages under a different shape:

```bash
INV=projects/llm-corpus-expansion/data/devonthink-inventory.tsv
grep -io 'https://[^\t]*<host>[^\t]*' "$INV" | sed 's#.*/##' | sort -u    # corpus slugs
grep -i '<host>' recall-only.txt | sed 's#.*/##' | sort -u                # Recall slugs
```

**Matching final slugs with different parent paths means a site restructure, not new
content.** That is how the 12 "missing" Unsloth pages were caught.

Add each confirmed alias to `apply_aliases()` in `scripts/compare_recall.py`, re-run, and
repeat until the residual stops shrinking. Aliases found so far:

| Kind | Example |
|---|---|
| Docs host move + tree restructure | `docs.unsloth.ai/X` → `unsloth.ai/docs/<slug>` |
| Path rename | `deeplearning.ai/short-courses/` → `/courses/` |
| Product rename | `lightning.ai/studios/` → `/templates/` |
| Domain move | `e2eml.school` → `brandonrohrer.com`, `eigenfoo.xyz` → `georgeho.org` |
| Repo rename | `OpenAccess-AI-Collective/axolotl` → `axolotl-ai-cloud/axolotl` |
| Blog subdomain vs path | `blog.lancedb.com/X` ↔ `lancedb.com/blog/X` |
| Same article, two homes | `vllm.ai/blog/2025-09-05-anatomy-of-vllm` = `aleksagordic.com/blog/vllm` |
| Permalink style | `apeatling.com/2024/01/08/X` → `/articles/X` |

**Never add an alias you have not verified on both sides.** A wrong alias silently hides a
genuinely new source, which is worse than a duplicate — nothing will surface it again.

### 5. Route and append

For each survivor decide subject and `source_type`, then append to
`resources/sources/<subject>/urls.txt` under a dated comment header:

```
# --- from Recall knowledge base 2026-08-30 (22 sources) ---
https://... | Title with no pipe characters | blog
```

Drop, with a written reason, anything that is:

- an auth wall or login page (edX, Panopto, Greenhouse, site logins)
- a site homepage or index rather than an article
- a 404 that the corpus already holds at a pinned commit
- out of scope for **every** vault (last time: four data-engineering "semantic layer"
  posts — no vault covers that subject)

Validate before committing:

```bash
for f in resources/sources/*/urls.txt; do
  awk -F'|' -v F="$f" '!/^#/ && $0!~/^[ \t]*$/ { u=$1; gsub(/^[ \t]+|[ \t]+$/,"",u);
    if (u=="") print "EMPTY URL: "F": "$0;
    if (u!="" && u !~ /^https?:\/\//) print "BAD URL: "F": "$0;
    if (NF<3) print "MISSING FIELDS: "F": "$0 }' "$f"
done
```

Titles must not contain `|` — it is the field delimiter.

### 6. Convert and sync

The new lines are ordinary pipeline input from here:

```bash
.venv/bin/python scripts/convert_pdfs.py
NLM_SKIP=1 .venv/bin/python scripts/sync_to_vault.py
```

Then they enter the normal ingest backlog — update `INGEST-LEDGER.md` counts.

---

## What the 2026-08-30 run found

| | |
|---|---|
| Recall cards | 561 (493 unique resolvable resources) |
| Corpus | 536 unique resources |
| In both | 429 |
| **Recall only** | **64** → 50 appended, 14 dropped |
| Corpus only | 107 |

**The shape was the finding.** 22 of the 50 were GPU/CUDA/architecture material — NVIDIA
Ampere and Hopper deep-dives, the CUDA C++ Programming Guide, Modal's GPU glossary,
Microway and Cornell memory-hierarchy pages, Stanford CS149 lectures, *What Every
Programmer Should Know About Memory*. The `cuda` vault went from 15 queued sources to 37.

That is a systematic gap, not a random one: Recall had been accumulating the **hardware
layer** while the DEVONthink reading list accumulated the **model layer**. Worth checking
each time — the composition of the difference says more than its size.

Appended: cuda 22 · llm-finetuning 8 · llm-inference-optimization 7 · ml-foundations 5 ·
transformers 3 · ai-engineering 2 · document-ai-retrieval 2 · llm-landscape 1.

### Known issue, pre-existing

`kipp.ly/p/transformer-inference-arithmetic` appears in both
`llm-inference-optimization/urls.txt` (twice) and `transformers/urls.txt`. It predates
this sync. Harmless — the converter's stem assignment and the vault's URL guard both
handle it — but worth cleaning up when someone is next in those files.
