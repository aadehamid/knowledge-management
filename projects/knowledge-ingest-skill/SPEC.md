# Skill specification: `knowledge-ingest`

A design spec to hand to another agent, so it can merge this work with its own and
produce one skill. **This is not the skill — it is the brief for building it.**

Written by the agent that ran the knowledge-management pipeline: 718 sources, 8 vaults,
3 reviewed ingest batches, a Recall↔corpus reconciliation. Everything below is either a
mechanism that worked or a failure that was caught in production.

---

## 1. Should this be a skill? Yes — but only part of it

**In scope.** The reusable methodology: turning a list of sources into a verified,
interlinked knowledge bundle, with resumable batching and a quality gate that does not
depend on the author grading itself. That generalises to any corpus, any subject, any
model.

**Out of scope — keep in the repo, not the skill:**

- Hardcoded paths, vault names, subject lists
- `convert_pdfs.py` / `sync_to_vault.py` — pipeline-specific plumbing
- The OKF v0.2 schema itself, which is one bundle format among several

The line to hold: **a skill encodes the method and the gates; the repo holds the
configuration and the plumbing.** If the merged skill contains an absolute path, it has
crossed the line.

Everything the skill needs about a specific project should come from **one config file
the skill reads**, not from its own text. Proposed `knowledge-ingest.config.json`:

```json
{
  "bundles": [{"name": "...", "path": "...", "sources": "...", "shape": "concept|catalog"}],
  "converted_dir": "...",
  "validator": "scripts/test_okf_bundle.py",
  "reviewer": {"kind": "codex", "cmd": "..."},
  "ledger": "projects/.../INGEST-LEDGER.md"
}
```

---

## 2. What the skill is for

> Take N unprocessed sources in a corpus and turn them into wiki knowledge, in batches,
> with every claim traceable to a source and every batch independently reviewed.

**Frontmatter description** — this is the trigger text, so it must be precise about when
*not* to fire:

```yaml
name: knowledge-ingest
description: Turn source documents into an interlinked, provenance-checked knowledge
  bundle, batch by batch, with an independent cross-model review gate. Use when a corpus
  of fetched sources must become wiki pages, when resuming a partially-ingested backlog,
  or when auditing an existing bundle for provenance and consistency. Not for one-off
  summarisation, not for fetching or converting sources, not for general note-taking.
argument-hint: "[ingest [n batches] | audit <bundle> | review <bundle> | status]"
```

---

## 3. The four invariants — the whole point of the skill

Everything else is convenience. These are what a weaker or hurried agent gets wrong, and
each is stated with the production failure that produced it.

### I1. Provenance: every factual sentence traces to a line in a source *for that batch*

Not "true", not "well known". Outside knowledge is invention however correct it is.

> Written: "FSDP shards parameters, gradients and optimizer state."
> Reality: `optimizer state` appears **0 times** in the four permitted sources. The author
> knew it from elsewhere. Caught by review, not by self-QA.

### I2. Hedges are part of the claim

If the source says *likely*, *suggests*, *on this dataset*, *assuming X*, *we found the
most support for* — those words survive into the page.

> Written: "the gap occurs **because** of BA optimization dynamics."
> Source: "The **likely reason** is ... **on this dataset**."
>
> Written: "the paper's **answer** is the generation-verification gap."
> Source: the paper "found the **most support** for" it, among competing hypotheses.

### I3. Disagreements are recorded, never resolved

Two sources conflicting is signal. Flattening it destroys the most valuable thing in the
corpus.

> Two guides gave different `alpha` advice. The author wrote that they were "the same
> point in a two-parameter space — neither is wrong." The source said the second maps to a
> *different* point and drew no equivalence verdict. The synthesis was the author's,
> presented as the source's.

### I4. Verify against the system before acting on it

Titles, keywords and plans all mislead. Check the artefact.

> A source titled "build a ChatGPT" was routed out of the inference vault. It was in fact
> cited by that vault's core `KV Cache` page as the reference implementation. One `grep`
> would have caught it — and did, before the move.

---

## 4. The gates

### Gate A — mechanical self-check (automated, every batch)

Ship `check_ingest.py` (in this repo) as a skill script, parameterised by bundle path.
Each check exists because a review caught that exact failure:

| Check | Failure it catches |
|---|---|
| Bundle validator passthrough | schema drift |
| Every relative link resolves | dangling links |
| `wiki_refs` reciprocal **both ways** | a page cites a source the source does not list |
| Trust stamps present and **not in the future** | stamps written as a rounded guess — `03:00:00Z` on a file written `02:34:40Z` |
| Populated stages not still marked SKELETON | stale scaffolding |
| Index coverage | orphan pages |
| Log: frontmatter-free, unique dates, newest first | duplicated date headings |

**It must print that it checks structure and never fidelity.** A clean run is not a
quality signal, and a weaker agent will read it as one.

### Gate B — independent review (every batch, or every third at most)

**A different agent instance, ideally a different model family.** Never self-review.

Run report-only. Enforce it structurally where possible — the Codex runtime's
`write: false` beats an instruction not to edit.

The brief must contain: the schema; **every source, labelled *the only permitted
evidence***; the explicit file list; check priorities in order; report-only; requested
output format; and — most valuable — **the claims the author is least sure of, named
explicitly.**

> Both reviews caught things the author had asked them to attack. Naming your own weak
> points is the highest-yield line in the brief.

Then: **verify each finding against the source before applying it.** Reviewers are
sometimes wrong; document overrides rather than deleting correct content. Apply every
HIGH and MEDIUM, re-run Gate A, record reviewer and finding counts in the log, archive the
review.

Expect `needs-rework`. Two of two reviews returned it, 8 findings each, all valid. That is
the gate working.

---

## 5. Resumability: the ledger

One file, updated at the end of every batch, that lets a cold agent resume without
guessing: per-bundle counts, batch order, per-batch status, standing decisions, and **the
snippet to recompute the counts**.

> Rule: *recompute the numbers, never hand-edit them.* A ledger trusted while stale is
> worse than no ledger.

Definition of processed: **a source is processed when some page in the bundle cites its
filename.** Frontmatter flags drift; the citation is the fact.

---

## 6. Bundle shapes — not every corpus deserves a wiki

A real decision the skill should force, not assume:

- **concept** — entities, concepts, curriculum. For durable subject matter.
- **catalog** — index, overview, log only. For dated ephemera.

> Diagnostic that worked: term-frequency the corpus. One subject's top terms (rlhf,
> benchmark, sft) were all *borrowed from other subjects* — the signature of a feed, not a
> domain. It was built catalog-shaped, and the deviation documented in its own schema file
> rather than left as an unexplained gap.

Same principle for stubs: **derive them from the corpus's own term frequencies, never from
a template.** A cloned list produces stubs nobody fills — this repo still carries one from
May. 38 corpus-derived stubs beat ~95 generic ones.

---

## 7. Traps worth encoding as checks

Each cost real time here.

| Trap | Guard |
|---|---|
| **Filename collisions** — `get_url_stem` is not injective; `Path.stem` truncates dotted names, so `llama3.3`→`llama3` and every arXiv `2401.*`→`2401`. Second file silently overwrites the first, taking its backlinks. | Resolve stems against a stem→owner map; first claimant keeps the name |
| **Unquoted YAML colons** — `description: Foo: bar` parses as a mapping and fails validation | Quote-on-demand helper; broke twice |
| **Future timestamps** — a stamp claiming a time after the file's mtime is worse than none | Compare stamp to mtime |
| **Silent converter degradation** — a missing optional extra made every video a 200-word shell instead of a 13,000-word transcript. Plausible-looking output, 98% of content missing. | Assert expected shape; pin the extra with a comment saying why |
| **URL identity** — a naive diff said 123 sources missing; the truth was 64. Host moves, path renames, repo renames, `/live/` vs `?v=` | Alias table, each entry verified on both sides |
| **A delete that undoes itself** — removing vault files without removing their `urls.txt` line means the next sync restores them | Trace what could reverse the change |

---

## 8. Proposed structure

```
knowledge-ingest/
├── SKILL.md                    # routing, the four invariants, the loop, stop conditions
├── references/
│   ├── ingest.md               # the batch loop in detail, page templates
│   ├── review.md               # brief construction, finding resolution, override policy
│   ├── bootstrap.md            # new bundle: shape choice, corpus-derived stubs
│   └── audit.md                # auditing an existing bundle
└── scripts/
    ├── check_ingest.py         # Gate A, --bundle parameterised
    ├── ledger.py               # recompute counts, update status
    └── url_identity.py         # normalisation + alias table (reusable beyond ingest)
```

Keep `SKILL.md` short. Put procedure in `references/`, loaded on demand.

---

## 9. Merging with the parallel project

**Assume theirs is right where it disagrees on mechanism, and hold this line where it
disagrees on gates.** Mechanism is taste; the gates are what stop silent corruption.

Reconcile in this order:

1. **Config schema first.** Both projects must describe themselves in one file. Agree that
   before anything else; everything downstream depends on it.
2. **Invariants — union, not intersection.** If their project learned a failure mode this
   one did not, add it. Do not drop I1–I4 for brevity; each has a production incident
   behind it.
3. **Gate A checks — union.** Every check should carry a one-line note of the failure it
   catches. A check without that note is a check someone will delete later.
4. **Gate B — one policy.** Decide review cadence together. If either project's authoring
   model is weaker, cadence should follow the weaker one.
5. **Where you genuinely disagree, put both behind a config flag rather than arguing** —
   e.g. `bundle.shape`, `review.cadence`, `stubs.source: corpus|template`.

**Expected conflicts, with a recommendation:**

| Likely conflict | Recommendation |
|---|---|
| Bundle schema (OKF here, probably something else there) | Abstract to a `validator` command in config; the skill never hardcodes a schema |
| Batch size | Config value. 4 gave tight reviews; 8–12 cut overhead. Both defensible |
| Review cadence | Config, default every batch, weaken only deliberately |
| Whether summaries are 1:1 with sources | Config off; folding related thin sources into one summary was right here |
| Ledger format | Theirs, if it recomputes. Only requirement: never hand-edited |

**What must not be negotiated away:** provenance-per-batch, hedge preservation, an
independent reviewer that is a different instance, and a self-check that announces it does
not verify fidelity.

---

## 10. How to actually build it — use `skill-creator`

Do **not** hand-write the skill from this spec. The official `skill-creator` plugin exists
for this and adds the thing a hand-written skill never gets: **eval-driven iteration on the
description**, which is what determines whether the skill triggers at the right moment.

It is not installed by default:

```
/plugin marketplace add claude-plugins-official      # if not already added
/plugin install skill-creator@claude-plugins-official
/reload-plugins
```

Its loop is: decide what the skill does → draft → write test prompts → run them → evaluate
→ rewrite → repeat, then run its description optimiser.

**This spec is the input to its first step, not a replacement for it.** `skill-creator`
knows how to shape and test a skill; it does not know what this one should contain. Hand it
this document as the "roughly how it should do it" and let it drive from there.

Test prompts worth putting in its eval set — the trigger boundary matters more than the
body, and half of these must *not* fire:

| Prompt | Should trigger? |
|---|---|
| "Ingest the next batch of sources into the CUDA vault" | yes |
| "Resume the ingest — where did we leave off?" | yes |
| "Audit this bundle for provenance and broken links" | yes |
| "Review the last three batches before I continue" | yes |
| "Summarise this PDF for me" | **no** — one-off summarisation |
| "Fetch these 20 URLs and convert them to markdown" | **no** — that is the pipeline, not the ingest |
| "Take notes on this meeting" | **no** — general note-taking |
| "What does my knowledge base say about KV caching?" | **no** — that is a query, not an ingest |

The last four are the ones that matter. A skill this broad will over-trigger unless the
description is explicit about what it is not for.

## 11. Evidence base

Available in this repo for the merging agent to read:

- `projects/llm-corpus-expansion/INGEST-RUNBOOK.md` — the operator's manual this spec generalises
- `projects/llm-corpus-expansion/reviews/` — both reviews verbatim, 16 findings
- `projects/llm-corpus-expansion/INGEST-LEDGER.md` — the ledger pattern in use
- `projects/llm-corpus-expansion/RECALL-SYNC.md` — the URL-identity work
- `scripts/check_ingest.py`, `scripts/bootstrap_vault.py`, `scripts/compare_recall.py`
- `scripts/test_stem_collisions.py` — the collision guard and its regression tests

Scale it was proven at: 718 sources, 8 bundles, 3 reviewed batches, 0 validator failures
across all bundles at the end.

**Honest limit:** the review gate is the only thing that caught fidelity errors. Gate A
never did and never will. If the merged skill makes Gate B optional, it becomes a
formatting tool.
