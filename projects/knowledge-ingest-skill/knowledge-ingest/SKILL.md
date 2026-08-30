---
name: knowledge-ingest
description: Turn fetched source documents into an interlinked, provenance-checked knowledge bundle — wiki pages, concept pages and a curriculum — in resumable batches with an independent cross-model review gate. Use this whenever a corpus of raw sources needs to become durable knowledge: ingesting a batch into a vault or bundle, resuming a partially-processed backlog, bootstrapping a new subject bundle, or auditing an existing one for provenance, dead links and consistency. Use it even when the user just says "process the rest of these sources" or "pick up where we left off" without naming a method. Not for one-off summarisation of a single document, not for fetching or converting sources into markdown, and not for answering questions from a knowledge base that already exists.
---

# Knowledge ingest

Turning a pile of fetched sources into knowledge that compounds. The mechanics are
ordinary — read sources, write pages, link them. What makes it hard is that the failure
mode is silent: a page that reads beautifully and says something its source never said is
indistinguishable from a good page until someone checks.

Everything here exists to make that failure visible.

## Orient first

Before touching anything, find out where the work stands. Read the project's ledger
(config `ledger`) and the target bundle's own schema file if it has one. Recompute the
counts rather than trusting the numbers written down — a ledger believed while stale is
worse than none.

A source counts as **processed when some page in the bundle cites its filename**.
Frontmatter flags drift; the citation is the fact.

## Configuration

Read `knowledge-ingest.config.json` from the project root. Never hardcode paths — if you
find yourself typing an absolute path, the project should be describing it instead.

```json
{
  "bundles": [
    {"name": "...", "path": "...", "sources": "...", "shape": "concept|catalog"}
  ],
  "converted_dir": "...",
  "validator": "scripts/test_okf_bundle.py",
  "reviewer": {"kind": "codex|subagent", "cmd": "..."},
  "ledger": "..."
}
```

If no config exists, ask for the bundle path and source directory rather than guessing,
and offer to write the config so the next run needs no questions.

---

## The four invariants

These are the skill. Everything else is procedure.

### 1. Every factual sentence traces to a line in a source *for this batch*

Not "true". Not "well known". **In the source, in this batch.** Outside knowledge is
invention however correct it happens to be — the reader's trust comes from the provenance
chain, and a true-but-unsourced sentence breaks that chain as thoroughly as a false one.

A page once said "FSDP shards parameters, gradients and optimizer state." Accurate, and
`optimizer state` appeared **zero times** across the four permitted sources. The author
knew it from elsewhere. Self-checking never caught it; an independent reviewer did.

The practical test: *would I write this sentence without the source open?* If yes, delete
it or go find the line.

### 2. Hedges belong to the claim

When a source says *likely*, *suggests*, *on this dataset*, *assuming X*, *we found the
most support for* — those words are load-bearing. Dropping them converts a researcher's
careful hypothesis into your confident assertion, and the reader has no way to tell.

- Written "the gap occurs **because** of BA dynamics"; the source said "the **likely
  reason** is ... **on this dataset**."
- Written "the paper's **answer** is the generation-verification gap"; the paper "found
  the **most support** for" it, among competing hypotheses it tested.

Both slipped past self-review. Both were caught cold by a different model.

### 3. Record disagreements; never resolve them

Two sources conflicting is the most valuable thing a corpus contains — it marks where the
field is unsettled. Flattening it into one confident answer destroys information nobody
can recover later.

Present both, name the sources, say plainly that they differ. If one has better evidence,
say why *and* keep the other. Resist the pull to synthesise: a synthesis you invented,
sitting in a page whose frontmatter cites two sources, reads exactly like something those
sources said.

### 4. Verify against the system before acting on it

Titles mislead. Keyword matches mislead. Plans go stale while you execute them.

A source titled "build a ChatGPT" was about to be moved out of an inference bundle as
off-topic. One grep showed the bundle's core `KV Cache` page cited it as the reference
implementation. The move would have broken a live citation to satisfy a guess about a
title.

Before relocating, deleting or reclassifying anything: grep for inbound references. Then
ask what could undo the change — deleting a page whose source line still exists means the
next sync quietly restores it.

---

## The loop

Full procedure, page templates and frontmatter rules: `references/ingest.md`. In outline:

1. **Pick a cluster** — 8–12 sources on one topic. Coherence matters more than count: a
   batch should produce pages that link to each other. Prefer sources with real substance;
   a 200-word page shell has little to give.
2. **Read every source completely.** Not the title, not a skim. You will be reviewed
   against these files. While reading, note the specific numbers, the hedged claims, and
   anything that contradicts another source in the batch.
3. **Write the pages** — summaries, concepts, entities, papers. Fill existing stubs before
   creating new ones.
4. **Wire it up** — index entries, back-references on each source, curriculum stage, log
   entry. A page nothing links to is invisible; a source nothing cites is unprocessed.
5. **Gate A**, then **Gate B**. Both below.
6. **Update the ledger** by recomputing, and commit.

If a source is genuinely thin, write a short page. Three honest lines beat twenty padded
ones. Where the sources don't cover something the topic needs, *say so on the page* — "the
sources here do not explain X" is useful to a reader and honest about the gap.

---

## Gate A — mechanical self-check

Run `scripts/check_ingest.py <bundle>` after every batch. It verifies schema conformance,
link resolution, reference reciprocity in both directions, trust stamps (including the
lovely failure mode of a stamp dated in the future), stale scaffolding markers, index
coverage and log ordering.

Every check exists because a review caught that exact failure once.

**A clean run means the structure is sound and nothing more.** It cannot tell you whether
a sentence is in its source. Reading "Clean" as "correct" is the most likely way for this
skill to fail in practice, which is why the script says so in its own output.

## Gate B — independent review

**A different agent instance, ideally a different model family.** Never self-review: the
agent that wrote a page is worst-placed to notice what it invented, because the invented
part felt like knowing.

Run it report-only. Where the runtime can enforce that structurally — a reviewer with no
write access — prefer that to an instruction not to edit.

The brief needs: the schema being checked against; **every source, labelled as the only
permitted evidence**; the explicit list of files created and edited; the check priorities;
report-only; and the requested output shape (findings tagged HIGH/MEDIUM/LOW with file and
line, then one verdict).

Then the highest-yield line in the procedure: **name the claims you are least sure of and
ask the reviewer to attack them.** Reviewers catch what they are pointed at. Listing your
own soft spots costs a paragraph and returns most of the value.

Resolving findings is yours, not the reviewer's:

- Verify each finding against the source before applying it. Reviewers are sometimes
  wrong, and deleting correct content to satisfy a false positive is a real loss.
- Document overrides rather than silently ignoring them.
- Apply every HIGH and MEDIUM, re-run Gate A, record which reviewer was used and how many
  findings were applied, archive the review.

Expect `needs-rework`. That is the gate working, not a setback.

Brief construction and worked examples: `references/review.md`.

---

## Bundle shape — not every corpus deserves a wiki

Two shapes, and choosing deliberately matters more than choosing correctly:

- **concept** — entities, concepts, curriculum. For durable subject matter.
- **catalog** — index, overview, log only. For dated ephemera: releases, announcements,
  news.

A useful diagnostic when unsure: run term frequencies over the corpus. If a subject's top
terms are all *borrowed from other subjects*, it is a feed rather than a domain, and a
concept wiki over it goes stale faster than it can be written. Build it catalog-shaped and
document why, so the deviation reads as a decision rather than an omission.

Same principle for stub pages when bootstrapping: **derive them from the corpus's own term
frequencies, never from a template.** A cloned list produces stubs nobody fills, which make
the graph look like coverage it does not have. See `references/bootstrap.md`.

---

## Traps

Each of these cost real time. Guard against them by default.

| Trap | What happens |
|---|---|
| Filename collisions | Stem functions are rarely injective, and `Path.stem` truncates dotted names — `llama3.3`→`llama3`, every arXiv `2401.*`→`2401`. The second file silently overwrites the first and inherits its backlinks. Resolve names against a claim map; first claimant keeps the name. |
| Unquoted YAML colons | `description: Foo: bar` parses as a mapping, not a string, and fails validation. Quote on demand. |
| Timestamps that are guesses | A stamp claiming a time after the file's own mtime is worse than no stamp. Use the real clock. |
| Silent converter degradation | A missing optional dependency turned every video into a 200-word shell instead of a 13,000-word transcript. Plausible output, 98% of the content missing. Assert the expected shape. |
| URL identity | The same article lives at many URLs — host moves, path renames, repo renames, `/live/` vs `?v=`. A naive diff once reported 123 missing sources where 64 was the truth. |
| Deletes that undo themselves | Removing a page without removing the source line that generates it means the next sync restores it. Trace what could reverse the change. |

---

## When to stop and ask

Narrow, because an agent that halts on every judgement call never finishes a backlog:

- You would delete or overwrite anything not version-controlled.
- A source clearly belongs to a different bundle.
- Two sources contradict on something load-bearing and you cannot represent both.
- Gate A reports a failure you cannot fix.
- A bundle's own schema contradicts this skill.

Everything else is yours to decide. Decide it, record it in the log, keep going.

---

## Other modes

- **audit** an existing bundle without ingesting: `references/audit.md`
- **bootstrap** a new bundle before its first ingest: `references/bootstrap.md`
- **status**: recompute ledger counts and report what remains, changing nothing
